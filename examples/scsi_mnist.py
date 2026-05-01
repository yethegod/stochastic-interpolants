"""Minimal SCSI-style random masking experiment on MNIST.

Training batches contain only corrupted observations ``(y, mask)``. Clean
MNIST images are used to build the corrupted dataset and for image restoration
metrics on the held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(Path(tempfile.gettempdir()) / "torch"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


Mode = Literal["ode", "sde"]


@dataclass(frozen=True)
class ObservedBatch:
    """Training batch containing no clean samples."""

    y: torch.Tensor
    mask: torch.Tensor


class TimeEmbedding(torch.nn.Module):
    def __init__(self, dim: int, n_frequencies: int) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        in_dim = 1 + 2 * n_frequencies
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        angles = 2.0 * math.pi * t * self.frequencies[None, :]
        features = torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.net(features)


def make_group_norm(channels: int) -> torch.nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return torch.nn.GroupNorm(groups, channels)


class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = torch.nn.Linear(time_dim, out_channels)
        self.norm2 = make_group_norm(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.skip = torch.nn.Identity()
        else:
            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(torch.nn.Module):
    """Small UNet for ``[x_t, y, mask]`` conditioned image velocity."""

    def __init__(self, base_channels: int = 64, time_dim: int = 128, time_frequencies: int = 8) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, time_frequencies)
        self.in_proj = torch.nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResConvBlock(base_channels, base_channels, time_dim)
        self.down1 = torch.nn.Conv2d(base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResConvBlock(2 * base_channels, 2 * base_channels, time_dim)
        self.down2 = torch.nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResConvBlock(4 * base_channels, 4 * base_channels, time_dim)
        self.mid2 = ResConvBlock(4 * base_channels, 4 * base_channels, time_dim)
        self.up1 = torch.nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResConvBlock(4 * base_channels, 2 * base_channels, time_dim)
        self.up2 = torch.nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResConvBlock(2 * base_channels, base_channels, time_dim)
        self.out_norm = make_group_norm(base_channels)
        self.out = torch.nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        h0 = self.in_proj(torch.cat([x_t, y, mask], dim=1))
        e1 = self.enc1(h0, time_emb)
        e2 = self.enc2(self.down1(e1), time_emb)
        mid = self.mid2(self.mid1(self.down2(e2), time_emb), time_emb)
        d1 = self.dec1(torch.cat([self.up1(mid), e2], dim=1), time_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1), time_emb)
        return self.out(F.silu(self.out_norm(d2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--reverse-steps", type=int, default=64)
    parser.add_argument("--samples-per-pair", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.5, help="Pixel masking probability.")
    parser.add_argument("--sigma-n", type=float, default=0.1, help="Observed-pixel noise.")
    parser.add_argument("--recorrupt-prob", type=float, default=0.9)
    parser.add_argument("--gamma-scale", type=float, default=0.05)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--time-frequencies", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scsi_mnist"))
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short finite-loss/metric smoke test for both ODE and SDE modes.",
    )
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.steps = 3
    args.batch_size = 8
    args.train_size = 128
    args.metric_samples = 8
    args.eval_every = 1
    args.reverse_steps = 4
    args.base_channels = 16
    args.time_dim = 32
    args.time_frequencies = 4
    args.skip_lpips = True
    args.output_dir = args.output_dir / "smoke"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def load_mnist_images(
    root: Path,
    train: bool,
    limit: int | None,
    download: bool,
    device: torch.device,
) -> torch.Tensor:
    dataset = MNIST(root=str(root), train=train, download=download)
    images = dataset.data.float().unsqueeze(1) / 255.0
    if limit is not None:
        images = images[: min(limit, images.shape[0])]
    return images.to(device)


def corrupt_observations(
    x: torch.Tensor,
    rho: float,
    sigma_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.rand_like(x) < (1.0 - rho)).float()
    eps = torch.randn_like(x)
    eta = torch.randn_like(x)
    y = mask * (x + sigma_n * eps) + (1.0 - mask) * eta
    return y, mask


def sample_observed_batch(y: torch.Tensor, mask: torch.Tensor, batch_size: int) -> ObservedBatch:
    idx = torch.randint(y.shape[0], (batch_size,), device=y.device)
    return ObservedBatch(y=y[idx], mask=mask[idx])


def broadcast_time(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (like.ndim - 1)))


def gamma(t: torch.Tensor, like: torch.Tensor, scale: float) -> torch.Tensor:
    tb = broadcast_time(t, like)
    return scale * tb * (1.0 - tb)


def gamma_dot(t: torch.Tensor, like: torch.Tensor, scale: float) -> torch.Tensor:
    tb = broadcast_time(t, like)
    return scale * (1.0 - 2.0 * tb)


def epsilon_t(t: torch.Tensor, like: torch.Tensor, scale: float) -> torch.Tensor:
    return gamma(t, like, scale)


@torch.no_grad()
def restore(
    model: ConditionalUNet,
    y: torch.Tensor,
    mask: torch.Tensor,
    mode: Mode,
    reverse_steps: int,
    gamma_scale: float,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x = y.clone()
    batch = y.shape[0]
    dt = 1.0 / reverse_steps
    for i in range(reverse_steps):
        t_value = 1.0 - i * dt
        t = torch.full((batch,), t_value, device=y.device, dtype=y.dtype)
        drift = model(x, t, y, mask)
        x = x - dt * drift
        if mode == "sde":
            t_mid = torch.full((batch,), max(t_value - 0.5 * dt, 0.0), device=y.device, dtype=y.dtype)
            noise_std = torch.sqrt(2.0 * epsilon_t(t_mid, x, gamma_scale) * dt)
            x = x + noise_std * torch.randn_like(x)
    if was_training:
        model.train()
    return x


def recorrupt_observations(
    x_hat: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    sigma_n: float,
    recorrupt_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(x_hat)
    eta = torch.randn_like(x_hat)
    y_tilde = mask * (x_hat + sigma_n * eps) + (1.0 - mask) * eta
    use_tilde = (torch.rand(x_hat.shape[0], 1, 1, 1, device=x_hat.device) < recorrupt_prob).float()
    y_bar = use_tilde * y_tilde + (1.0 - use_tilde) * y
    return y_bar, mask


def make_interpolant_training_data(
    model: ConditionalUNet,
    batch: ObservedBatch,
    mode: Mode,
    reverse_steps: int,
    samples_per_pair: int,
    sigma_n: float,
    recorrupt_prob: float,
    gamma_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        x_hat = restore(model, batch.y, batch.mask, mode, reverse_steps, gamma_scale)
        y_bar, cond_mask = recorrupt_observations(x_hat, batch.y, batch.mask, sigma_n, recorrupt_prob)
        x0 = x_hat.detach()
        x1 = y_bar.detach()

    if samples_per_pair > 1:
        x0 = x0.repeat_interleave(samples_per_pair, dim=0)
        x1 = x1.repeat_interleave(samples_per_pair, dim=0)
        y_cond = batch.y.repeat_interleave(samples_per_pair, dim=0)
        mask_cond = cond_mask.repeat_interleave(samples_per_pair, dim=0)
    else:
        y_cond = batch.y
        mask_cond = cond_mask

    n = x0.shape[0]
    t = 1e-3 + (1.0 - 2e-3) * torch.rand(n, device=x0.device)
    t_view = broadcast_time(t, x0)
    base_velocity = x1 - x0
    if mode == "ode":
        x_t = (1.0 - t_view) * x0 + t_view * x1
        target = base_velocity
    else:
        z = torch.randn_like(x0)
        gam = gamma(t, x0, gamma_scale)
        gdot = gamma_dot(t, x0, gamma_scale)
        eps = epsilon_t(t, x0, gamma_scale)
        eps_over_gamma = torch.where(gam.abs() > 1e-8, eps / gam, torch.zeros_like(gam))
        x_t = (1.0 - t_view) * x0 + t_view * x1 + gam * z
        target = base_velocity + (gdot + eps_over_gamma) * z
    return x_t, t, y_cond, mask_cond, target


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def restore_in_batches(
    model: ConditionalUNet,
    y: torch.Tensor,
    mask: torch.Tensor,
    mode: Mode,
    reverse_steps: int,
    gamma_scale: float,
    batch_size: int,
) -> torch.Tensor:
    restored = []
    for start in range(0, y.shape[0], batch_size):
        stop = min(start + batch_size, y.shape[0])
        restored.append(restore(model, y[start:stop], mask[start:stop], mode, reverse_steps, gamma_scale).cpu())
    return torch.cat(restored, dim=0)


def mse_metric(restored: torch.Tensor, clean: torch.Tensor) -> float:
    return float(torch.mean((restored - clean) ** 2).cpu())


def psnr_metric(restored: torch.Tensor, clean: torch.Tensor) -> float:
    mse = torch.mean((restored - clean) ** 2).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).cpu())


def gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).reshape(1, 1, size, size)
    return window


def ssim_metric(restored: torch.Tensor, clean: torch.Tensor) -> float:
    restored = restored.clamp(0.0, 1.0)
    clean = clean.clamp(0.0, 1.0)
    window = gaussian_window(11, 1.5, restored.device, restored.dtype)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.conv2d(restored, window, padding=5)
    mu_y = F.conv2d(clean, window, padding=5)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(restored * restored, window, padding=5) - mu_x2
    sigma_y = F.conv2d(clean * clean, window, padding=5) - mu_y2
    sigma_xy = F.conv2d(restored * clean, window, padding=5) - mu_xy
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean().cpu())


class LPIPSEvaluator:
    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install lpips or pass --skip-lpips to skip LPIPS evaluation.") from exc
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, restored: torch.Tensor, clean: torch.Tensor) -> float:
        restored = restored.to(next(self.model.parameters()).device)
        clean = clean.to(restored.device)
        restored_rgb = F.interpolate(restored.clamp(0, 1).repeat(1, 3, 1, 1), size=(64, 64), mode="bilinear")
        clean_rgb = F.interpolate(clean.clamp(0, 1).repeat(1, 3, 1, 1), size=(64, 64), mode="bilinear")
        values = self.model(restored_rgb * 2.0 - 1.0, clean_rgb * 2.0 - 1.0)
        return float(values.mean().cpu())


@torch.no_grad()
def evaluate(
    model: ConditionalUNet,
    clean_eval: torch.Tensor,
    y_eval: torch.Tensor,
    mask_eval: torch.Tensor,
    mode: Mode,
    reverse_steps: int,
    gamma_scale: float,
    batch_size: int,
    lpips_eval: LPIPSEvaluator | None,
) -> tuple[dict[str, float], torch.Tensor]:
    restored = restore_in_batches(model, y_eval, mask_eval, mode, reverse_steps, gamma_scale, batch_size)
    clean_cpu = clean_eval.detach().cpu()
    restored_clamped = restored.clamp(0.0, 1.0)
    metrics = {
        "mse": mse_metric(restored_clamped, clean_cpu),
        "psnr": psnr_metric(restored_clamped, clean_cpu),
        "ssim": ssim_metric(restored_clamped, clean_cpu),
    }
    if lpips_eval is not None:
        metrics["lpips"] = lpips_eval(restored_clamped, clean_cpu)
    return metrics, restored_clamped


def save_image_grid(
    clean: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    restored: torch.Tensor,
    path: Path,
    max_images: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_images, clean.shape[0])
    rows = [
        ("clean", clean[:count].detach().cpu().clamp(0, 1)),
        ("corrupted", observed[:count].detach().cpu().clamp(0, 1)),
        ("mask", mask[:count].detach().cpu().clamp(0, 1)),
        ("restored", restored[:count].detach().cpu().clamp(0, 1)),
    ]
    fig, axes = plt.subplots(len(rows), count, figsize=(count * 1.1, len(rows) * 1.25), constrained_layout=True)
    for row_idx, (label, values) in enumerate(rows):
        for col_idx in range(count):
            ax = axes[row_idx, col_idx] if count > 1 else axes[row_idx]
            ax.imshow(values[col_idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def assert_smoke_invariants(
    y_train: torch.Tensor,
    mask_train: torch.Tensor,
    sigma_n: float,
    recorrupt_prob: float,
) -> None:
    batch = sample_observed_batch(y_train, mask_train, batch_size=min(8, y_train.shape[0]))
    if set(batch.__dataclass_fields__.keys()) != {"y", "mask"}:
        raise AssertionError("Training batch must contain only y and mask.")
    x_hat = torch.randn_like(batch.y)
    _, returned_mask = recorrupt_observations(x_hat, batch.y, batch.mask, sigma_n, recorrupt_prob)
    if returned_mask is not batch.mask:
        raise AssertionError("Re-corruption must keep conditioning on the original mask object.")
    if not torch.equal(returned_mask, batch.mask):
        raise AssertionError("Re-corruption changed mask values.")


def train_once(args: argparse.Namespace, mode: Mode) -> dict[str, object]:
    set_seed(args.seed + (0 if mode == "ode" else 10_000))
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_clean = load_mnist_images(args.data_dir, train=True, limit=args.train_size, download=args.download, device=device)
    y_train, mask_train = corrupt_observations(train_clean, args.rho, args.sigma_n)
    del train_clean

    clean_eval = load_mnist_images(
        args.data_dir,
        train=False,
        limit=args.metric_samples,
        download=args.download,
        device=device,
    )
    y_eval, mask_eval = corrupt_observations(clean_eval, args.rho, args.sigma_n)

    if args.smoke:
        assert_smoke_invariants(y_train, mask_train, args.sigma_n, args.recorrupt_prob)

    model = ConditionalUNet(
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        time_frequencies=args.time_frequencies,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_steps)
    lpips_eval = None if args.skip_lpips else LPIPSEvaluator(device)
    selection_metric = "mse" if args.skip_lpips else "lpips"

    best_metric = float("inf")
    step_of_best = 0
    final_metrics: dict[str, float] = {}
    final_restored = None
    final_loss = float("nan")

    for step in range(1, args.steps + 1):
        model.train()
        batch = sample_observed_batch(y_train, mask_train, args.batch_size)
        x_t, t, y_cond, mask_cond, target = make_interpolant_training_data(
            model=model,
            batch=batch,
            mode=mode,
            reverse_steps=args.reverse_steps,
            samples_per_pair=args.samples_per_pair,
            sigma_n=args.sigma_n,
            recorrupt_prob=args.recorrupt_prob,
            gamma_scale=args.gamma_scale,
        )
        pred = model(x_t, t, y_cond, mask_cond)
        loss = ((pred - target) ** 2).flatten(1).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach().cpu())

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_eval:
            final_metrics, final_restored = evaluate(
                model=model,
                clean_eval=clean_eval,
                y_eval=y_eval,
                mask_eval=mask_eval,
                mode=mode,
                reverse_steps=args.reverse_steps,
                gamma_scale=args.gamma_scale,
                batch_size=args.batch_size,
                lpips_eval=lpips_eval,
            )
            for name, value in final_metrics.items():
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite {name} at step {step}: {value}")
            current_metric = final_metrics[selection_metric]
            if current_metric < best_metric:
                best_metric = current_metric
                step_of_best = step
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "step": step,
                        "loss": final_loss,
                        "metrics": final_metrics,
                        "best_metric": best_metric,
                        "selection_metric": selection_metric,
                    }
                ),
                flush=True,
            )

    if final_restored is None:
        final_metrics, final_restored = evaluate(
            model=model,
            clean_eval=clean_eval,
            y_eval=y_eval,
            mask_eval=mask_eval,
            mode=mode,
            reverse_steps=args.reverse_steps,
            gamma_scale=args.gamma_scale,
            batch_size=args.batch_size,
            lpips_eval=lpips_eval,
        )

    grid_path = None
    if not args.no_plot:
        grid_path = args.output_dir / mode / "grid.png"
        save_image_grid(clean_eval, y_eval, mask_eval, final_restored.to(clean_eval), grid_path)

    result = {
        "mode": mode,
        "final_loss": final_loss,
        "final_metrics": final_metrics,
        "selection_metric": selection_metric,
        "best_metric": best_metric,
        "step_of_best_metric": step_of_best,
        "grid_path": str(grid_path) if grid_path is not None else None,
        "config": vars(args) | {"mode": mode, "output_dir": str(args.output_dir), "data_dir": str(args.data_dir)},
    }
    result_path = args.output_dir / mode / "metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_overrides(args)
        modes: list[Mode] = ["ode", "sde"]
    else:
        modes = [args.mode]

    results = [train_once(args, mode) for mode in modes]
    summary = {
        "final_metrics": results[-1]["final_metrics"],
        "selection_metric": results[-1]["selection_metric"],
        "best_metric": results[-1]["best_metric"],
        "step_of_best_metric": results[-1]["step_of_best_metric"],
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
