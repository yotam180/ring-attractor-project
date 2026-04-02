"""
Training loop for autonomous RNN on ring attractor data.

Core idea: teacher-force the first K steps to seed the hidden state, then
let the RNN run autonomously.  Loss is computed only on the autonomous
phase, forcing the network to embed attractor dynamics in its recurrent
weights rather than routing information through the input pathway.

Usage
-----
    from src.train.training import train, TrainingConfig
    result = train("data/ring_attractor_dataset.npz", TrainingConfig())

Or from the command line:
    python -m src.train.training [data_path] [--model lowrank] [--epochs 2000] ...
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import create_model
from .dataset import RingAttractorDataset


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    # Model
    model_type: str = "vanilla"   # "vanilla" or "lowrank"
    hidden_dim: int = 100
    alpha: float = 0.5            # leak rate — 1 bin ~ 0.5 tau
    rank: int = 2                 # low-rank only

    # Teacher forcing
    k_min: int = 5
    k_max: int = 20

    # Optimisation
    learning_rate: float = 1e-3
    n_epochs: int = 1500
    batch_size: int = 32
    clip_grad: float = 1.0
    weight_decay: float = 0.0

    # Loss weighting  (Section 3.3 of revised_training_strategy.md)
    #   convergence_weight > 1 up-weights the first convergence_steps
    #   autonomous steps to emphasise the transient attractor dynamics
    #   rather than steady-state maintenance.
    convergence_weight: float = 1.0
    convergence_steps: int = 0    # 0 = uniform weighting

    # Noise injection: adds Gaussian noise to h during training
    # to prevent discrete attractor basins (polygon → continuous ring).
    # If noise_std_final is set, noise linearly anneals from noise_std
    # to noise_std_final over training.
    noise_std: float = 0.0
    noise_std_final: float | None = None  # None = constant noise

    # Circular shift augmentation: randomly shift neuron dimension per batch.
    # Exploits rotational symmetry to provide continuous angle coverage.
    circular_shift_augment: bool = False

    # LR scheduling
    scheduler: str = "plateau"    # "plateau", "cosine", or "none"
    plateau_patience: int = 50
    plateau_factor: float = 0.5
    min_lr: float = 1e-5

    # Data
    observation_fraction: float = 1.0   # k/N; 1.0 = full observation
    val_fraction: float = 0.2
    val_seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 200

    # Logging
    log_every: int = 25

    # Device
    device: str = "auto"


# ── Helpers ──────────────────────────────────────────────────────────────


def get_device(config: TrainingConfig) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    convergence_weight: float = 1.0,
    convergence_steps: int = 0,
) -> torch.Tensor:
    """
    Masked MSE over the autonomous phase.

    Args:
        y_pred: (B, T, N)
        y_true: (B, T, N)
        mask:   (B, T) — 1 for autonomous steps, 0 for teacher-forced
        convergence_weight: extra weight for the first `convergence_steps`
                            autonomous steps (emphasise transient dynamics)
        convergence_steps:  number of steps to up-weight (0 = uniform)
    """
    mse = ((y_pred - y_true) ** 2).mean(dim=-1)          # (B, T)

    if convergence_steps > 0 and convergence_weight != 1.0:
        # auto_step counts 1, 2, 3, ... from the first autonomous step
        auto_step = mask.cumsum(dim=1)
        weight = torch.where(
            (auto_step > 0) & (auto_step <= convergence_steps),
            torch.tensor(convergence_weight, device=mse.device),
            torch.ones(1, device=mse.device),
        )
        mse = mse * weight

    return (mse * mask).sum() / mask.sum().clamp(min=1)


def split_by_angle(
    n_trials: int,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split trials keeping Group A + B for each angle together.

    The dataset has pairs (A_i, B_i) for angle i, so trial 2i = Group A
    and trial 2i+1 = Group B.  We split *angles*, not individual trials.
    """
    n_angles = n_trials // 2
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_angles)

    n_val = max(1, int(n_angles * val_fraction))
    val_angles = set(perm[:n_val].tolist())

    train_idx, val_idx = [], []
    for i in range(n_trials):
        (val_idx if i // 2 in val_angles else train_idx).append(i)

    return np.array(train_idx), np.array(val_idx)


# ── Main training function ──────────────────────────────────────────────


def train(
    data_path: str | Path,
    config: TrainingConfig | None = None,
) -> dict:
    """
    Full training pipeline.

    Returns a dict with: model, losses, config, metadata.
    """
    if config is None:
        config = TrainingConfig()
    data_path = Path(data_path)
    device = get_device(config)

    # ── Load data ────────────────────────────────────────────────
    data = np.load(data_path)
    trajectories = data["trajectories"]        # (n_trials, T, N) standardised
    groups = data["groups"]
    neuron_angles = data["neuron_angles"]
    norm_mean = data["mean"]
    norm_std = data["std"]

    n_trials, T, N = trajectories.shape
    print(f"Data: {n_trials} trials, T={T} bins, N={N} neurons")

    # ── Observed neurons ─────────────────────────────────────────
    observed_idx = None
    if config.observation_fraction < 1.0:
        rng = np.random.default_rng(config.val_seed + 1000)
        n_obs = max(1, int(N * config.observation_fraction))
        observed_idx = np.sort(rng.choice(N, n_obs, replace=False))
        print(f"Observing {n_obs}/{N} neurons")

    input_dim = len(observed_idx) if observed_idx is not None else N

    # ── Train / val split ────────────────────────────────────────
    train_idx, val_idx = split_by_angle(
        n_trials, config.val_fraction, config.val_seed,
    )
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val  "
          f"(by angle, seed={config.val_seed})")

    train_ds = RingAttractorDataset(
        trajectories[train_idx], config.k_min, config.k_max, observed_idx,
    )
    val_ds = RingAttractorDataset(
        trajectories[val_idx], config.k_min, config.k_max, observed_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    # ── Model ────────────────────────────────────────────────────
    model = create_model(
        config.model_type,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=N,
        alpha=config.alpha,
        rank=config.rank,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model_type}  |  {n_params:,} params  |  "
          f"alpha={config.alpha}  |  device={device}")
    if config.model_type == "lowrank":
        print(f"  rank={config.rank}")

    # ── Optimiser & scheduler ────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.plateau_patience,
            factor=config.plateau_factor, min_lr=config.min_lr,
        )
    elif config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, eta_min=config.min_lr,
        )

    # ── Checkpoint dir ───────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ── Training loop ────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses: list[float] = []
    grad_norms: list[float] = []
    lrs: list[float] = []

    t0 = time.time()
    print()

    for epoch in range(config.n_epochs):
        # --- noise schedule ---
        if config.noise_std_final is not None:
            frac = epoch / max(config.n_epochs - 1, 1)
            noise = config.noise_std * (1 - frac) + config.noise_std_final * frac
        else:
            noise = config.noise_std

        # --- train ---
        model.train()
        epoch_loss = 0.0
        epoch_gnorm = 0.0
        n_batches = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            if config.circular_shift_augment and x.shape[-1] == y.shape[-1]:
                shift = torch.randint(0, y.shape[-1], (1,)).item()
                x = torch.roll(x, shifts=shift, dims=-1)
                y = torch.roll(y, shifts=shift, dims=-1)

            optimizer.zero_grad()
            y_pred, _ = model(x, noise_std=noise)
            loss = compute_loss(
                y_pred, y, mask,
                config.convergence_weight, config.convergence_steps,
            )
            loss.backward()

            if config.clip_grad > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad,
                ).item()
            else:
                gnorm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

            optimizer.step()

            epoch_loss += loss.item()
            epoch_gnorm += gnorm
            n_batches += 1

        avg_train = epoch_loss / n_batches
        avg_gnorm = epoch_gnorm / n_batches
        train_losses.append(avg_train)
        grad_norms.append(avg_gnorm)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            vl_sum, n_vl = 0.0, 0
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_pred, _ = model(x)
                vl_sum += compute_loss(y_pred, y, mask).item()
                n_vl += 1
            avg_val = vl_sum / max(n_vl, 1)
        val_losses.append(avg_val)

        # --- LR scheduler ---
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(avg_val)
            else:
                scheduler.step()

        # --- best model ---
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

        # --- log ---
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch+1:4d}/{config.n_epochs}  "
                f"train={avg_train:.6f}  val={avg_val:.6f}  "
                f"gnorm={avg_gnorm:.3f}  lr={lr:.1e}  "
                f"[{elapsed:.0f}s]"
                f"{'  *best*' if is_best else ''}"
            )

        # --- checkpoint ---
        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "config": asdict(config),
            }, ckpt_dir / f"epoch_{epoch+1:04d}.pt")

    # ── Finish ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  |  best val loss = {best_val_loss:.6f}")

    # Load best weights
    model.load_state_dict(
        torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "lrs": lrs,
        "best_val_loss": best_val_loss,
        "config": config,
        "neuron_angles": neuron_angles,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "observed_idx": observed_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "device": device,
    }


# ── CLI entry point ──────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description="Train autonomous RNN on ring attractor data")
    p.add_argument("data", nargs="?", default="data/ring_attractor_dataset.npz")
    p.add_argument("--model", default="vanilla", choices=["vanilla", "lowrank"])
    p.add_argument("--hidden-dim", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--obs-fraction", type=float, default=1.0)
    p.add_argument("--conv-weight", type=float, default=1.0,
                   help="Up-weight for early autonomous steps")
    p.add_argument("--conv-steps", type=int, default=0,
                   help="How many early steps to up-weight (0=uniform)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Noise std injected into h during training")
    p.add_argument("--noise-final", type=float, default=None,
                   help="Final noise std (linear annealing from --noise)")
    p.add_argument("--circular-shift", action="store_true",
                   help="Enable circular shift augmentation")
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = TrainingConfig(
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        rank=args.rank,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        observation_fraction=args.obs_fraction,
        convergence_weight=args.conv_weight,
        convergence_steps=args.conv_steps,
        noise_std=args.noise,
        noise_std_final=args.noise_final,
        circular_shift_augment=args.circular_shift,
        device=args.device,
    )
    result = train(args.data, cfg)
