"""
Checkpoint utilities for saving and loading training state.

Supports:
  - Saving/loading model weights, optimizer state, and training progress
  - Automatic checkpoint naming with timestamps
  - Resume from latest checkpoint
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainingState:
    """Tracks training progress for checkpointing."""

    epoch: int
    global_step: int
    best_test_loss: float
    train_losses: list[float]
    test_losses: list[float]

    # Training config (for verification on resume)
    model_type: str
    hidden_dim: int
    observation_level: float
    learning_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingState":
        return cls(**d)


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    state: TrainingState,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Output file path
        model: The model to save
        optimizer: The optimizer to save
        state: Training state (epoch, losses, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": state.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> TrainingState:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        TrainingState from the checkpoint
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    state = TrainingState.from_dict(checkpoint["training_state"])

    return state


def find_latest_checkpoint(checkpoint_dir: Path | str, prefix: str = "") -> Path | None:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search
        prefix: Optional prefix to filter checkpoints

    Returns:
        Path to latest checkpoint, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pattern = f"{prefix}*.pt" if prefix else "*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def generate_checkpoint_name(
    model_type: str,
    observation_level: float,
    epoch: int,
) -> str:
    """Generate a descriptive checkpoint filename."""
    obs_pct = int(observation_level * 100)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_obs{obs_pct}_epoch{epoch:04d}_{timestamp}.pt"


class CheckpointManager:
    """
    Manages checkpoints for a training run.

    Handles:
      - Periodic checkpointing
      - Best model tracking
      - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        model_type: str,
        observation_level: float,
        keep_last_n: int = 3,
        keep_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type
        self.observation_level = observation_level
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best

        self.obs_pct = int(observation_level * 100)
        self.prefix = f"{model_type}_obs{self.obs_pct}_"

        self._best_path: Path | None = None
        self._checkpoints: list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        state: TrainingState,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint and manage old ones."""
        name = generate_checkpoint_name(
            self.model_type,
            self.observation_level,
            state.epoch,
        )
        path = self.checkpoint_dir / name

        save_checkpoint(path, model, optimizer, state)
        self._checkpoints.append(path)

        # Save best separately
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / f"{self.model_type}_obs{self.obs_pct}_best.pt"
            save_checkpoint(best_path, model, optimizer, state)
            self._best_path = best_path

        # Cleanup old checkpoints
        self._cleanup()

        return path

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        while len(self._checkpoints) > self.keep_last_n:
            old = self._checkpoints.pop(0)
            if old.exists() and old != self._best_path:
                old.unlink()

    def get_latest(self) -> Path | None:
        """Get the most recent checkpoint."""
        return find_latest_checkpoint(self.checkpoint_dir, self.prefix)

    def get_best(self) -> Path | None:
        """Get the best checkpoint."""
        best_path = self.checkpoint_dir / f"{self.model_type}_obs{self.obs_pct}_best.pt"
        return best_path if best_path.exists() else None
