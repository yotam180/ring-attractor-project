"""
Training loop for RNN models with checkpoint support.

Features:
  - Configurable training parameters
  - Periodic checkpointing
  - Resume from checkpoint
  - Progress logging
  - MPS/CUDA/CPU device support
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train.models import create_model, VanillaRateRNN, LowRankRateRNN
from train.checkpoint import (
    TrainingState,
    CheckpointManager,
    load_checkpoint,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_type: str = "vanilla"  # "vanilla" or "lowrank"
    hidden_dim: int = 100
    alpha: float = 0.1
    rank: int = 2  # Only for lowrank

    # Optimization
    learning_rate: float = 1e-3
    n_epochs: int = 300
    clip_grad: float | None = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 50  # Save every N epochs
    keep_last_n: int = 3

    # Logging
    log_every: int = 10  # Log every N epochs

    # Device
    device: str = "auto"  # "auto", "mps", "cuda", "cpu"


@dataclass
class TrainingResult:
    """Results from a training run."""

    model: nn.Module
    train_losses: list[float]
    test_losses: list[float]
    final_train_mse: float
    final_test_mse: float
    angle_error: float
    epochs_trained: int
    training_time: float

    # For analysis
    hidden_states_test: np.ndarray | None = None
    predictions_test: np.ndarray | None = None
    theta_pred_test: np.ndarray | None = None


def get_device(config: TrainingConfig) -> torch.device:
    """Determine the best available device."""
    if config.device != "auto":
        return torch.device(config.device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def decode_angle(rates: np.ndarray) -> np.ndarray:
    """
    Decode bump angle from firing rates using population vector.

    Args:
        rates: Array of shape (T, N) or (B, T, N)

    Returns:
        Decoded angles of shape (T,) or (B, T)
    """
    if rates.ndim == 2:
        T, N = rates.shape
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = (rates * np.cos(angles)).sum(axis=1)
        y = (rates * np.sin(angles)).sum(axis=1)
        return np.arctan2(y, x)
    elif rates.ndim == 3:
        B, T, N = rates.shape
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = (rates * np.cos(angles)).sum(axis=2)
        y = (rates * np.sin(angles)).sum(axis=2)
        return np.arctan2(y, x)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {rates.ndim}D")


def circular_angle_error(theta_pred: np.ndarray, theta_true: np.ndarray) -> float:
    """Compute mean absolute circular error between angles."""
    diff = np.angle(np.exp(1j * (theta_pred - theta_true)))
    return float(np.mean(np.abs(diff)))


class Trainer:
    """
    Trains RNN models on ring attractor data.

    Usage:
        trainer = Trainer(config)
        result = trainer.train(X_train, Y_train, X_test, Y_test, Y_theta_test)

    To resume from checkpoint:
        trainer = Trainer(config)
        trainer.resume_from_checkpoint(checkpoint_path)
        result = trainer.train(X_train, Y_train, X_test, Y_test, Y_theta_test)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device(config)

        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.checkpoint_manager: CheckpointManager | None = None

        # State for resume
        self._start_epoch = 0
        self._train_losses: list[float] = []
        self._test_losses: list[float] = []
        self._best_test_loss = float("inf")

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create the model based on config."""
        model = create_model(
            model_type=self.config.model_type,
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            alpha=self.config.alpha,
            rank=self.config.rank,
        )
        return model.to(self.device)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def resume_from_checkpoint(
        self,
        checkpoint_path: Path | str,
        input_dim: int,
        output_dim: int,
    ) -> None:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            input_dim: Input dimension (must match checkpoint)
            output_dim: Output dimension (must match checkpoint)
        """
        self.model = self._create_model(input_dim, output_dim)
        self.optimizer = self._create_optimizer()

        state = load_checkpoint(checkpoint_path, self.model, self.optimizer)

        # Verify config matches
        if state.model_type != self.config.model_type:
            raise ValueError(
                f"Model type mismatch: checkpoint has {state.model_type}, "
                f"config has {self.config.model_type}"
            )
        if state.hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: checkpoint has {state.hidden_dim}, "
                f"config has {self.config.hidden_dim}"
            )

        self._start_epoch = state.epoch
        self._train_losses = state.train_losses
        self._test_losses = state.test_losses
        self._best_test_loss = state.best_test_loss

        print(f"Resumed from checkpoint: epoch {state.epoch}, best test loss {state.best_test_loss:.6f}")

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        Y_theta_test: np.ndarray,
        observation_level: float = 1.0,
        progress_callback: Callable[[int, float, float], None] | None = None,
    ) -> TrainingResult:
        """
        Train the model.

        Args:
            X_train: Training inputs (n_train, T, input_dim)
            Y_train: Training targets (n_train, T, output_dim)
            X_test: Test inputs (n_test, T, input_dim)
            Y_test: Test targets (n_test, T, output_dim)
            Y_theta_test: Test angle targets (n_test, T)
            observation_level: Fraction of neurons observed (for logging)
            progress_callback: Optional callback(epoch, train_loss, test_loss)

        Returns:
            TrainingResult with model and metrics
        """
        start_time = time.time()

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=self.device)

        input_dim = X_train.shape[-1]
        output_dim = Y_train.shape[-1]

        # Create model if not resuming
        if self.model is None:
            self.model = self._create_model(input_dim, output_dim)
            self.optimizer = self._create_optimizer()

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            model_type=self.config.model_type,
            observation_level=observation_level,
            keep_last_n=self.config.keep_last_n,
        )

        criterion = nn.MSELoss()

        print(f"Training {self.config.model_type} RNN on {self.device}")
        print(f"  Input dim: {input_dim}, Hidden dim: {self.config.hidden_dim}, Output dim: {output_dim}")
        print(f"  Epochs: {self._start_epoch} -> {self.config.n_epochs}")
        print()

        # Training loop
        for epoch in range(self._start_epoch, self.config.n_epochs):
            # Train step
            self.model.train()
            self.optimizer.zero_grad()

            y_hat, _ = self.model(X_train_t)
            train_loss = criterion(y_hat, Y_train_t)

            train_loss.backward()
            if self.config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad,
                )
            self.optimizer.step()

            train_loss_val = train_loss.item()
            self._train_losses.append(train_loss_val)

            # Eval step
            self.model.eval()
            with torch.no_grad():
                y_pred_test, _ = self.model(X_test_t)
                test_loss = criterion(y_pred_test, Y_test_t)
                test_loss_val = test_loss.item()
                self._test_losses.append(test_loss_val)

            # Track best
            is_best = test_loss_val < self._best_test_loss
            if is_best:
                self._best_test_loss = test_loss_val

            # Logging
            if (epoch + 1) % self.config.log_every == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:4d}/{self.config.n_epochs}  "
                    f"train_loss={train_loss_val:.6f}  "
                    f"test_loss={test_loss_val:.6f}"
                    f"{'  *best*' if is_best else ''}"
                )

            # Callback
            if progress_callback is not None:
                progress_callback(epoch + 1, train_loss_val, test_loss_val)

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                state = TrainingState(
                    epoch=epoch + 1,
                    global_step=epoch + 1,
                    best_test_loss=self._best_test_loss,
                    train_losses=self._train_losses.copy(),
                    test_losses=self._test_losses.copy(),
                    model_type=self.config.model_type,
                    hidden_dim=self.config.hidden_dim,
                    observation_level=observation_level,
                    learning_rate=self.config.learning_rate,
                )
                path = self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    state,
                    is_best=is_best,
                )
                print(f"  Saved checkpoint: {path.name}")

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred_train, _ = self.model(X_train_t)
            y_pred_test, h_test = self.model(X_test_t)

            final_train_mse = criterion(y_pred_train, Y_train_t).item()
            final_test_mse = criterion(y_pred_test, Y_test_t).item()

        # Decode angles
        y_pred_test_np = y_pred_test.cpu().numpy()
        theta_pred_test = decode_angle(y_pred_test_np)
        angle_error = circular_angle_error(theta_pred_test, Y_theta_test)

        training_time = time.time() - start_time

        print()
        print(f"Training complete in {training_time:.1f}s")
        print(f"  Final train MSE: {final_train_mse:.6f}")
        print(f"  Final test MSE:  {final_test_mse:.6f}")
        print(f"  Angle error:     {angle_error:.4f} rad ({np.degrees(angle_error):.2f}°)")

        # Save final checkpoint
        state = TrainingState(
            epoch=self.config.n_epochs,
            global_step=self.config.n_epochs,
            best_test_loss=self._best_test_loss,
            train_losses=self._train_losses.copy(),
            test_losses=self._test_losses.copy(),
            model_type=self.config.model_type,
            hidden_dim=self.config.hidden_dim,
            observation_level=observation_level,
            learning_rate=self.config.learning_rate,
        )
        final_path = self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            state,
            is_best=final_test_mse <= self._best_test_loss,
        )
        print(f"  Final checkpoint: {final_path.name}")

        return TrainingResult(
            model=self.model,
            train_losses=self._train_losses,
            test_losses=self._test_losses,
            final_train_mse=final_train_mse,
            final_test_mse=final_test_mse,
            angle_error=angle_error,
            epochs_trained=self.config.n_epochs - self._start_epoch,
            training_time=training_time,
            hidden_states_test=h_test.cpu().numpy(),
            predictions_test=y_pred_test_np,
            theta_pred_test=theta_pred_test,
        )
