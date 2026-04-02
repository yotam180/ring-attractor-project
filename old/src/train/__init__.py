"""
Training module for RNN models.

Components:
  - VanillaRateRNN: Standard RNN architecture
  - LowRankRateRNN: Low-rank constrained RNN
  - Trainer: Training loop with checkpointing
  - CheckpointManager: Checkpoint save/load utilities

Usage:
    from train import Trainer, TrainingConfig

    config = TrainingConfig(
        model_type="vanilla",
        hidden_dim=100,
        n_epochs=300,
    )
    trainer = Trainer(config)
    result = trainer.train(X_train, Y_train, X_test, Y_test, Y_theta_test)
"""

from train.models import (
    VanillaRateRNN,
    LowRankRateRNN,
    create_model,
)
from train.trainer import (
    Trainer,
    TrainingConfig,
    TrainingResult,
)
from train.checkpoint import (
    TrainingState,
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)

__all__ = [
    # Models
    "VanillaRateRNN",
    "LowRankRateRNN",
    "create_model",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    # Checkpointing
    "TrainingState",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]
