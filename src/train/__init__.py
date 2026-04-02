"""
Training package for ring attractor RNN.

Quick-start::

    from src.train import train, TrainingConfig
    result = train("data/ring_attractor_dataset.npz", TrainingConfig())

    # result['model'] is the best model (loaded from checkpoint)
    # result['train_losses'], result['val_losses'] for loss curves

    from src.train import full_evaluation
    report = full_evaluation(
        result['model'], "data/ring_attractor_dataset.npz",
        result['device'], result['val_idx'], result['observed_idx'],
    )
"""

from .models import VanillaRateRNN, LowRankRateRNN, create_model
from .dataset import RingAttractorDataset
from .training import TrainingConfig, train, compute_loss
from .evaluation import (
    full_evaluation,
    evaluate_predictions,
    autonomous_fixed_points,
    generalization_test,
    eigenvalue_analysis,
    singular_value_analysis,
    PredictionMetrics,
    RingScore,
    GeneralizationResult,
    EigenvalueResult,
)

__all__ = [
    # Models
    "VanillaRateRNN",
    "LowRankRateRNN",
    "create_model",
    # Dataset
    "RingAttractorDataset",
    # Training
    "TrainingConfig",
    "train",
    "compute_loss",
    # Evaluation
    "full_evaluation",
    "evaluate_predictions",
    "autonomous_fixed_points",
    "generalization_test",
    "eigenvalue_analysis",
    "singular_value_analysis",
    "PredictionMetrics",
    "RingScore",
    "GeneralizationResult",
    "EigenvalueResult",
]
