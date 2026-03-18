"""
Dataset generation and manipulation for ring attractor experiments.

Main components:
  - DatasetGenerator: Generates multi-trial datasets from the ring attractor model
  - DatasetConfig: Configuration for dataset generation
  - NeuronDropout: Simulates partial observation by dropping neurons
  - save_dataset / load_dataset: I/O utilities

Usage:
    from dataset import DatasetGenerator, DatasetConfig, save_dataset_with_reductions

    config = DatasetConfig(
        network_size=100,
        n_single_cue_trials=40,
        steps_per_trial=5000,
    )
    generator = DatasetGenerator(config)
    dataset = generator.generate()
    save_dataset_with_reductions(dataset, "my_dataset.npz")
"""

from dataset.generator import DatasetGenerator, DatasetConfig, GeneratedDataset
from dataset.trial_types import TrialType, TrialConfig, TrialData
from dataset.reduction import NeuronDropout
from dataset.io import (
    save_dataset,
    load_dataset,
    save_dataset_with_reductions,
    load_dataset_for_training,
)

__all__ = [
    # Generator
    "DatasetGenerator",
    "DatasetConfig",
    "GeneratedDataset",
    # Trial types
    "TrialType",
    "TrialConfig",
    "TrialData",
    # Reduction
    "NeuronDropout",
    # I/O
    "save_dataset",
    "load_dataset",
    "save_dataset_with_reductions",
    "load_dataset_for_training",
]
