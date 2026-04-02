"""
Dataset I/O utilities for saving and loading ring attractor datasets.

Saves in .npz format compatible with the RNN training notebooks.
"""

from pathlib import Path
from dataclasses import asdict
import json
import numpy as np

from dataset.generator import GeneratedDataset, DatasetConfig
from dataset.trial_types import TrialType
from dataset.reduction import NeuronDropout


def save_dataset(dataset: GeneratedDataset, path: str | Path) -> None:
    """
    Save a generated dataset to an .npz file.

    The format is compatible with the RNN training notebooks (11, 12).

    Args:
        dataset: The generated dataset
        path: Output file path (should end in .npz)
    """
    if dataset.Y_rates is None:
        raise ValueError("Dataset not finalized. Call dataset.finalize() first.")

    path = Path(path)

    # Prepare config as JSON-serializable dict
    config_dict = {
        "network_size": dataset.config.network_size,
        "j0": dataset.config.j0,
        "j1": dataset.config.j1,
        "dt": dataset.config.dt,
        "tau": dataset.config.tau,
        "sigma": dataset.config.sigma,
        "rate_scale": dataset.config.rate_scale,
        "bin_factor": dataset.config.bin_factor,
        "smoothing_window": dataset.config.smoothing_window,
        "steps_per_trial": dataset.config.steps_per_trial,
        "bins_per_trial": dataset.config.bins_per_trial,
        "dt_bin": dataset.config.dt_bin,
        "cue_duration": dataset.config.cue_duration,
        "cue_amplitude": dataset.config.cue_amplitude,
        "n_noise_trials": dataset.config.n_noise_trials,
        "n_single_cue_trials": dataset.config.n_single_cue_trials,
        "n_perturbation_trials": dataset.config.n_perturbation_trials,
        "seed": dataset.config.seed,
    }

    # Trial type strings
    trial_type_strs = [t.value for t in dataset.trial_types]

    # Theta targets (None -> NaN for numpy compatibility)
    theta_targets = np.array(
        [t if t is not None else np.nan for t in dataset.theta_targets],
        dtype=np.float32,
    )

    np.savez_compressed(
        path,
        # Main training arrays
        X_cue=dataset.X_cue,
        Y_rates=dataset.Y_rates,
        Y_theta=dataset.Y_theta,
        # Train/test split
        train_idx=dataset.train_idx,
        test_idx=dataset.test_idx,
        # Metadata
        trial_types=np.array(trial_type_strs, dtype=object),
        theta_targets=theta_targets,
        config_json=json.dumps(config_dict),
    )

    print(f"Saved dataset to {path}")
    print(f"  Trials: {len(dataset.trials)}")
    print(f"  Shape: Y_rates={dataset.Y_rates.shape}")
    print(f"  Train/test: {len(dataset.train_idx)}/{len(dataset.test_idx)}")


def load_dataset(path: str | Path) -> dict:
    """
    Load a dataset from an .npz file.

    Returns a dict with all arrays and metadata.
    This is a lightweight loader — use for RNN training.

    Args:
        path: Path to the .npz file

    Returns:
        Dict with keys: X_cue, Y_rates, Y_theta, train_idx, test_idx,
                        trial_types, theta_targets, config
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    config = json.loads(str(data["config_json"]))

    return {
        "X_cue": data["X_cue"],
        "Y_rates": data["Y_rates"],
        "Y_theta": data["Y_theta"],
        "train_idx": data["train_idx"],
        "test_idx": data["test_idx"],
        "trial_types": [TrialType(t) for t in data["trial_types"]],
        "theta_targets": data["theta_targets"],
        "config": config,
    }


def save_dataset_with_reductions(
    dataset: GeneratedDataset,
    path: str | Path,
    observation_fractions: list[float] = [1.0, 0.5, 0.25, 0.1],
    seed: int = 42,
) -> None:
    """
    Save a dataset with pre-computed neuron dropout reductions.

    This creates multiple X_rates_obs_* arrays at different observation levels,
    matching the format expected by notebooks 11 and 12.

    Args:
        dataset: The generated dataset
        path: Output file path
        observation_fractions: List of fractions to include (e.g., [1.0, 0.5, 0.25, 0.1])
        seed: Random seed for reproducible dropout masks
    """
    if dataset.Y_rates is None:
        raise ValueError("Dataset not finalized. Call dataset.finalize() first.")

    path = Path(path)
    rng = np.random.default_rng(seed)

    # Prepare config
    config_dict = {
        "network_size": dataset.config.network_size,
        "j0": dataset.config.j0,
        "j1": dataset.config.j1,
        "dt": dataset.config.dt,
        "tau": dataset.config.tau,
        "sigma": dataset.config.sigma,
        "rate_scale": dataset.config.rate_scale,
        "bin_factor": dataset.config.bin_factor,
        "smoothing_window": dataset.config.smoothing_window,
        "steps_per_trial": dataset.config.steps_per_trial,
        "bins_per_trial": dataset.config.bins_per_trial,
        "dt_bin": dataset.config.dt_bin,
        "cue_duration": dataset.config.cue_duration,
        "cue_amplitude": dataset.config.cue_amplitude,
        "n_noise_trials": dataset.config.n_noise_trials,
        "n_single_cue_trials": dataset.config.n_single_cue_trials,
        "n_perturbation_trials": dataset.config.n_perturbation_trials,
        "seed": dataset.config.seed,
        "observation_fractions": observation_fractions,
    }

    trial_type_strs = [t.value for t in dataset.trial_types]
    theta_targets = np.array(
        [t if t is not None else np.nan for t in dataset.theta_targets],
        dtype=np.float32,
    )

    # Build save dict
    save_dict = {
        "X_cue": dataset.X_cue,
        "Y_rates": dataset.Y_rates,
        "Y_theta": dataset.Y_theta,
        "train_idx": dataset.train_idx,
        "test_idx": dataset.test_idx,
        "trial_types": np.array(trial_type_strs, dtype=object),
        "theta_targets": theta_targets,
        "config_json": json.dumps(config_dict),
    }

    # Generate reduced observations for each fraction
    n_trials, T_bin, N = dataset.Y_rates.shape

    for frac in observation_fractions:
        pct = int(frac * 100)
        key = f"X_rates_obs_{pct}"

        if frac >= 1.0:
            # Full observation — just copy Y_rates
            save_dict[key] = dataset.Y_rates.copy()
            save_dict[f"obs_idx_{pct}"] = np.arange(N)
        else:
            # Apply neuron dropout
            dropout = NeuronDropout(
                network_size=N,
                keep_fraction=frac,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )

            # Create reduced version (select columns, don't zero)
            kept_idx = dropout.kept_indices
            reduced = dataset.Y_rates[:, :, kept_idx]

            save_dict[key] = reduced
            save_dict[f"obs_idx_{pct}"] = kept_idx

    np.savez_compressed(path, **save_dict)

    print(f"Saved dataset with reductions to {path}")
    print(f"  Trials: {n_trials}")
    print(f"  Shape: Y_rates={dataset.Y_rates.shape}")
    print(f"  Observation levels: {[f'{int(f*100)}%' for f in observation_fractions]}")
    print(f"  Train/test: {len(dataset.train_idx)}/{len(dataset.test_idx)}")


def load_dataset_for_training(
    path: str | Path,
    observation_level: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load a dataset ready for RNN training.

    Convenience function that returns the arrays in the format expected by
    the training notebooks.

    Args:
        path: Path to the .npz file
        observation_level: Fraction of neurons to observe (e.g., 0.5 for 50%)

    Returns:
        Tuple of (X, Y, Y_theta, train_idx, test_idx, config)
        where X is the observed rates at the specified level.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    pct = int(observation_level * 100)
    x_key = f"X_rates_obs_{pct}"

    if x_key not in data:
        available = [k for k in data.files if k.startswith("X_rates_obs_")]
        raise ValueError(
            f"Observation level {observation_level} not found. "
            f"Available: {available}"
        )

    config = json.loads(str(data["config_json"]))

    return (
        data[x_key],
        data["Y_rates"],
        data["Y_theta"],
        data["train_idx"],
        data["test_idx"],
        config,
    )
