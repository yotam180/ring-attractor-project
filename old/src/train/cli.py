"""
Command-line interface for training RNN models.

Usage:
    # Train vanilla RNN at 50% observation
    python -m train.cli --dataset data/ring_dataset.npz --obs-level 0.5

    # Train low-rank RNN
    python -m train.cli --dataset data/ring_dataset.npz --model lowrank --rank 2

    # Resume from checkpoint
    python -m train.cli --dataset data/ring_dataset.npz --resume checkpoints/vanilla_obs50_epoch0100_*.pt

    # Train all observation levels
    python -m train.cli --dataset data/ring_dataset.npz --all-obs-levels
"""

import argparse
from pathlib import Path
import numpy as np

from train.trainer import Trainer, TrainingConfig, TrainingResult
from train.checkpoint import find_latest_checkpoint
from dataset.io import load_dataset_for_training


def train_single(
    dataset_path: str,
    observation_level: float,
    config: TrainingConfig,
    resume_path: str | None = None,
) -> TrainingResult:
    """Train a single model at one observation level."""
    print("=" * 60)
    print(f"Training {config.model_type} RNN at {int(observation_level * 100)}% observation")
    print("=" * 60)
    print()

    # Load data
    X, Y, Y_theta, train_idx, test_idx, data_config = load_dataset_for_training(
        dataset_path,
        observation_level=observation_level,
    )

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Y_theta_test = Y_theta[test_idx]

    print(f"Dataset: {dataset_path}")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print()

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if resume_path is not None:
        trainer.resume_from_checkpoint(
            resume_path,
            input_dim=X_train.shape[-1],
            output_dim=Y_train.shape[-1],
        )

    # Train
    result = trainer.train(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        Y_theta_test=Y_theta_test,
        observation_level=observation_level,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train RNN models on ring attractor data"
    )

    # Required
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to dataset .npz file"
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="vanilla",
        choices=["vanilla", "lowrank"],
        help="Model type (default: vanilla)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=100,
        help="Hidden state dimension (default: 100)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Leak rate (default: 0.1)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="Rank for lowrank model (default: 2)"
    )

    # Training
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=300,
        help="Number of epochs (default: 300)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping (default: 1.0, 0 to disable)"
    )

    # Observation level
    parser.add_argument(
        "--obs-level", "-o",
        type=float,
        default=1.0,
        help="Observation level as fraction (default: 1.0)"
    )
    parser.add_argument(
        "--all-obs-levels",
        action="store_true",
        help="Train on all observation levels in dataset"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Checkpoint every N epochs (default: 50)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file"
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest checkpoint"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        rank=args.rank,
        learning_rate=args.lr,
        n_epochs=args.epochs,
        clip_grad=args.clip_grad if args.clip_grad > 0 else None,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
    )

    # Determine observation levels
    if args.all_obs_levels:
        # Load dataset to find available levels
        data = np.load(args.dataset, allow_pickle=True)
        obs_levels = []
        for key in data.files:
            if key.startswith("X_rates_obs_"):
                pct = int(key.split("_")[-1])
                obs_levels.append(pct / 100.0)
        obs_levels.sort(reverse=True)
        print(f"Found observation levels: {[f'{int(l*100)}%' for l in obs_levels]}")
    else:
        obs_levels = [args.obs_level]

    # Find resume checkpoint
    resume_path = args.resume
    if args.resume_latest:
        obs_pct = int(args.obs_level * 100)
        prefix = f"{args.model}_obs{obs_pct}_"
        resume_path = find_latest_checkpoint(args.checkpoint_dir, prefix)
        if resume_path:
            print(f"Found latest checkpoint: {resume_path}")
        else:
            print(f"No checkpoint found with prefix '{prefix}'")

    # Train
    results = {}
    for obs_level in obs_levels:
        # For multi-level training, only resume at the specified level
        rp = resume_path if obs_level == args.obs_level else None

        result = train_single(
            dataset_path=args.dataset,
            observation_level=obs_level,
            config=config,
            resume_path=rp,
        )
        results[obs_level] = result
        print()

    # Summary
    if len(results) > 1:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Obs Level':>10s}  {'Train MSE':>12s}  {'Test MSE':>12s}  {'Angle Err':>12s}")
        print("-" * 50)
        for obs_level in sorted(results.keys(), reverse=True):
            r = results[obs_level]
            print(
                f"{int(obs_level*100):>9d}%  "
                f"{r.final_train_mse:>12.6f}  "
                f"{r.final_test_mse:>12.6f}  "
                f"{r.angle_error:>11.4f}rad"
            )


if __name__ == "__main__":
    main()
