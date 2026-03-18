"""
Command-line interface for dataset generation.

Usage:
    python -m dataset.cli --output data/ring_dataset.npz
    python -m dataset.cli --output data/ring_dataset.npz --n-trials 100 --steps 10000
"""

import argparse
from pathlib import Path

from dataset.generator import DatasetGenerator, DatasetConfig
from dataset.io import save_dataset_with_reductions


def main():
    parser = argparse.ArgumentParser(
        description="Generate ring attractor datasets for RNN training"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ring_dataset.npz",
        help="Output file path (default: ring_dataset.npz)"
    )

    # Network parameters
    parser.add_argument("--network-size", type=int, default=100, help="Number of neurons")
    parser.add_argument("--j0", type=float, default=-2.0, help="Global inhibition")
    parser.add_argument("--j1", type=float, default=4.0, help="Local excitation")

    # Simulation parameters
    parser.add_argument("--dt", type=float, default=0.01, help="Integration timestep")
    parser.add_argument("--tau", type=float, default=100.0, help="Time constant")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level")

    # Trial structure
    parser.add_argument("--steps", type=int, default=5000, help="Steps per trial")
    parser.add_argument("--bin-factor", type=int, default=50, help="Integration steps per bin")
    parser.add_argument("--cue-duration", type=int, default=200, help="Cue duration in steps")

    # Trial counts
    parser.add_argument("--n-noise", type=int, default=10, help="Number of noise-only trials")
    parser.add_argument("--n-cue", type=int, default=30, help="Number of single-cue trials")
    parser.add_argument("--n-pert", type=int, default=10, help="Number of perturbation trials")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--obs-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1],
        help="Observation fractions to include"
    )

    args = parser.parse_args()

    config = DatasetConfig(
        network_size=args.network_size,
        j0=args.j0,
        j1=args.j1,
        dt=args.dt,
        tau=args.tau,
        sigma=args.sigma,
        steps_per_trial=args.steps,
        bin_factor=args.bin_factor,
        cue_duration=args.cue_duration,
        n_noise_trials=args.n_noise,
        n_single_cue_trials=args.n_cue,
        n_perturbation_trials=args.n_pert,
        seed=args.seed,
    )

    print("=" * 60)
    print("Ring Attractor Dataset Generator")
    print("=" * 60)
    print(f"Network size: {config.network_size}")
    print(f"Steps per trial: {config.steps_per_trial}")
    print(f"Bins per trial: {config.bins_per_trial}")
    print(f"Trials: {config.n_noise_trials} noise + {config.n_single_cue_trials} cue + {config.n_perturbation_trials} pert = {config.total_trials}")
    print(f"Observation fractions: {args.obs_fractions}")
    print("=" * 60)
    print()

    print("Generating dataset...")
    generator = DatasetGenerator(config)
    dataset = generator.generate()

    print(f"\nSaving to {args.output}...")
    save_dataset_with_reductions(
        dataset,
        args.output,
        observation_fractions=args.obs_fractions,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
