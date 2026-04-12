#!/usr/bin/env python3
"""
Run 8 additional seeds per k/N condition to get robust statistics.

Seeds 42, 123, 7, 99, 2024 already exist for some conditions.
This adds seeds 300-307 across all obs fracs, skipping any that already exist.

Expected runtime: ~18 min/run × 56 runs ≈ 17 hours on MPS.
Saves incrementally to data/sweep_results/summary.json (appends to existing).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.train import train, TrainingConfig
from src.train.evaluation import (
    evaluate_predictions,
    autonomous_fixed_points,
    generalization_test,
    eigenvalue_analysis,
    singular_value_analysis,
)

DATA_PATH = "data/ring_attractor_dataset.npz"
OUT_DIR = Path("data/sweep_results")

# All obs fracs we want dense coverage on
OBS_FRACS = [1.0, 0.75, 0.50, 0.25, 0.20, 0.15, 0.10]

# 8 new seeds
NEW_SEEDS = [300, 301, 302, 303, 304, 305, 306, 307]


def run_one(obs_frac: float, seed: int) -> dict:
    tag = f"obs{int(obs_frac * 100):03d}_seed{seed}"
    ckpt_dir = f"checkpoints/sweep_{tag}"
    print(f"\n{'='*60}")
    print(f"  obs_frac={obs_frac}  seed={seed}  [{tag}]")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    t0 = time.time()

    result = train(DATA_PATH, TrainingConfig(
        model_type="vanilla",
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=30,
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=False,
        observation_fraction=obs_frac,
        checkpoint_dir=ckpt_dir,
        log_every=500,
    ))
    train_time = time.time() - t0

    model = result["model"]
    device = result["device"]
    observed_idx = result["observed_idx"]

    data = np.load(DATA_PATH)
    trajectories = data["trajectories"]
    neuron_angles = data["neuron_angles"]
    norm_mean = data["mean"]
    norm_std = data["std"]
    val_traj = trajectories[result["val_idx"]]

    pred = evaluate_predictions(
        model, val_traj, neuron_angles, norm_mean, norm_std, device,
        observed_idx=observed_idx,
    )
    ring = autonomous_fixed_points(
        model, neuron_angles, norm_mean, norm_std, device,
    )
    gen = generalization_test(
        model, neuron_angles, norm_mean, norm_std, device,
        observed_idx=observed_idx,
    )

    n_fp = min(5, len(ring.h_final))
    fp_indices = np.linspace(0, len(ring.h_final) - 1, n_fp, dtype=int)
    eig_mags_all = []
    for idx in fp_indices:
        er = eigenvalue_analysis(model, ring.h_final[idx])
        eig_mags_all.append(er.magnitudes)

    svs = singular_value_analysis(model)

    m1_pass = (
        ring.uniformity > 0.8
        and ring.circularity > 0.7
        and gen.mean_abs_drift_deg < 5.0
    )

    print(f"\n  [{tag}] unif={ring.uniformity:.3f}  circ={ring.circularity:.3f}  "
          f"drift={gen.mean_abs_drift_deg:.2f}°  "
          f"{'PASS' if m1_pass else 'FAIL'}  ({train_time:.0f}s)")

    np.savez_compressed(str(OUT_DIR / f"{tag}.npz"),
        pca_proj=ring.pca_proj,
        theta_final=ring.theta_final,
        h_final=ring.h_final,
        test_angles=gen.test_angles,
        final_angles=gen.final_angles,
        drift_deg=gen.drift_deg,
        eig_mags=np.array(eig_mags_all[0]),
        singular_values=svs,
    )

    return {
        "obs_frac": obs_frac,
        "seed": seed,
        "tag": tag,
        "uniformity": ring.uniformity,
        "circularity": ring.circularity,
        "spread": ring.spread,
        "drift_mean": gen.mean_abs_drift_deg,
        "drift_max": float(np.abs(gen.drift_deg).max()),
        "mse": pred.mse,
        "angle_error": pred.angle_error_deg,
        "best_val_loss": result["best_val_loss"],
        "milestone_1_pass": m1_pass,
        "train_time_s": train_time,
        "svs_top5": svs[:5].tolist(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"

    # Load existing results
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
    else:
        results = []

    existing_tags = {r["tag"] for r in results}

    # Build run list
    runs = []
    for obs_frac in OBS_FRACS:
        for seed in NEW_SEEDS:
            tag = f"obs{int(obs_frac * 100):03d}_seed{seed}"
            if tag not in existing_tags:
                runs.append((obs_frac, seed))

    print(f"Total runs to do: {len(runs)}")
    print(f"Existing results: {len(results)}")
    print(f"Estimated time: {len(runs) * 18 / 60:.1f} hours\n")

    t0 = time.time()
    for i, (obs_frac, seed) in enumerate(runs):
        print(f"\n>>> Run {i+1}/{len(runs)}")
        r = run_one(obs_frac, seed)
        results.append(r)

        # Save incrementally
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - t0
        per_run = elapsed / (i + 1)
        remaining = per_run * (len(runs) - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f} min | "
              f"ETA: {remaining/60:.1f} min ({remaining/3600:.1f} hrs)")

    total = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  ALL DONE  ({total / 3600:.1f} hours)")
    print(f"{'='*60}\n")

    # Print pass rates
    results_sorted = sorted(results, key=lambda x: (-x["obs_frac"], x["seed"]))
    obs_fracs = sorted(set(r["obs_frac"] for r in results), reverse=True)
    print("Pass rates:")
    for obs in obs_fracs:
        runs_at = [r for r in results if r["obs_frac"] == obs]
        n_pass = sum(1 for r in runs_at if r["milestone_1_pass"])
        print(f"  k/N={obs:.2f}: {n_pass}/{len(runs_at)} "
              f"({100*n_pass/len(runs_at):.0f}%)")


if __name__ == "__main__":
    main()
