#!/usr/bin/env python3
"""
Additional sweep runs:
  - More seeds at k/N=0.75 and 0.10 (resolve ambiguity)
  - Fine-grained sweep at k/N=0.15 and 0.20 (pin down the cliff)
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

EXTRA_RUNS = [
    # More seeds at ambiguous points
    (0.75, 7), (0.75, 99), (0.75, 2024),
    (0.10, 7), (0.10, 99), (0.10, 2024),
    # Fine-grained threshold sweep
    (0.20, 42), (0.20, 123),
    (0.15, 42), (0.15, 123),
]


def run_one(obs_frac, seed):
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
    # Load existing results
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path) as f:
        results = json.load(f)

    existing_tags = {r["tag"] for r in results}
    t0 = time.time()

    for obs_frac, seed in EXTRA_RUNS:
        tag = f"obs{int(obs_frac * 100):03d}_seed{seed}"
        if tag in existing_tags:
            print(f"Skipping {tag} (already exists)")
            continue

        r = run_one(obs_frac, seed)
        results.append(r)

        # Save incrementally
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  EXTRA RUNS COMPLETE  ({total / 60:.1f} min)")
    print(f"{'='*60}\n")

    # Print full summary sorted by obs_frac, seed
    results_sorted = sorted(results, key=lambda x: (-x["obs_frac"], x["seed"]))
    print(f"{'obs':>6} {'seed':>5} {'unif':>6} {'circ':>6} {'drift':>7} {'mse':>8} {'pass':>5}")
    print("-" * 50)
    for r in results_sorted:
        print(f"{r['obs_frac']:6.2f} {r['seed']:5d} {r['uniformity']:6.3f} "
              f"{r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{r['mse']:8.4f} {'YES' if r['milestone_1_pass'] else 'NO':>5}")

    # Print pass rates per obs_frac
    print(f"\nPass rates:")
    obs_fracs = sorted(set(r["obs_frac"] for r in results), reverse=True)
    for obs in obs_fracs:
        runs = [r for r in results if r["obs_frac"] == obs]
        n_pass = sum(1 for r in runs if r["milestone_1_pass"])
        print(f"  k/N={obs:.2f}: {n_pass}/{len(runs)} pass")


if __name__ == "__main__":
    main()
