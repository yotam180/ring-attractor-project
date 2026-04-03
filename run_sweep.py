#!/usr/bin/env python3
"""
Milestone 2: Partial observation sweep.

Trains a vanilla RNN at multiple observation fractions and evaluates
whether the learned dynamics still form a continuous ring attractor.

Results are saved to data/sweep_results/ for visualization.
"""
import json
import time
import sys
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

# ── Configuration ────────────────────────────────────────────────────────

OBS_FRACS = [1.0, 0.75, 0.5, 0.25, 0.1]
SEEDS = [42, 123]
DATA_PATH = "data/ring_attractor_dataset.npz"
OUT_DIR = Path("data/sweep_results")

# ── Helpers ──────────────────────────────────────────────────────────────

def run_one(obs_frac: float, seed: int) -> dict:
    """Train + evaluate one condition.  Returns a dict of scalar metrics + arrays."""
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

    # ── Evaluate ─────────────────────────────────────────────────
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

    # Eigenvalues at a few fixed points
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

    # ── Save arrays for visualization ────────────────────────────
    arr_path = OUT_DIR / f"{tag}.npz"
    np.savez_compressed(str(arr_path),
        pca_proj=ring.pca_proj,
        theta_final=ring.theta_final,
        h_final=ring.h_final,
        test_angles=gen.test_angles,
        final_angles=gen.final_angles,
        drift_deg=gen.drift_deg,
        eig_mags=np.array(eig_mags_all[0]),  # first FP
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results = []
    for obs_frac in OBS_FRACS:
        for seed in SEEDS:
            r = run_one(obs_frac, seed)
            results.append(r)

            # Save incrementally in case of crash
            with open(OUT_DIR / "summary.json", "w") as f:
                json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  SWEEP COMPLETE  ({total / 60:.1f} min)")
    print(f"{'='*60}\n")

    # Print summary table
    print(f"{'obs':>6} {'seed':>5} {'unif':>6} {'circ':>6} {'drift':>7} {'mse':>8} {'pass':>5}")
    print("-" * 50)
    for r in results:
        print(f"{r['obs_frac']:6.2f} {r['seed']:5d} {r['uniformity']:6.3f} "
              f"{r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{r['mse']:8.4f} {'YES' if r['milestone_1_pass'] else 'NO':>5}")


if __name__ == "__main__":
    main()
