#!/usr/bin/env python3
"""
T-axis sweep: vary observation time (trial length) at two observation fractions.

Tests T_bin = {200, 100, 50, 25} at k/N = {1.0, 0.25} with 2 seeds each.
k/N=1.0 establishes the baseline; k/N=0.25 tests the interaction with partial observation.

Results appended to data/sweep_results_T/
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
OUT_DIR = Path("data/sweep_results_T")

T_VALUES = [200, 100, 50, 25]
OBS_FRACS = [1.0, 0.25]
SEEDS = [42, 123]


def run_one(T_bin, obs_frac, seed):
    tag = f"T{T_bin:03d}_obs{int(obs_frac * 100):03d}_seed{seed}"
    ckpt_dir = f"checkpoints/sweep_T_{tag}"
    print(f"\n{'='*60}")
    print(f"  T={T_bin}  obs_frac={obs_frac}  seed={seed}  [{tag}]")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    t0 = time.time()

    # Adjust k_max so autonomous phase is at least 40% of trial
    k_max = min(20, max(3, T_bin // 4))
    k_min = min(5, k_max - 1) if k_max > 3 else 2

    result = train(DATA_PATH, TrainingConfig(
        model_type="vanilla",
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=min(30, T_bin // 4),
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=False,
        observation_fraction=obs_frac,
        max_trial_length=T_bin,
        k_min=k_min,
        k_max=k_max,
        checkpoint_dir=ckpt_dir,
        log_every=500,
    ))
    train_time = time.time() - t0

    model = result["model"]
    device = result["device"]
    observed_idx = result["observed_idx"]

    data = np.load(DATA_PATH)
    neuron_angles = data["neuron_angles"]
    norm_mean = data["mean"]
    norm_std = data["std"]
    trajectories = data["trajectories"]

    # Truncate val trajectories to match training
    val_traj = trajectories[result["val_idx"]]
    if T_bin < val_traj.shape[1]:
        val_traj = val_traj[:, :T_bin, :]

    pred = evaluate_predictions(
        model, val_traj, neuron_angles, norm_mean, norm_std, device,
        k_eval=min(12, k_max), observed_idx=observed_idx,
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
    er = eigenvalue_analysis(model, ring.h_final[fp_indices[0]])
    svs = singular_value_analysis(model)

    m1_pass = (
        ring.uniformity > 0.8
        and ring.circularity > 0.7
        and gen.mean_abs_drift_deg < 5.0
    )

    print(f"\n  [{tag}] unif={ring.uniformity:.3f}  circ={ring.circularity:.3f}  "
          f"drift={gen.mean_abs_drift_deg:.2f}°  "
          f"{'PASS' if m1_pass else 'FAIL'}  ({train_time:.0f}s)")

    # Save arrays
    np.savez_compressed(str(OUT_DIR / f"{tag}.npz"),
        pca_proj=ring.pca_proj,
        theta_final=ring.theta_final,
        h_final=ring.h_final,
        test_angles=gen.test_angles,
        final_angles=gen.final_angles,
        drift_deg=gen.drift_deg,
        eig_mags=er.magnitudes,
        singular_values=svs,
        observed_idx=observed_idx if observed_idx is not None else np.arange(100),
    )

    # Save config
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        N = 100
        obs_idx = None
        if obs_frac < 1.0:
            rng = np.random.default_rng(42 + 1000)
            n_obs = max(1, int(N * obs_frac))
            obs_idx = np.sort(rng.choice(N, n_obs, replace=False))
        config_dict = {
            'model_type': 'vanilla', 'hidden_dim': 100, 'alpha': 0.5,
            'observation_fraction': obs_frac, 'seed': seed,
            'max_trial_length': T_bin, 'n_epochs': 5000,
            'k_min': k_min, 'k_max': k_max,
            'observed_idx': obs_idx.tolist() if obs_idx is not None else None,
            'best_val_loss': result['best_val_loss'],
        }
        with open(ckpt_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

    return {
        "T_bin": T_bin,
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
        "k_min": k_min,
        "k_max": k_max,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"

    results = []
    if summary_path.exists():
        results = json.load(open(summary_path))
    existing_tags = {r["tag"] for r in results}

    t0 = time.time()
    for T_bin in T_VALUES:
        for obs_frac in OBS_FRACS:
            for seed in SEEDS:
                tag = f"T{T_bin:03d}_obs{int(obs_frac * 100):03d}_seed{seed}"
                if tag in existing_tags:
                    print(f"Skipping {tag} (already exists)")
                    continue
                r = run_one(T_bin, obs_frac, seed)
                results.append(r)
                with open(summary_path, "w") as f:
                    json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  T-AXIS SWEEP COMPLETE  ({total / 60:.1f} min)")
    print(f"{'='*60}\n")

    results_sorted = sorted(results, key=lambda x: (-x["T_bin"], -x["obs_frac"], x["seed"]))
    print(f"{'T':>5} {'obs':>5} {'seed':>5} {'unif':>6} {'circ':>6} {'drift':>7} {'mse':>8} {'pass':>5}")
    print("-" * 55)
    for r in results_sorted:
        print(f"{r['T_bin']:5d} {r['obs_frac']:5.2f} {r['seed']:5d} {r['uniformity']:6.3f} "
              f"{r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{r['mse']:8.4f} {'YES' if r['milestone_1_pass'] else 'NO':>5}")


if __name__ == "__main__":
    main()
