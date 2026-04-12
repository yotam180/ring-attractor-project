#!/usr/bin/env python3
"""
Phase 5B: Baseline training on real HD cell data.

Trains the RNN with full observation on real data and evaluates
whether the learned dynamics form a ring attractor.

For each session, the generalization test uses held-out real data segments
at target angles instead of the teacher simulator.
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

from src.real_data.loading import load_session, list_sessions
from src.real_data.preprocessing import prepare_dataset
from src.train import train, TrainingConfig
from src.train.evaluation import (
    autonomous_fixed_points,
    evaluate_predictions,
    eigenvalue_analysis,
    singular_value_analysis,
)


def generalization_test_real(
    model,
    neuron_angles,
    norm_mean,
    norm_std,
    device,
    trajectories,
    trial_hd_angles,
    n_test: int = 36,
    k_teacher: int = 15,
    T_gen: int = 500,
    observed_idx=None,
):
    """
    Generalization test using real data segments at target angles.

    Instead of generating synthetic bumps, finds real trials closest to each
    test angle and uses them as teacher-forcing seeds.
    """
    test_spacing = 2 * np.pi / n_test
    test_angles = np.linspace(0, 2 * np.pi, n_test, endpoint=False) + test_spacing / 2

    model.eval()
    input_dim = model.input_dim
    T_total_seq = k_teacher + T_gen
    N = trajectories.shape[-1]

    final_angles = np.zeros(n_test)

    for i, theta_test in enumerate(test_angles):
        # Find the trial closest to this test angle
        dists = np.abs(np.angle(np.exp(1j * (trial_hd_angles - theta_test))))
        best_trial = np.argmin(dists)
        traj = trajectories[best_trial]  # (T_trial, N) standardised

        # Build input
        x = np.zeros((1, T_total_seq, input_dim), dtype=np.float32)
        if observed_idx is not None:
            x[0, :k_teacher, :] = traj[:k_teacher][:, observed_idx]
        else:
            x[0, :k_teacher, :] = traj[:k_teacher, :input_dim]
        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            y_pred, _ = model(x_t)

        y_last = y_pred[0, -1].cpu().numpy()
        y_raw = y_last * norm_std + norm_mean
        z = y_raw @ np.exp(1j * neuron_angles)
        final_angles[i] = np.angle(z)

    drift = np.degrees(np.angle(np.exp(1j * (final_angles - test_angles))))
    mean_drift = float(np.abs(drift).mean())

    return {
        "test_angles": test_angles,
        "final_angles": final_angles,
        "drift_deg": drift,
        "mean_abs_drift_deg": mean_drift,
    }


def run_one(data_path, obs_frac, seed, tag, ckpt_dir):
    """Train + evaluate one condition."""
    print(f"\n{'='*60}")
    print(f"  {tag}  obs_frac={obs_frac}  seed={seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    t0 = time.time()

    data = np.load(data_path)
    N = data["trajectories"].shape[-1]

    result = train(data_path, TrainingConfig(
        model_type="vanilla",
        hidden_dim=100,
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=30,
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=False,
        observation_fraction=obs_frac,
        val_split_mode="random",
        checkpoint_dir=ckpt_dir,
        log_every=500,
    ))
    train_time = time.time() - t0

    model = result["model"]
    device = result["device"]
    observed_idx = result["observed_idx"]
    neuron_angles = data["neuron_angles"]
    norm_mean = data["mean"]
    norm_std = data["std"]

    # Ring score (autonomous fixed points)
    ring = autonomous_fixed_points(
        model, neuron_angles, norm_mean, norm_std, device,
    )

    # Generalization test (real data segments)
    gen = generalization_test_real(
        model, neuron_angles, norm_mean, norm_std, device,
        data["trajectories"], data["trial_hd_angles"],
        observed_idx=observed_idx,
    )

    # Prediction metrics on val set
    val_traj = data["trajectories"][result["val_idx"]]
    pred = evaluate_predictions(
        model, val_traj, neuron_angles, norm_mean, norm_std, device,
        observed_idx=observed_idx,
    )

    m1_pass = (
        ring.uniformity > 0.8
        and ring.circularity > 0.7
        and gen["mean_abs_drift_deg"] < 5.0
    )

    print(f"\n  [{tag}] unif={ring.uniformity:.3f}  circ={ring.circularity:.3f}  "
          f"drift={gen['mean_abs_drift_deg']:.2f}°  "
          f"{'PASS' if m1_pass else 'FAIL'}  ({train_time:.0f}s)")

    return {
        "tag": tag,
        "obs_frac": obs_frac,
        "seed": seed,
        "n_neurons": N,
        "uniformity": ring.uniformity,
        "circularity": ring.circularity,
        "spread": ring.spread,
        "drift_mean": gen["mean_abs_drift_deg"],
        "drift_max": float(np.abs(gen["drift_deg"]).max()),
        "mse": pred.mse,
        "angle_error": pred.angle_error_deg,
        "best_val_loss": result["best_val_loss"],
        "milestone_1_pass": m1_pass,
        "train_time_s": train_time,
        "pca_proj": ring.pca_proj,
        "theta_final": ring.theta_final,
        "drift_deg": gen["drift_deg"],
        "test_angles": gen["test_angles"],
    }


def main():
    OUT_DIR = Path("data/real_sweep_results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Top 3 sessions by HD cell count
    sessions_info = list_sessions()[:3]

    SEEDS = [42, 123]
    OBS_FRACS = [1.0]  # baseline only

    # Preprocess sessions
    data_paths = {}
    for info in sessions_info:
        subject = info["subject"]
        data_path = Path(f"data/real_{subject.replace('sub-', '')}.npz")
        if not data_path.exists():
            session = load_session(info["nwb_path"])
            prepare_dataset(session, output_path=data_path)
        data_paths[subject] = str(data_path)

    results = []
    summary_path = OUT_DIR / "summary.json"

    for info in sessions_info:
        subject = info["subject"]
        data_path = data_paths[subject]
        for obs_frac in OBS_FRACS:
            for seed in SEEDS:
                tag = f"real_{subject}_{int(obs_frac*100):03d}_seed{seed}"
                ckpt_dir = f"checkpoints/real_{tag}"

                r = run_one(data_path, obs_frac, seed, tag, ckpt_dir)

                # Save arrays
                np.savez_compressed(str(OUT_DIR / f"{tag}.npz"),
                    pca_proj=r.pop("pca_proj"),
                    theta_final=r.pop("theta_final"),
                    drift_deg=r.pop("drift_deg"),
                    test_angles=r.pop("test_angles"),
                )
                r["subject"] = subject
                r["n_hd"] = info["n_hd"]
                results.append(r)

                with open(summary_path, "w") as f:
                    json.dump(results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}\n")
    print(f"{'subject':<12} {'seed':>5} {'N_HD':>5} {'unif':>6} {'circ':>6} {'drift':>7} {'pass':>5}")
    print("-" * 55)
    for r in results:
        print(f"{r['subject']:<12} {r['seed']:5d} {r['n_hd']:5d} {r['uniformity']:6.3f} "
              f"{r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{'YES' if r['milestone_1_pass'] else 'NO':>5}")


if __name__ == "__main__":
    main()
