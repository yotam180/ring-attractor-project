#!/usr/bin/env python3
"""
Phase 5B+C: Real data baseline + k/N sweep.

For each of the top 3 sessions by HD cell count:
  - Baseline: full observation (obs_frac=1.0), 2 seeds
  - k/N sweep: obs_frac ∈ {0.75, 0.5, 0.25, 0.15, 0.1}, 2 seeds

Evaluation:
  - Autonomous fixed-point analysis (uniformity, circularity, spread)
  - Generalization test using held-out real data segments
  - Prediction metrics (MSE, angle error)

Results saved to data/real_sweep_results/summary.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.real_data.loading import load_session, list_sessions
from src.real_data.preprocessing import prepare_dataset
from src.train import train, TrainingConfig
from src.train.evaluation import (
    autonomous_fixed_points,
    evaluate_predictions,
    singular_value_analysis,
)


# ── Configuration ────────────────────────────────────────────────────────

TOP_K_SESSIONS = 3
OBS_FRACS = [1.0, 0.75, 0.5, 0.25, 0.15, 0.1]
SEEDS = [42, 123]
N_EPOCHS = 3000
HIDDEN_DIM = 128
OUT_DIR = Path("data/real_sweep_results")


# ── Generalization test (real data version) ──────────────────────────────

def generalization_test_real(
    model, neuron_angles, norm_mean, norm_std, device,
    trajectories, trial_hd_angles,
    n_test=36, k_teacher=15, T_gen=500, observed_idx=None,
):
    """Find real trials near each test angle, teacher-force, run autonomous."""
    test_spacing = 2 * np.pi / n_test
    test_angles = np.linspace(0, 2 * np.pi, n_test, endpoint=False) + test_spacing / 2

    model.eval()
    input_dim = model.input_dim
    T_total = k_teacher + T_gen

    final_angles = np.zeros(n_test)
    for i, theta in enumerate(test_angles):
        dists = np.abs(np.angle(np.exp(1j * (trial_hd_angles - theta))))
        best = np.argmin(dists)
        traj = trajectories[best]

        x = np.zeros((1, T_total, input_dim), dtype=np.float32)
        if observed_idx is not None:
            x[0, :k_teacher] = traj[:k_teacher][:, observed_idx]
        else:
            x[0, :k_teacher] = traj[:k_teacher, :input_dim]

        with torch.no_grad():
            y_pred, _ = model(torch.from_numpy(x).to(device))

        y_last = y_pred[0, -1].cpu().numpy()
        y_raw = y_last * norm_std + norm_mean
        final_angles[i] = np.angle(y_raw @ np.exp(1j * neuron_angles))

    drift = np.degrees(np.angle(np.exp(1j * (final_angles - test_angles))))
    return {
        "test_angles": test_angles,
        "final_angles": final_angles,
        "drift_deg": drift,
        "mean_abs_drift_deg": float(np.abs(drift).mean()),
    }


# ── Single run ───────────────────────────────────────────────────────────

def run_one(data_path, obs_frac, seed, tag, ckpt_dir):
    print(f"\n{'='*60}")
    print(f"  {tag}  obs_frac={obs_frac}  seed={seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    t0 = time.time()

    result = train(data_path, TrainingConfig(
        model_type="vanilla",
        hidden_dim=HIDDEN_DIM,
        n_epochs=N_EPOCHS,
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

    d = np.load(data_path)
    neuron_angles = d["neuron_angles"]
    norm_mean = d["mean"]
    norm_std = d["std"]

    # Ring score
    ring = autonomous_fixed_points(model, neuron_angles, norm_mean, norm_std, device)

    # Generalization test
    gen = generalization_test_real(
        model, neuron_angles, norm_mean, norm_std, device,
        d["trajectories"], d["trial_hd_angles"],
        observed_idx=observed_idx,
    )

    # Prediction metrics
    val_traj = d["trajectories"][result["val_idx"]]
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

    # Save arrays
    np.savez_compressed(str(OUT_DIR / f"{tag}.npz"),
        pca_proj=ring.pca_proj, theta_final=ring.theta_final,
        h_final=ring.h_final,
        test_angles=gen["test_angles"], final_angles=gen["final_angles"],
        drift_deg=gen["drift_deg"],
    )

    return {
        "tag": tag, "obs_frac": obs_frac, "seed": seed,
        "uniformity": ring.uniformity, "circularity": ring.circularity,
        "spread": ring.spread,
        "drift_mean": gen["mean_abs_drift_deg"],
        "drift_max": float(np.abs(gen["drift_deg"]).max()),
        "mse": pred.mse, "angle_error": pred.angle_error_deg,
        "best_val_loss": result["best_val_loss"],
        "milestone_1_pass": m1_pass, "train_time_s": train_time,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"

    # Resume from existing results
    results = []
    if summary_path.exists():
        results = json.load(open(summary_path))
    existing_tags = {r["tag"] for r in results}

    # Top sessions
    sessions_info = list_sessions()[:TOP_K_SESSIONS]
    labels = [f"{s['subject']} ({s['n_hd']} HD)" for s in sessions_info]
    print(f"Sessions: {labels}")

    # Preprocess
    data_paths = {}
    for info in sessions_info:
        subject = info["subject"]
        name = subject.replace("sub-", "")
        data_path = Path(f"data/real_{name}.npz")
        if not data_path.exists():
            session = load_session(info["nwb_path"])
            prepare_dataset(session, output_path=data_path)
        data_paths[subject] = str(data_path)

    t0 = time.time()
    for info in sessions_info:
        subject = info["subject"]
        name = subject.replace("sub-", "")
        data_path = data_paths[subject]

        # Add session-level info to results
        d = np.load(data_path)
        n_hd = d["trajectories"].shape[-1]
        n_trials = d["trajectories"].shape[0]

        for obs_frac in OBS_FRACS:
            for seed in SEEDS:
                tag = f"real_{name}_obs{int(obs_frac*100):03d}_seed{seed}"
                if tag in existing_tags:
                    print(f"Skipping {tag} (exists)")
                    continue

                ckpt_dir = f"checkpoints/real_{tag}"
                r = run_one(data_path, obs_frac, seed, tag, ckpt_dir)
                r["subject"] = subject
                r["n_hd"] = n_hd
                r["n_trials"] = n_trials
                results.append(r)

                with open(summary_path, "w") as f:
                    json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  REAL DATA SWEEP COMPLETE  ({total / 60:.1f} min)")
    print(f"{'='*60}\n")

    # Summary table
    print(f"{'subject':<10} {'obs':>5} {'seed':>5} {'N':>4} {'unif':>6} {'circ':>6} "
          f"{'drift':>7} {'mse':>8} {'pass':>5}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: (x.get("subject", ""), -x["obs_frac"], x["seed"])):
        s = r.get("subject", "?")[-5:]
        print(f"{s:<10} {r['obs_frac']:5.2f} {r['seed']:5d} {r.get('n_hd', 0):4d} "
              f"{r['uniformity']:6.3f} {r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{r['mse']:8.4f} {'YES' if r['milestone_1_pass'] else 'NO':>5}")


if __name__ == "__main__":
    main()
