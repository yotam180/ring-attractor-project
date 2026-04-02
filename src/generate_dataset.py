#!/usr/bin/env python
"""
Generate training dataset for the ring attractor RNN.

Groups
------
  A – Bump maintenance: converged bump at each of 36 angles, recorded
      during steady-state autonomous dynamics.
  B – Perturbation recovery: Gaussian noise added to a converged bump,
      then the simulator recovers to a clean bump.

Each trial: simulator -> Poisson spikes -> temporal binning -> causal smoothing.
The full dataset is standardised per neuron (zero mean, unit variance).

Output
------
data/ring_attractor_dataset.npz  with keys:
    trajectories      (n_trials, T_bin, N)  standardised smoothed counts
    trajectories_raw  (n_trials, T_bin, N)  un-standardised smoothed counts
    groups            (n_trials,)           'A' or 'B'
    target_angles     (n_trials,)           cue angle (rad)
    neuron_angles     (N,)                  preferred angle per neuron
    mean              (N,)                  per-neuron mean (for de-standardisation)
    std               (N,)                  per-neuron std

Usage
-----
    python src/generate_dataset.py
"""

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.ring_attractor import RingAttractor, SpikeProcessor, decode_theta
from src.ring_attractor import defaults as D

# ── Dataset generation parameters (not in defaults — specific to this script)
SEED = 42
OUT_DIR = _ROOT / "data"
OUT_FILE = OUT_DIR / "ring_attractor_dataset.npz"


def generate_dataset():
    """Generate Group A (maintenance) and Group B (perturbation recovery) trials."""
    ring = RingAttractor()  # uses defaults from src.ring_attractor.defaults
    sp = SpikeProcessor()   # uses defaults from src.ring_attractor.defaults

    target_angles = np.linspace(0, 2 * np.pi, D.N_ANGLES, endpoint=False)
    rng = np.random.default_rng(SEED)

    trajectories = []
    groups = []
    angles_list = []

    # Per-trial diagnostics from raw simulator rates (before spike processing)
    raw_angle_errors = []  # degrees, from last 1000 integration steps
    raw_confidences = []
    raw_early_confidences = []  # Group B only: first 500 integration steps

    t0 = time.time()
    for i, theta in enumerate(target_angles):
        # ── Group A: Bump maintenance ────────────────────────────────
        T_total_a = D.T_CUE + D.T_SETTLE + D.T_RECORD
        res_a = ring.simulate(
            T=T_total_a,
            cue_angles=[theta],
            cue_duration=D.T_CUE,
            seed=int(rng.integers(1 << 31)),
        )

        # Raw-rate diagnostics (validates the *simulator*, not the pipeline)
        late_raw = res_a.rates[-1000:]
        theta_dec, conf_dec = decode_theta(late_raw, ring.angles)
        z = np.exp(1j * theta_dec).mean()
        err = np.abs(np.angle(np.exp(1j * (np.angle(z) - theta))))
        raw_angle_errors.append(np.degrees(err))
        raw_confidences.append(conf_dec.mean())
        raw_early_confidences.append(np.nan)  # N/A for Group A

        # Spike processing on the recording window
        rec_rates_a = res_a.rates[D.T_CUE + D.T_SETTLE :]
        data_a = sp.process(rec_rates_a, seed=int(rng.integers(1 << 31)))
        trajectories.append(data_a.smoothed)
        groups.append("A")
        angles_list.append(theta)

        # ── Group B: Perturbation recovery ───────────────────────────
        converged = res_a.rates[D.T_CUE + D.T_SETTLE - 1]
        sigma_perturb = D.SIGMA_PERTURB_FRAC * converged.max()
        perturbed = converged + sigma_perturb * rng.standard_normal(D.N)

        res_b = ring.simulate(
            T=D.T_RECORD,
            cue_angles=None,
            init_rates=perturbed,
            seed=int(rng.integers(1 << 31)),
        )

        # Raw-rate diagnostics for Group B
        late_raw_b = res_b.rates[-1000:]
        theta_dec_b, conf_dec_b = decode_theta(late_raw_b, ring.angles)
        z_b = np.exp(1j * theta_dec_b).mean()
        err_b = np.abs(np.angle(np.exp(1j * (np.angle(z_b) - theta))))
        raw_angle_errors.append(np.degrees(err_b))
        raw_confidences.append(conf_dec_b.mean())
        # Early confidence (first 500 steps) — measures perturbation strength
        _, early_conf_b = decode_theta(res_b.rates[:500], ring.angles)
        raw_early_confidences.append(early_conf_b.mean())

        data_b = sp.process(res_b.rates, seed=int(rng.integers(1 << 31)))
        trajectories.append(data_b.smoothed)
        groups.append("B")
        angles_list.append(theta)

        if (i + 1) % 6 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"  angle {i + 1:2d}/{D.N_ANGLES}  "
                f"({np.degrees(theta):5.1f} deg)  "
                f"[{elapsed:.1f}s]"
            )

    # ── Stack and standardise ────────────────────────────────────────
    trajectories = np.stack(trajectories)  # (n_trials, T_bin, N)
    groups = np.array(groups)
    target_angles_arr = np.array(angles_list)

    # Per-neuron standardisation (across all trials and timesteps)
    mean = trajectories.mean(axis=(0, 1))  # (N,)
    std = trajectories.std(axis=(0, 1))  # (N,)
    std[std < 1e-8] = 1.0
    trajectories_std = (trajectories - mean) / std

    return dict(
        trajectories=trajectories_std,
        trajectories_raw=trajectories,
        groups=groups,
        target_angles=target_angles_arr,
        neuron_angles=ring.angles,
        mean=mean,
        std=std,
        # Diagnostics (small, saved for reference)
        raw_angle_errors=np.array(raw_angle_errors),
        raw_confidences=np.array(raw_confidences),
        raw_early_confidences=np.array(raw_early_confidences),
    )


def validate(ds):
    """Two-layer validation: simulator correctness, then spike pipeline quality."""
    traj = ds["trajectories_raw"]
    groups = ds["groups"]
    angles = ds["neuron_angles"]
    targets = ds["target_angles"]
    raw_errs = ds["raw_angle_errors"]
    raw_confs = ds["raw_confidences"]
    raw_early = ds["raw_early_confidences"]

    n_trials, T_bin, N = traj.shape
    a_mask = groups == "A"
    b_mask = groups == "B"

    print(f"\n── Dataset shape ──")
    print(f"  Trials: {n_trials} (A: {a_mask.sum()}, B: {b_mask.sum()})")
    print(f"  Time bins: {T_bin},  Neurons: {N}")
    print(f"  Raw spike-count range: [{traj.min():.0f}, {traj.max():.0f}]")
    print(f"  Standardised range:    [{ds['trajectories'].min():.2f}, "
          f"{ds['trajectories'].max():.2f}]")

    # ── Layer 1: Simulator (raw rates, before spike processing) ──────
    print(f"\n── Layer 1: Simulator (raw rates) ──")
    errs_a = raw_errs[a_mask]
    errs_b = raw_errs[b_mask]
    conf_a = raw_confs[a_mask]
    conf_b_late = raw_confs[b_mask]
    conf_b_early = raw_early[b_mask]

    print(f"  Group A  angle |error|: mean {errs_a.mean():.2f} deg, "
          f"max {errs_a.max():.2f} deg")
    print(f"  Group A  confidence:    mean {conf_a.mean():.3f}")
    print(f"  Group B  angle |error|: mean {errs_b.mean():.2f} deg, "
          f"max {errs_b.max():.2f} deg  (after recovery)")
    print(f"  Group B  confidence:    {conf_b_early.mean():.3f} (early) "
          f"-> {conf_b_late.mean():.3f} (late)")

    if errs_a.max() > 20.0:
        print(f"  WARNING: Group A raw-rate max error > 20 deg — "
              f"simulator may have failed to form a bump")
    elif errs_a.max() > 5.0:
        print(f"  (Note: {errs_a.max():.1f} deg max error is expected — "
              f"noise-induced angular drift over {D.T_RECORD} steps)")

    # ── Layer 2: Spike pipeline (processed data) ─────────────────────
    print(f"\n── Layer 2: Spike pipeline (smoothed counts) ──")

    # Group A: angle accuracy and temporal stability
    proc_errors_a = []
    drift_a = []
    for idx in np.where(a_mask)[0]:
        trial = traj[idx]
        tgt = targets[idx]
        # Late angle accuracy
        theta_late, _ = decode_theta(trial[-50:], angles)
        z_late = np.exp(1j * theta_late).mean()
        err = np.abs(np.angle(np.exp(1j * (np.angle(z_late) - tgt))))
        proc_errors_a.append(np.degrees(err))
        # Temporal stability: early vs late decoded angle
        theta_early, _ = decode_theta(trial[:20], angles)
        z_early = np.exp(1j * theta_early).mean()
        d = np.abs(np.angle(np.exp(1j * (np.angle(z_late) - np.angle(z_early)))))
        drift_a.append(np.degrees(d))

    proc_errors_a = np.array(proc_errors_a)
    drift_a = np.array(drift_a)
    print(f"  Group A  angle |error|: mean {proc_errors_a.mean():.2f} deg, "
          f"max {proc_errors_a.max():.2f} deg")
    print(f"  Group A  drift (early vs late): "
          f"mean {drift_a.mean():.2f} deg, max {drift_a.max():.2f} deg")

    # Group B: confidence change and angle convergence
    proc_conf_early_b = []
    proc_conf_late_b = []
    proc_errors_b = []
    for idx in np.where(b_mask)[0]:
        trial = traj[idx]
        tgt = targets[idx]
        _, ce = decode_theta(trial[:10], angles)
        _, cl = decode_theta(trial[-10:], angles)
        proc_conf_early_b.append(ce.mean())
        proc_conf_late_b.append(cl.mean())
        # Angle convergence: does recovered bump match target?
        theta_late, _ = decode_theta(trial[-50:], angles)
        z_late = np.exp(1j * theta_late).mean()
        err = np.abs(np.angle(np.exp(1j * (np.angle(z_late) - tgt))))
        proc_errors_b.append(np.degrees(err))

    print(f"  Group B  confidence:    "
          f"{np.mean(proc_conf_early_b):.3f} (early) -> "
          f"{np.mean(proc_conf_late_b):.3f} (late)")
    print(f"  Group B  angle |error|: mean {np.mean(proc_errors_b):.2f} deg  "
          f"(recovered to correct angle?)")

    # ── Per-neuron statistics ────────────────────────────────────────
    print(f"\n── Standardisation ──")
    print(f"  Per-neuron mean: [{ds['mean'].min():.2f}, {ds['mean'].max():.2f}]  "
          f"(should be similar — uniform angle coverage)")
    print(f"  Per-neuron std:  [{ds['std'].min():.2f}, {ds['std'].max():.2f}]")


def main():
    print(f"Generating ring attractor dataset...")
    print(
        f"  {D.N_ANGLES} angles x 2 groups = {2 * D.N_ANGLES} trials"
    )
    print(
        f"  {D.T_RECORD} integration steps -> "
        f"{D.T_RECORD // D.BIN_FACTOR} time bins per trial"
    )
    print()

    ds = generate_dataset()
    validate(ds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUT_FILE), **ds)
    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nSaved to {OUT_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
