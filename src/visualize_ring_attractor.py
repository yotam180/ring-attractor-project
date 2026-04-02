"""
Ring attractor visualization demo.

Generates a multi-panel figure showing that the simulator produces a genuine
ring attractor:
  1. Firing rate heatmap over time (neurons × time)
  2. Decoded θ and confidence over time
  3. Polar ring snapshot at several timepoints
  4. Bump maintenance: θ stability across many trials
  5. Spontaneous bump formation from noise
  6. Perturbation recovery dynamics

Run:  python src/visualize_ring_attractor.py   (saves figures to figs/)

Can also be imported and called from a Jupyter notebook:
    from src.visualize_ring_attractor import run_all
    run_all()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.simulator import simulate, decode_theta, simulate_trial


FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────

def _polar_snapshot(ax, rates, angles, title=""):
    """Draw a polar bar plot of neuron firing rates."""
    width = 2 * np.pi / len(angles)
    r_pos = np.maximum(rates, 0.0)
    r_norm = r_pos / (r_pos.max() + 1e-12)
    colors = plt.cm.inferno(r_norm)
    ax.bar(angles, r_pos, width=width, bottom=0, color=colors, edgecolor="none")
    ax.set_yticks([])
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
    ax.set_title(title, fontsize=9, pad=8)


def _circ_error(theta_hat, theta_true):
    """Signed circular distance in degrees."""
    d = theta_hat - theta_true
    return np.degrees(np.arctan2(np.sin(d), np.cos(d)))


# ── Figure 1: Single trial overview ─────────────────────────────────────

def fig_single_trial(seed=42):
    """Heatmap + decoded theta/confidence + polar snapshots for one cued trial."""
    target = np.pi / 3  # 60°
    res = simulate(T=7500, cue_angles=[target], cue_duration=2000,
                   cue_amplitude=2.0, sigma=0.1, seed=seed)
    rates = res["rates"]
    theta = res["theta"]
    conf = res["confidence"]
    angles = res["angles"]
    T, N = rates.shape

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Ring Attractor — Single Cued Trial (θ_target = 60°)", fontsize=13)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[2, 1, 1])

    # -- Row 1: heatmap
    ax_heat = fig.add_subplot(gs[0, :3])
    # Sort neurons by preferred angle for a nice display
    order = np.argsort(angles)
    im = ax_heat.imshow(
        rates[:, order].T, aspect="auto", origin="lower",
        cmap="inferno", interpolation="nearest",
    )
    ax_heat.set_xlabel("Integration step")
    ax_heat.set_ylabel("Neuron (sorted by pref. angle)")
    ax_heat.axvline(2000, color="cyan", ls="--", lw=1, label="cue off")
    ax_heat.legend(fontsize=8, loc="upper right")
    plt.colorbar(im, ax=ax_heat, label="Firing rate", shrink=0.8)

    # -- Polar snapshots at 4 timepoints
    snap_times = [500, 2000, 4000, 7000]
    for i, t_snap in enumerate(snap_times):
        ax_p = fig.add_subplot(gs[0, 3], projection="polar") if i == 0 else None
        # Use small inset polar axes manually
    # Actually, let's put polar snapshots in row 3
    for i, t_snap in enumerate(snap_times):
        ax_p = fig.add_subplot(gs[2, i], projection="polar")
        _polar_snapshot(ax_p, rates[t_snap], angles, title=f"t={t_snap}")

    # -- Row 2: theta and confidence
    ax_theta = fig.add_subplot(gs[1, :2])
    ax_theta.plot(np.degrees(theta), lw=0.5, color="tab:blue", alpha=0.8)
    ax_theta.axhline(np.degrees(target), color="red", ls="--", lw=1, label=f"target={np.degrees(target):.0f}°")
    ax_theta.axvline(2000, color="cyan", ls="--", lw=0.8)
    ax_theta.set_ylabel("Decoded θ (°)")
    ax_theta.set_xlabel("Integration step")
    ax_theta.legend(fontsize=8)
    ax_theta.set_ylim(-200, 200)

    ax_conf = fig.add_subplot(gs[1, 2:])
    ax_conf.plot(conf, lw=0.8, color="tab:green")
    ax_conf.axvline(2000, color="cyan", ls="--", lw=0.8)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_xlabel("Integration step")
    ax_conf.set_ylim(0, 1)
    ax_conf.axhline(0.84, color="gray", ls=":", lw=0.5, label="equilibrium ~0.84")
    ax_conf.legend(fontsize=8)

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "01_single_trial.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Figure 2: Multi-angle ring coverage ─────────────────────────────────

def fig_multi_angle(n_angles=36, seed_base=0):
    """Show that 36 cues at different angles all produce stable bumps."""
    targets = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    final_thetas = []
    final_confs = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle(f"Ring Attractor — {n_angles} Cued Trials at Different Angles", fontsize=13)

    # Collect results
    all_late_theta = []
    all_late_conf = []
    for i, tgt in enumerate(targets):
        res = simulate(T=7500, cue_angles=[tgt], cue_duration=2000, seed=seed_base + i * 7)
        late = slice(4000, 7500)
        z = np.exp(1j * res["theta"][late]).mean()
        final_thetas.append(np.angle(z))
        final_confs.append(res["confidence"][late].mean())
        all_late_theta.append(res["theta"][late])
        all_late_conf.append(res["confidence"][late])

    final_thetas = np.array(final_thetas)
    final_confs = np.array(final_confs)

    # Panel 1: target vs decoded
    ax = axes[0]
    errors = _circ_error(final_thetas, targets)
    ax.scatter(np.degrees(targets), np.degrees(final_thetas), s=20, c="tab:blue", zorder=3)
    ax.plot([0, 360], [0, 360], "k--", lw=0.8, label="identity")
    ax.set_xlabel("Target θ (°)")
    ax.set_ylabel("Decoded θ (°)")
    ax.set_title("Angle accuracy")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # Panel 2: angle error
    ax = axes[1]
    ax.bar(np.degrees(targets), errors, width=8, color="tab:orange")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Target θ (°)")
    ax.set_ylabel("Error (°)")
    ax.set_title(f"Angle error (mean={np.abs(errors).mean():.1f}°)")
    ax.set_ylim(-10, 10)

    # Panel 3: confidence distribution
    ax = axes[2]
    ax.bar(np.degrees(targets), final_confs, width=8, color="tab:green")
    ax.set_xlabel("Target θ (°)")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Confidence (mean={final_confs.mean():.3f})")
    ax.set_ylim(0, 1)

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "02_multi_angle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Figure 3: Long-term stability ───────────────────────────────────────

def fig_long_term_stability(seed=99):
    """100k free-run steps showing bump doesn't decay."""
    target = np.pi / 4  # Use 45° to avoid atan2 wrapping artifacts at ±180°
    res = simulate(T=102000, cue_angles=[target], cue_duration=2000, seed=seed)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    fig.suptitle("Long-Term Bump Stability (100k steps free-run after cue)", fontsize=13)

    t_axis = np.arange(res["rates"].shape[0])

    # Peak rate over time
    axes[0].plot(t_axis, res["rates"].max(axis=1), lw=0.5, color="tab:red")
    axes[0].axvline(2000, color="cyan", ls="--", lw=0.8, label="cue off")
    axes[0].set_ylabel("Peak firing rate")
    axes[0].legend(fontsize=8)

    # Confidence over time
    axes[1].plot(t_axis, res["confidence"], lw=0.5, color="tab:green")
    axes[1].axvline(2000, color="cyan", ls="--", lw=0.8)
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim(0, 1)

    # Decoded theta over time (unwrap to avoid jumps)
    theta_unwrap = np.unwrap(res["theta"])
    axes[2].plot(t_axis, np.degrees(theta_unwrap), lw=0.3, color="tab:blue", alpha=0.7)
    axes[2].axhline(np.degrees(target), color="red", ls="--", lw=1, label=f"target={np.degrees(target):.0f}°")
    axes[2].axvline(2000, color="cyan", ls="--", lw=0.8)
    axes[2].set_ylabel("Decoded θ (°)")
    axes[2].set_xlabel("Integration step")
    axes[2].legend(fontsize=8)

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "03_long_term_stability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Figure 4: Spontaneous bump from noise ────────────────────────────────

def fig_noise_bump(seed=77):
    """No cue — show bump emerging from noise."""
    res = simulate(T=50000, cue_angles=None, sigma=0.1, seed=seed, init_noise_scale=0.1)
    rates = res["rates"]
    angles = res["angles"]

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Spontaneous Bump Formation (no cue, noise only)", fontsize=13)
    gs = GridSpec(2, 5, figure=fig)

    # Heatmap
    ax_heat = fig.add_subplot(gs[0, :])
    order = np.argsort(angles)
    im = ax_heat.imshow(rates[:, order].T, aspect="auto", origin="lower",
                        cmap="inferno", interpolation="nearest")
    ax_heat.set_ylabel("Neuron (sorted)")
    ax_heat.set_xlabel("Integration step")
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label="Rate")

    # Polar snapshots
    snap_times = [500, 2000, 5000, 20000, 45000]
    for i, t_snap in enumerate(snap_times):
        ax_p = fig.add_subplot(gs[1, i], projection="polar")
        _polar_snapshot(ax_p, rates[t_snap], angles, title=f"t={t_snap}")

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "04_noise_bump_formation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Figure 5: Perturbation recovery ─────────────────────────────────────

def fig_perturbation_recovery(seed=123):
    """Form bump, perturb, show recovery."""
    target = 0.0
    res1 = simulate(T=5000, cue_angles=[target], cue_duration=2000, seed=seed)
    clean_rates = res1["rates"][-1].copy()

    # Perturb
    rng = np.random.default_rng(456)
    perturbed = clean_rates + 0.5 * rng.standard_normal(clean_rates.shape)

    res2 = simulate(T=20000, cue_angles=None, sigma=0.1, seed=789, init_rates=perturbed)
    angles = res2["angles"]
    rates = res2["rates"]

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Perturbation Recovery (large noise added to clean bump)", fontsize=13)
    gs = GridSpec(2, 5, figure=fig, height_ratios=[2, 1])

    # Heatmap
    ax_heat = fig.add_subplot(gs[0, :4])
    order = np.argsort(angles)
    im = ax_heat.imshow(rates[:, order].T, aspect="auto", origin="lower",
                        cmap="inferno", interpolation="nearest")
    ax_heat.set_ylabel("Neuron")
    ax_heat.set_xlabel("Step (after perturbation)")
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label="Rate")

    # Confidence timeline
    ax_c = fig.add_subplot(gs[0, 4])
    ax_c.plot(res2["confidence"], np.arange(len(res2["confidence"])), lw=0.8, color="tab:green")
    ax_c.set_xlabel("Confidence")
    ax_c.set_ylabel("Step")
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 1)

    # Polar snapshots
    snap_times = [0, 500, 2000, 5000, 15000]
    for i, t_snap in enumerate(snap_times):
        ax_p = fig.add_subplot(gs[1, i], projection="polar")
        _polar_snapshot(ax_p, rates[t_snap], angles, title=f"t={t_snap}")

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "05_perturbation_recovery.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Figure 6: Spike pipeline demo ───────────────────────────────────────

def fig_spike_pipeline(seed=42):
    """Show the full pipeline: rates → spikes → binned → smoothed."""
    from src.simulator import generate_spikes, bin_spikes, smooth_bins
    target = np.pi / 4
    res = simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=seed)
    rates = res["rates"]
    angles = res["angles"]
    order = np.argsort(angles)

    spikes = generate_spikes(rates, dt=0.01, rate_scale=100, seed=seed)
    binned = bin_spikes(spikes, bin_factor=50)
    smoothed = smooth_bins(binned, window=3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Data Pipeline: Rates → Spikes → Binned → Smoothed", fontsize=13)

    panels = [
        (rates[:, order].T, "Raw firing rates (100 neurons × 7500 steps)", "inferno"),
        (spikes[:, order].T, "Poisson spikes", "gray_r"),
        (binned[:, order].T, f"Binned spikes (bin={50})", "inferno"),
        (smoothed[:, order].T, f"Smoothed (window={3})", "inferno"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Neuron")
        ax.set_xlabel("Time bin" if data.shape[1] < 1000 else "Step")
        plt.colorbar(im, ax=ax, shrink=0.8)

    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, "06_spike_pipeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Run all ──────────────────────────────────────────────────────────────

def run_all():
    """Generate all figures."""
    print("Generating ring attractor visualization figures...")
    paths = []
    paths.append(fig_single_trial())
    paths.append(fig_multi_angle())
    paths.append(fig_long_term_stability())
    paths.append(fig_noise_bump())
    paths.append(fig_perturbation_recovery())
    paths.append(fig_spike_pipeline())
    print(f"\nAll {len(paths)} figures saved to {FIG_DIR}/")
    return paths


if __name__ == "__main__":
    run_all()
