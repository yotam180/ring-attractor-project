"""
Ring attractor visualisation — generates 6 diagnostic figures to ``figs/``.

Run:  python -m src.visualize_ring_attractor
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.ring_attractor import RingAttractor, SpikeProcessor
from src.ring_attractor.plotting import (
    polar_snapshot,
    rate_heatmap,
    circ_error,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
RING = RingAttractor()


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    _ensure_fig_dir()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# -- Figure 1 ---------------------------------------------------------------

def fig_single_trial(seed=42):
    target = np.pi / 3
    res = RING.simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=seed)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Ring Attractor — Single Cued Trial (θ_target = 60°)", fontsize=13)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[2, 1, 1])

    ax_heat = fig.add_subplot(gs[0, :3])
    im = rate_heatmap(ax_heat, res, cue_off_step=2000)
    plt.colorbar(im, ax=ax_heat, label="Firing rate", shrink=0.8)

    for i, t_snap in enumerate([500, 2000, 4000, 7000]):
        ax_p = fig.add_subplot(gs[2, i], projection="polar")
        polar_snapshot(ax_p, res.rates[t_snap], res.angles, title=f"t={t_snap}")

    ax_theta = fig.add_subplot(gs[1, :2])
    ax_theta.plot(np.degrees(res.theta), lw=0.5, color="tab:blue", alpha=0.8)
    ax_theta.axhline(np.degrees(target), color="red", ls="--", lw=1, label=f"target={np.degrees(target):.0f}°")
    ax_theta.axvline(2000, color="cyan", ls="--", lw=0.8)
    ax_theta.set_ylabel("Decoded θ (°)")
    ax_theta.set_xlabel("Integration step")
    ax_theta.legend(fontsize=8)
    ax_theta.set_ylim(-200, 200)

    ax_conf = fig.add_subplot(gs[1, 2:])
    ax_conf.plot(res.confidence, lw=0.8, color="tab:green")
    ax_conf.axvline(2000, color="cyan", ls="--", lw=0.8)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_xlabel("Integration step")
    ax_conf.set_ylim(0, 1)
    ax_conf.axhline(0.84, color="gray", ls=":", lw=0.5, label="equilibrium ~0.84")
    ax_conf.legend(fontsize=8)

    return _save(fig, "01_single_trial.png")


# -- Figure 2 ---------------------------------------------------------------

def fig_multi_angle(n_angles=36, seed_base=0):
    targets = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    final_thetas, final_confs = [], []

    for i, tgt in enumerate(targets):
        res = RING.simulate(T=7500, cue_angles=[tgt], cue_duration=2000, seed=seed_base + i * 7)
        z = np.exp(1j * res.theta[4000:]).mean()
        final_thetas.append(np.angle(z))
        final_confs.append(res.confidence[4000:].mean())

    final_thetas = np.array(final_thetas)
    final_confs = np.array(final_confs)
    errors = circ_error(final_thetas, targets)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle(f"Ring Attractor — {n_angles} Cued Trials at Different Angles", fontsize=13)

    axes[0].scatter(np.degrees(targets), np.degrees(final_thetas), s=20, c="tab:blue", zorder=3)
    axes[0].plot([0, 360], [0, 360], "k--", lw=0.8, label="identity")
    axes[0].set_xlabel("Target θ (°)"); axes[0].set_ylabel("Decoded θ (°)")
    axes[0].set_title("Angle accuracy"); axes[0].legend(fontsize=8); axes[0].set_aspect("equal")

    axes[1].bar(np.degrees(targets), errors, width=8, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Target θ (°)"); axes[1].set_ylabel("Error (°)")
    axes[1].set_title(f"Angle error (mean={np.abs(errors).mean():.1f}°)"); axes[1].set_ylim(-10, 10)

    axes[2].bar(np.degrees(targets), final_confs, width=8, color="tab:green")
    axes[2].set_xlabel("Target θ (°)"); axes[2].set_ylabel("Confidence")
    axes[2].set_title(f"Confidence (mean={final_confs.mean():.3f})"); axes[2].set_ylim(0, 1)

    return _save(fig, "02_multi_angle.png")


# -- Figure 3 ---------------------------------------------------------------

def fig_long_term_stability(seed=99):
    target = np.pi / 4
    res = RING.simulate(T=102000, cue_angles=[target], cue_duration=2000, seed=seed)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    fig.suptitle("Long-Term Bump Stability (100k steps free-run after cue)", fontsize=13)
    t = np.arange(len(res.theta))

    axes[0].plot(t, res.rates.max(axis=1), lw=0.5, color="tab:red")
    axes[0].axvline(2000, color="cyan", ls="--", lw=0.8, label="cue off")
    axes[0].set_ylabel("Peak firing rate"); axes[0].legend(fontsize=8)

    axes[1].plot(t, res.confidence, lw=0.5, color="tab:green")
    axes[1].axvline(2000, color="cyan", ls="--", lw=0.8)
    axes[1].set_ylabel("Confidence"); axes[1].set_ylim(0, 1)

    axes[2].plot(t, np.degrees(np.unwrap(res.theta)), lw=0.3, color="tab:blue", alpha=0.7)
    axes[2].axhline(np.degrees(target), color="red", ls="--", lw=1, label=f"target={np.degrees(target):.0f}°")
    axes[2].axvline(2000, color="cyan", ls="--", lw=0.8)
    axes[2].set_ylabel("Decoded θ (°)"); axes[2].set_xlabel("Integration step"); axes[2].legend(fontsize=8)

    return _save(fig, "03_long_term_stability.png")


# -- Figure 4 ---------------------------------------------------------------

def fig_noise_bump(seed=77):
    res = RING.simulate(T=50000, cue_angles=None, seed=seed, init_noise_scale=0.1)

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Spontaneous Bump Formation (no cue, noise only)", fontsize=13)
    gs = GridSpec(2, 5, figure=fig)

    ax_heat = fig.add_subplot(gs[0, :])
    im = rate_heatmap(ax_heat, res)
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label="Rate")

    for i, t_snap in enumerate([500, 2000, 5000, 20000, 45000]):
        ax_p = fig.add_subplot(gs[1, i], projection="polar")
        polar_snapshot(ax_p, res.rates[t_snap], res.angles, title=f"t={t_snap}")

    return _save(fig, "04_noise_bump_formation.png")


# -- Figure 5 ---------------------------------------------------------------

def fig_perturbation_recovery(seed=123):
    res1 = RING.simulate(T=5000, cue_angles=[0.0], cue_duration=2000, seed=seed)
    rng = np.random.default_rng(456)
    perturbed = res1.rates[-1] + 0.5 * rng.standard_normal(RING.N)

    res2 = RING.simulate(T=20000, cue_angles=None, seed=789, init_rates=perturbed)

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Perturbation Recovery (large noise added to clean bump)", fontsize=13)
    gs = GridSpec(2, 5, figure=fig, height_ratios=[2, 1])

    ax_heat = fig.add_subplot(gs[0, :4])
    im = rate_heatmap(ax_heat, res2)
    ax_heat.set_xlabel("Step (after perturbation)")
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label="Rate")

    ax_c = fig.add_subplot(gs[0, 4])
    ax_c.plot(res2.confidence, np.arange(len(res2.confidence)), lw=0.8, color="tab:green")
    ax_c.set_xlabel("Confidence"); ax_c.set_ylabel("Step"); ax_c.invert_yaxis(); ax_c.set_xlim(0, 1)

    for i, t_snap in enumerate([0, 500, 2000, 5000, 15000]):
        ax_p = fig.add_subplot(gs[1, i], projection="polar")
        polar_snapshot(ax_p, res2.rates[t_snap], res2.angles, title=f"t={t_snap}")

    return _save(fig, "05_perturbation_recovery.png")


# -- Figure 6 ---------------------------------------------------------------

def fig_spike_pipeline(seed=42):
    res = RING.simulate(T=7500, cue_angles=[np.pi / 4], cue_duration=2000, seed=seed)
    order = np.argsort(res.angles)

    sp = SpikeProcessor()
    data = sp.process(res.rates, seed=seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Data Pipeline: Rates → Spikes → Binned → Smoothed", fontsize=13)

    panels = [
        (res.rates[:, order].T, "Raw firing rates (100 neurons × 7500 steps)", "inferno"),
        (data.spikes[:, order].T, "Poisson spikes", "gray_r"),
        (data.binned[:, order].T, "Binned spikes (bin=50)", "inferno"),
        (data.smoothed[:, order].T, "Smoothed (window=3)", "inferno"),
    ]
    for ax, (d, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(d, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_ylabel("Neuron")
        ax.set_xlabel("Time bin" if d.shape[1] < 1000 else "Step")
        plt.colorbar(im, ax=ax, shrink=0.8)

    return _save(fig, "06_spike_pipeline.png")


# -- Run all -----------------------------------------------------------------

def run_all():
    print("Generating ring attractor visualisation figures...")
    paths = [f() for f in [
        fig_single_trial, fig_multi_angle, fig_long_term_stability,
        fig_noise_bump, fig_perturbation_recovery, fig_spike_pipeline,
    ]]
    print(f"\nAll {len(paths)} figures saved to {FIG_DIR}/")
    return paths


if __name__ == "__main__":
    run_all()
