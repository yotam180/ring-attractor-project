#!/usr/bin/env python3
"""
Paper-quality figures for the k/N sweep results.

Outputs (new files, does not overwrite existing figures):
  figs/paper_drift_vs_mse.png  — dual-panel: drift and MSE vs k/N
  figs/paper_pca_selected.png  — PCA of converged states at 4 selected k/N values
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = Path("data/sweep_results")
FIG_DIR = Path("figs")


def load_results():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)

    obs_fracs = sorted(set(r["obs_frac"] for r in summary))

    arrays = {}
    for r in summary:
        tag = r["tag"]
        arr_path = RESULTS_DIR / f"{tag}.npz"
        if arr_path.exists():
            arrays[tag] = dict(np.load(str(arr_path)))

    return summary, obs_fracs, arrays


def fig_drift_vs_mse(summary, obs_fracs):
    """Two-panel figure: drift (with cliff) vs MSE (smooth) on the same k/N axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)

    # --- Collect per-condition stats ---
    drift_by_obs = {}
    mse_by_obs = {}
    for r in summary:
        obs = r["obs_frac"]
        drift_by_obs.setdefault(obs, []).append(r["drift_mean"])
        mse_by_obs.setdefault(obs, []).append(r["mse"])

    xs = sorted(drift_by_obs.keys())
    drift_means = [np.mean(drift_by_obs[x]) for x in xs]
    drift_stds = [np.std(drift_by_obs[x]) for x in xs]
    mse_means = [np.mean(mse_by_obs[x]) for x in xs]
    mse_stds = [np.std(mse_by_obs[x]) for x in xs]

    # --- Panel (a): Drift ---
    ax1.errorbar(xs, drift_means, yerr=drift_stds, fmt="o-", color="#2176AE",
                 linewidth=2, markersize=7, capsize=4, capthick=1.5,
                 label="Mean drift", zorder=5)

    # Individual seeds as transparent dots
    for r in summary:
        ax1.plot(r["obs_frac"], r["drift_mean"], "o", color="#2176AE",
                 alpha=0.2, markersize=4, zorder=3)

    ax1.axhline(5.0, color="red", ls="--", alpha=0.6, linewidth=1.5, label="Threshold (5°)")
    ax1.fill_between([-0.02, 1.08], 5.0, max(max(drift_means) + 3, 8),
                     color="red", alpha=0.06)

    ax1.set_ylabel("Mean |drift| (degrees)", fontsize=12)
    ax1.set_xlabel("Observation fraction (k/N)", fontsize=12)
    ax1.set_title("(a)  Drift — mechanistic accuracy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xlim(-0.02, 1.08)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)

    # --- Panel (b): MSE ---
    ax2.errorbar(xs, mse_means, yerr=mse_stds, fmt="s-", color="#F26419",
                 linewidth=2, markersize=7, capsize=4, capthick=1.5,
                 label="Mean MSE", zorder=5)

    for r in summary:
        ax2.plot(r["obs_frac"], r["mse"], "s", color="#F26419",
                 alpha=0.2, markersize=4, zorder=3)

    ax2.set_ylabel("MSE (reconstruction error)", fontsize=12)
    ax2.set_xlabel("Observation fraction (k/N)", fontsize=12)
    ax2.set_title("(b)  MSE — predictive accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim(-0.02, 1.08)
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)

    fig.tight_layout(w_pad=3)
    out = FIG_DIR / "paper_drift_vs_mse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_pca_selected(summary, arrays):
    """PCA panels at 4 selected k/N values, seed 42."""
    selected = [(1.0, 42), (0.25, 42), (0.15, 42), (0.10, 42)]
    # Fallback: if a seed doesn't exist, try 123
    resolved = []
    for obs, seed in selected:
        tag = f"obs{int(obs * 100):03d}_seed{seed}"
        if tag not in arrays:
            tag = f"obs{int(obs * 100):03d}_seed123"
        if tag in arrays:
            resolved.append((obs, seed, tag))

    n = len(resolved)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (obs, seed, tag) in zip(axes, resolved):
        proj = arrays[tag]["pca_proj"]
        theta = arrays[tag]["theta_final"]

        sc = ax.scatter(proj[:, 0], proj[:, 1], c=theta, cmap="hsv",
                        s=12, alpha=0.8, vmin=-np.pi, vmax=np.pi)
        ax.set_aspect("equal")
        ax.set_title(f"k/N = {obs:.2f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15)

        # Find pass/fail for this condition
        match = [r for r in summary if r["tag"] == tag]
        if match:
            r = match[0]
            status = "PASS" if r["milestone_1_pass"] else "FAIL"
            color = "#2e7d32" if r["milestone_1_pass"] else "#c62828"
            ax.text(0.05, 0.95, status, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color=color,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, alpha=0.8))

    fig.tight_layout(w_pad=2)
    out = FIG_DIR / "paper_pca_selected.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_mse_vs_drift_scatter(summary):
    """Scatter plot: MSE vs drift, each dot = one run, colored by pass/fail."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for r in summary:
        color = "#4CAF50" if r["milestone_1_pass"] else "#E53935"
        marker = "o" if r["milestone_1_pass"] else "x"
        size = 40 if r["milestone_1_pass"] else 50
        lw = 1 if r["milestone_1_pass"] else 2
        ax.scatter(r["mse"], r["drift_mean"], c=color, marker=marker,
                   s=size, alpha=0.7, linewidths=lw, zorder=4)

    # Threshold lines
    ax.axhline(5.0, color="red", ls="--", alpha=0.4, linewidth=1.5)
    ax.axvline(0.05, color="orange", ls=":", alpha=0.4, linewidth=1.5)
    ax.text(0.051, 1.0, "MSE = 0.05", fontsize=9, color="orange", alpha=0.7)
    ax.text(0.001, 5.15, "Drift = 5°", fontsize=9, color="red", alpha=0.7)

    # Annotate regions
    ax.text(0.01, 2.0, "Good prediction\nCorrect mechanism",
            fontsize=9, color="#2e7d32", fontstyle="italic", alpha=0.8)
    ax.text(0.09, 2.0, "Poor prediction\nCorrect mechanism",
            fontsize=9, color="#1565C0", fontstyle="italic", alpha=0.8)
    ax.text(0.09, 6.0, "Poor prediction\nWrong mechanism",
            fontsize=9, color="#c62828", fontstyle="italic", alpha=0.8)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="PASS"),
        Line2D([0], [0], marker="x", color="#E53935", markerfacecolor="#E53935",
               markersize=8, markeredgewidth=2, linestyle="None", label="FAIL"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    ax.set_xlabel("MSE (reconstruction error)", fontsize=12)
    ax.set_ylabel("Mean |drift| (degrees)", fontsize=12)
    ax.set_title("Predictive accuracy vs mechanistic correctness\n(each point = one trained model)", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = FIG_DIR / "paper_mse_vs_drift_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_pass_rate_by_kn(summary):
    """Pass rate vs k/N with proper error bars (Wilson score interval)."""
    from collections import defaultdict

    by_obs = defaultdict(list)
    for r in summary:
        by_obs[r["obs_frac"]].append(r["milestone_1_pass"])

    obs_fracs = sorted(by_obs.keys())
    rates = []
    ci_lo = []
    ci_hi = []
    ns = []

    for obs in obs_fracs:
        passes = by_obs[obs]
        n = len(passes)
        k = sum(passes)
        p = k / n
        rates.append(p)
        ns.append(n)
        # Wilson score interval
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        ci_lo.append(centre - margin)
        ci_hi.append(centre + margin)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.errorbar(obs_fracs, rates,
                yerr=[np.array(rates) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(rates)],
                fmt="o-", color="#2176AE", linewidth=2, markersize=8,
                capsize=5, capthick=1.5, zorder=5)

    for x, r, n in zip(obs_fracs, rates, ns):
        k = int(round(r * n))
        ax.annotate(f"{k}/{n}", (x, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Observation fraction (k/N)", fontsize=12)
    ax.set_ylabel("Pass rate", fontsize=12)
    ax.set_title("Ring attractor recovery rate vs observation fraction\n(with 95% Wilson confidence intervals)",
                 fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.02, 1.08)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = FIG_DIR / "paper_pass_rate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, obs_fracs, arrays = load_results()
    print(f"Loaded {len(summary)} results across {len(obs_fracs)} obs fracs\n")

    fig_drift_vs_mse(summary, obs_fracs)
    fig_pca_selected(summary, arrays)
    fig_mse_vs_drift_scatter(summary)
    fig_pass_rate_by_kn(summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
