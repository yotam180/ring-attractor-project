#!/usr/bin/env python3
"""
Visualize the partial observation sweep results.

Reads from data/sweep_results/ and generates figures in figs/.

Figures:
  1. sweep_metrics.png     — uniformity, circularity, drift vs k/N
  2. sweep_pca_grid.png    — PCA of converged hidden states per condition
  3. sweep_drift_grid.png  — drift vs angle per condition
  4. sweep_eigenvalues.png — eigenvalue spectra per condition
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = Path("data/sweep_results")
FIG_DIR = Path("figs")


def load_results():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)

    # Group by obs_frac
    obs_fracs = sorted(set(r["obs_frac"] for r in summary), reverse=True)
    seeds = sorted(set(r["seed"] for r in summary))

    arrays = {}
    for r in summary:
        tag = r["tag"]
        arr_path = RESULTS_DIR / f"{tag}.npz"
        if arr_path.exists():
            arrays[tag] = dict(np.load(str(arr_path)))

    return summary, obs_fracs, seeds, arrays


def fig1_metrics(summary, obs_fracs, seeds):
    """Three-panel plot: uniformity, circularity, drift vs observation fraction."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ("uniformity", "Uniformity", 0.8, "higher is better"),
        ("circularity", "Circularity", 0.7, "higher is better"),
        ("drift_mean", "Mean |drift| (deg)", 5.0, "lower is better"),
    ]

    colors = {42: "#2176AE", 123: "#F26419"}

    for ax, (key, label, thresh, note) in zip(axes, metrics):
        for seed in seeds:
            xs = []
            ys = []
            for r in summary:
                if r["seed"] == seed:
                    xs.append(r["obs_frac"])
                    ys.append(r[key])
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, "o-", color=colors[seed], label=f"seed {seed}",
                    markersize=7, linewidth=2)

        # Threshold line
        ax.axhline(thresh, color="red", ls="--", alpha=0.6, linewidth=1.5)
        if key == "drift_mean":
            ax.fill_between([0, 1.1], thresh, ax.get_ylim()[1] if ax.get_ylim()[1] > thresh else thresh + 5,
                            color="red", alpha=0.07)
            ax.text(0.05, thresh + 0.3, f"threshold = {thresh}°", color="red",
                    fontsize=9, alpha=0.8)
        else:
            ax.fill_between([0, 1.1], 0, thresh, color="red", alpha=0.07)
            ax.text(0.05, thresh + 0.02, f"threshold = {thresh}", color="red",
                    fontsize=9, alpha=0.8)

        ax.set_xlabel("Observation fraction (k/N)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlim(-0.02, 1.08)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ring Attractor Metrics vs Observation Fraction", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_metrics.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_metrics.png")
    plt.close(fig)


def fig2_pca_grid(summary, obs_fracs, seeds, arrays):
    """Grid of PCA scatter plots — one column per obs_frac, one row per seed."""
    n_obs = len(obs_fracs)
    n_seeds = len(seeds)
    fig, axes = plt.subplots(n_seeds, n_obs, figsize=(3.2 * n_obs, 3.2 * n_seeds))
    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    for j, obs in enumerate(obs_fracs):
        for i, seed in enumerate(seeds):
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            ax = axes[i, j]

            if tag in arrays:
                proj = arrays[tag]["pca_proj"]
                theta = arrays[tag]["theta_final"]
                sc = ax.scatter(proj[:, 0], proj[:, 1], c=theta, cmap="hsv",
                                s=4, alpha=0.7, vmin=-np.pi, vmax=np.pi)
                ax.set_aspect("equal")

            ax.set_title(f"k/N={obs:.2f}" + (f", seed {seed}" if i == 0 or True else ""),
                         fontsize=9)
            ax.tick_params(labelsize=7)
            if i == n_seeds - 1:
                ax.set_xlabel("PC1", fontsize=8)
            if j == 0:
                ax.set_ylabel("PC2", fontsize=8)

    fig.suptitle("PCA of Converged Hidden States", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_pca_grid.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_pca_grid.png")
    plt.close(fig)


def fig3_drift_grid(summary, obs_fracs, seeds, arrays):
    """Grid of drift-vs-angle plots."""
    n_obs = len(obs_fracs)
    fig, axes = plt.subplots(1, n_obs, figsize=(3.2 * n_obs, 3.5), sharey=True)

    colors = {42: "#2176AE", 123: "#F26419"}

    for j, obs in enumerate(obs_fracs):
        ax = axes[j]
        for seed in seeds:
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            if tag in arrays:
                test_deg = np.degrees(arrays[tag]["test_angles"])
                drift = arrays[tag]["drift_deg"]
                ax.plot(test_deg, drift, "o-", color=colors[seed],
                        markersize=3, linewidth=1, alpha=0.8, label=f"seed {seed}")

        ax.axhline(0, color="k", ls="--", alpha=0.3)
        ax.axhline(5, color="red", ls=":", alpha=0.4)
        ax.axhline(-5, color="red", ls=":", alpha=0.4)
        ax.set_xlabel("Test angle (deg)", fontsize=9)
        ax.set_title(f"k/N = {obs:.2f}", fontsize=10)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("Drift (deg)", fontsize=9)
            ax.legend(fontsize=7)
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Generalization Drift vs Test Angle", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_drift_grid.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_drift_grid.png")
    plt.close(fig)


def fig4_eigenvalues(summary, obs_fracs, seeds, arrays):
    """Eigenvalue magnitude distributions per condition."""
    fig, axes = plt.subplots(1, len(obs_fracs), figsize=(3.2 * len(obs_fracs), 3.5))

    for j, obs in enumerate(obs_fracs):
        ax = axes[j]
        for seed in seeds:
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            if tag in arrays:
                mags = arrays[tag]["eig_mags"]
                mags_sorted = np.sort(mags)[::-1]
                ax.plot(range(len(mags_sorted)), mags_sorted, ".-",
                        markersize=2, linewidth=0.8, alpha=0.8,
                        label=f"seed {seed}")

        ax.axhline(1.0, color="red", ls="--", alpha=0.5, linewidth=1)
        ax.set_title(f"k/N = {obs:.2f}", fontsize=10)
        ax.set_xlabel("Eigenvalue index", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("|lambda|", fontsize=9)
            ax.legend(fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Jacobian Eigenvalue Magnitudes (sorted)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_eigenvalues.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_eigenvalues.png")
    plt.close(fig)


def fig5_summary_table(summary, obs_fracs, seeds):
    """Summary table as a figure for easy inclusion in reports."""
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(summary)))
    ax.axis("off")

    headers = ["k/N", "Seed", "Uniformity", "Circularity", "Drift (deg)", "MSE", "Pass?"]
    rows = []
    cell_colors = []
    for r in sorted(summary, key=lambda x: (-x["obs_frac"], x["seed"])):
        p = r["milestone_1_pass"]
        row = [
            f"{r['obs_frac']:.2f}",
            str(r["seed"]),
            f"{r['uniformity']:.3f}",
            f"{r['circularity']:.3f}",
            f"{r['drift_mean']:.2f}",
            f"{r['mse']:.4f}",
            "PASS" if p else "FAIL",
        ]
        rows.append(row)
        bg = "#d4edda" if p else "#f8d7da"
        cell_colors.append([bg] * len(headers))

    table = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors,
                     colColours=["#e9ecef"] * len(headers),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    fig.suptitle("Partial Observation Sweep Results", fontsize=13, y=0.98)
    fig.savefig(FIG_DIR / "sweep_summary_table.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_summary_table.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, obs_fracs, seeds, arrays = load_results()

    print(f"Loaded {len(summary)} results, {len(obs_fracs)} obs fracs, {len(seeds)} seeds\n")

    fig1_metrics(summary, obs_fracs, seeds)
    fig2_pca_grid(summary, obs_fracs, seeds, arrays)
    fig3_drift_grid(summary, obs_fracs, seeds, arrays)
    fig4_eigenvalues(summary, obs_fracs, seeds, arrays)
    fig5_summary_table(summary, obs_fracs, seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()
