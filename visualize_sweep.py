#!/usr/bin/env python3
"""
Visualize the partial observation sweep results.

Reads from data/sweep_results/ and generates figures in figs/.

Figures:
  1. sweep_metrics.png     — uniformity, circularity, drift vs k/N
  2. sweep_pca_grid.png    — PCA of converged hidden states per condition
  3. sweep_drift_grid.png  — drift vs angle per condition
  4. sweep_eigenvalues.png — eigenvalue spectra per condition
  5. sweep_summary_table.png — results table
  6. sweep_pass_rates.png  — pass rate bar chart
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("data/sweep_results")
FIG_DIR = Path("figs")

SEED_COLORS = {
    42: "#2176AE", 123: "#F26419", 7: "#4CAF50",
    99: "#9C27B0", 2024: "#FF9800",
}


def load_results():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)

    obs_fracs = sorted(set(r["obs_frac"] for r in summary), reverse=True)
    seeds = sorted(set(r["seed"] for r in summary))

    arrays = {}
    for r in summary:
        tag = r["tag"]
        arr_path = RESULTS_DIR / f"{tag}.npz"
        if arr_path.exists():
            arrays[tag] = dict(np.load(str(arr_path)))

    return summary, obs_fracs, seeds, arrays


def _get_color(seed):
    return SEED_COLORS.get(seed, f"C{seed % 10}")


def fig1_metrics(summary, obs_fracs):
    """Three-panel plot: uniformity, circularity, drift vs k/N with mean + individual points."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ("uniformity", "Uniformity", 0.8),
        ("circularity", "Circularity", 0.7),
        ("drift_mean", "Mean |drift| (deg)", 5.0),
    ]

    for ax, (key, label, thresh) in zip(axes, metrics):
        # Individual points
        for r in summary:
            ax.plot(r["obs_frac"], r[key], "o",
                    color=_get_color(r["seed"]), alpha=0.4, markersize=5)

        # Mean per obs_frac
        for obs in obs_fracs:
            vals = [r[key] for r in summary if r["obs_frac"] == obs]
            mean_val = np.mean(vals)
            ax.plot(obs, mean_val, "D", color="black", markersize=8, zorder=5)

        # Connect means
        mean_xs = sorted(obs_fracs)
        mean_ys = [np.mean([r[key] for r in summary if r["obs_frac"] == obs]) for obs in mean_xs]
        ax.plot(mean_xs, mean_ys, "-", color="black", linewidth=2, zorder=4)

        # Threshold
        ax.axhline(thresh, color="red", ls="--", alpha=0.6, linewidth=1.5)
        if key == "drift_mean":
            ylim = ax.get_ylim()
            ax.fill_between([-0.05, 1.1], thresh, max(ylim[1], thresh + 2),
                            color="red", alpha=0.07)
        else:
            ax.fill_between([-0.05, 1.1], 0, thresh, color="red", alpha=0.07)

        ax.set_xlabel("Observation fraction (k/N)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlim(-0.02, 1.08)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ring Attractor Metrics vs Observation Fraction", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_metrics.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_metrics.png")
    plt.close(fig)


def fig2_pca_grid(summary, obs_fracs, arrays):
    """Grid of PCA scatter plots — one column per obs_frac, using seed 42 (top) and seed 123 (bottom)."""
    display_seeds = [42, 123]
    n_obs = len(obs_fracs)
    n_rows = len(display_seeds)
    fig, axes = plt.subplots(n_rows, n_obs, figsize=(3.2 * n_obs, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j, obs in enumerate(obs_fracs):
        for i, seed in enumerate(display_seeds):
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            ax = axes[i, j]

            if tag in arrays:
                proj = arrays[tag]["pca_proj"]
                theta = arrays[tag]["theta_final"]
                ax.scatter(proj[:, 0], proj[:, 1], c=theta, cmap="hsv",
                           s=4, alpha=0.7, vmin=-np.pi, vmax=np.pi)
                ax.set_aspect("equal")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")

            ax.set_title(f"k/N={obs:.2f}, seed {seed}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == n_rows - 1:
                ax.set_xlabel("PC1", fontsize=8)
            if j == 0:
                ax.set_ylabel("PC2", fontsize=8)

    fig.suptitle("PCA of Converged Hidden States", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_pca_grid.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_pca_grid.png")
    plt.close(fig)


def fig3_drift_grid(summary, obs_fracs, arrays):
    """Grid of drift-vs-angle plots using seed 42 and 123."""
    display_seeds = [42, 123]
    n_obs = len(obs_fracs)
    fig, axes = plt.subplots(1, n_obs, figsize=(3.2 * n_obs, 3.5), sharey=True)

    for j, obs in enumerate(obs_fracs):
        ax = axes[j]
        for seed in display_seeds:
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            if tag in arrays:
                test_deg = np.degrees(arrays[tag]["test_angles"])
                drift = arrays[tag]["drift_deg"]
                ax.plot(test_deg, drift, "o-", color=_get_color(seed),
                        markersize=2, linewidth=0.8, alpha=0.8, label=f"seed {seed}")

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


def fig4_eigenvalues(summary, obs_fracs, arrays):
    """Eigenvalue magnitude distributions per condition (seed 42)."""
    fig, axes = plt.subplots(1, len(obs_fracs), figsize=(3.2 * len(obs_fracs), 3.5))

    for j, obs in enumerate(obs_fracs):
        ax = axes[j]
        for seed in [42, 123]:
            tag = f"obs{int(obs * 100):03d}_seed{seed}"
            if tag in arrays:
                mags = arrays[tag]["eig_mags"]
                mags_sorted = np.sort(mags)[::-1]
                ax.plot(range(len(mags_sorted)), mags_sorted, ".-",
                        markersize=2, linewidth=0.8, alpha=0.8,
                        color=_get_color(seed), label=f"seed {seed}")

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


def fig5_summary_table(summary, obs_fracs):
    """Summary table as a figure."""
    sorted_results = sorted(summary, key=lambda x: (-x["obs_frac"], x["seed"]))
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.35 * len(sorted_results)))
    ax.axis("off")

    headers = ["k/N", "Seed", "Uniformity", "Circularity", "Drift (deg)", "MSE", "Pass?"]
    rows = []
    cell_colors = []
    for r in sorted_results:
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
    table.set_fontsize(9)
    table.scale(1, 1.3)

    fig.suptitle("Partial Observation Sweep — All Results", fontsize=13, y=0.98)
    fig.savefig(FIG_DIR / "sweep_summary_table.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_summary_table.png")
    plt.close(fig)


def fig6_pass_rates(summary, obs_fracs):
    """Bar chart of pass rates per observation fraction."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    xs = []
    rates = []
    n_runs = []
    for obs in obs_fracs:
        runs = [r for r in summary if r["obs_frac"] == obs]
        n_pass = sum(1 for r in runs if r["milestone_1_pass"])
        xs.append(obs)
        rates.append(n_pass / len(runs))
        n_runs.append(len(runs))

    colors = ["#4CAF50" if rate >= 0.8 else "#FF9800" if rate >= 0.5 else "#F44336"
              for rate in rates]

    bars = ax.bar(range(len(xs)), rates, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:.2f}\n(n={n})" for x, n in zip(xs, n_runs)])
    ax.set_xlabel("Observation fraction (k/N)", fontsize=11)
    ax.set_ylabel("Pass rate", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.3)

    for bar, rate, n in zip(bars, rates, n_runs):
        n_pass = int(round(rate * n))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{n_pass}/{n}", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("Milestone 1 Pass Rate by Observation Fraction", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_pass_rates.png", dpi=150, bbox_inches="tight")
    print(f"  Saved figs/sweep_pass_rates.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, obs_fracs, seeds, arrays = load_results()

    print(f"Loaded {len(summary)} results, {len(obs_fracs)} obs fracs, {len(seeds)} seeds\n")

    fig1_metrics(summary, obs_fracs)
    fig2_pca_grid(summary, obs_fracs, arrays)
    fig3_drift_grid(summary, obs_fracs, arrays)
    fig4_eigenvalues(summary, obs_fracs, arrays)
    fig5_summary_table(summary, obs_fracs)
    fig6_pass_rates(summary, obs_fracs)

    print("\nDone.")


if __name__ == "__main__":
    main()
