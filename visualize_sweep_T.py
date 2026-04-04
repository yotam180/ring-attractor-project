#!/usr/bin/env python3
"""
Visualize the T-axis sweep results.

Reads from data/sweep_results_T/ and generates figures in figs/04_observation_time/.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("data/sweep_results_T")
FIG_DIR = Path("figs/04_observation_time")

OBS_COLORS = {1.0: "#2176AE", 0.25: "#F26419"}
SEED_MARKERS = {42: "o", 123: "s"}


def load_results():
    with open(RESULTS_DIR / "summary.json") as f:
        return json.load(f)


def fig1_metrics_vs_T(results):
    """Three-panel: uniformity, circularity, drift vs T, with k/N as color."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ("uniformity", "Uniformity", 0.8),
        ("circularity", "Circularity", 0.7),
        ("drift_mean", "Mean |drift| (deg)", 5.0),
    ]

    for ax, (key, label, thresh) in zip(axes, metrics):
        for obs in [1.0, 0.25]:
            for seed in [42, 123]:
                runs = [r for r in results if r["obs_frac"] == obs and r["seed"] == seed]
                runs.sort(key=lambda r: r["T_bin"])
                xs = [r["T_bin"] for r in runs]
                ys = [r[key] for r in runs]
                ax.plot(xs, ys, marker=SEED_MARKERS[seed], color=OBS_COLORS[obs],
                        linewidth=1.5, markersize=7, alpha=0.7)

            # Mean line
            T_vals = sorted(set(r["T_bin"] for r in results))
            means = []
            for T in T_vals:
                vals = [r[key] for r in results if r["obs_frac"] == obs and r["T_bin"] == T]
                means.append(np.mean(vals))
            ax.plot(T_vals, means, "-", color=OBS_COLORS[obs], linewidth=2.5,
                    label=f"k/N={obs:.2f}", zorder=5)

        ax.axhline(thresh, color="red", ls="--", alpha=0.6, linewidth=1.5)
        if key == "drift_mean":
            ax.fill_between([20, 210], thresh, max(15, thresh + 2), color="red", alpha=0.07)
        else:
            ax.fill_between([20, 210], 0, thresh, color="red", alpha=0.07)

        ax.set_xlabel("Trial length (time bins)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlim(15, 215)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ring Attractor Metrics vs Observation Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "T_sweep_metrics.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {FIG_DIR}/T_sweep_metrics.png")
    plt.close(fig)


def fig2_pass_heatmap(results):
    """2D heatmap: T vs k/N pass rates."""
    T_vals = sorted(set(r["T_bin"] for r in results), reverse=True)
    obs_vals = sorted(set(r["obs_frac"] for r in results))

    grid = np.zeros((len(T_vals), len(obs_vals)))
    labels = np.empty((len(T_vals), len(obs_vals)), dtype=object)

    for i, T in enumerate(T_vals):
        for j, obs in enumerate(obs_vals):
            runs = [r for r in results if r["T_bin"] == T and r["obs_frac"] == obs]
            n_pass = sum(1 for r in runs if r["milestone_1_pass"])
            grid[i, j] = n_pass / len(runs) if runs else 0
            labels[i, j] = f"{n_pass}/{len(runs)}"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(obs_vals)))
    ax.set_xticklabels([f"{o:.2f}" for o in obs_vals])
    ax.set_yticks(range(len(T_vals)))
    ax.set_yticklabels([str(T) for T in T_vals])
    ax.set_xlabel("Observation fraction (k/N)", fontsize=11)
    ax.set_ylabel("Trial length (time bins)", fontsize=11)

    for i in range(len(T_vals)):
        for j in range(len(obs_vals)):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if grid[i, j] < 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Pass rate")
    ax.set_title("Milestone 1 Pass Rate: T vs k/N", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "T_sweep_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {FIG_DIR}/T_sweep_heatmap.png")
    plt.close(fig)


def fig3_drift_vs_T(results):
    """Drift progression showing the T threshold clearly."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for obs in [1.0, 0.25]:
        for seed in [42, 123]:
            runs = sorted([r for r in results if r["obs_frac"] == obs and r["seed"] == seed],
                          key=lambda r: r["T_bin"])
            xs = [r["T_bin"] for r in runs]
            ys = [r["drift_mean"] for r in runs]
            ax.plot(xs, ys, marker=SEED_MARKERS[seed], color=OBS_COLORS[obs],
                    linewidth=1, markersize=6, alpha=0.5)

        # Mean
        T_vals = sorted(set(r["T_bin"] for r in results))
        means = [np.mean([r["drift_mean"] for r in results
                          if r["obs_frac"] == obs and r["T_bin"] == T]) for T in T_vals]
        ax.plot(T_vals, means, "-", color=OBS_COLORS[obs], linewidth=2.5,
                label=f"k/N={obs:.2f} (mean)", zorder=5)

    ax.axhline(5.0, color="red", ls="--", alpha=0.6, linewidth=1.5, label="threshold (5°)")
    ax.fill_between([20, 210], 5.0, 20, color="red", alpha=0.07)

    ax.set_xlabel("Trial length (time bins)", fontsize=12)
    ax.set_ylabel("Mean |drift| (deg)", fontsize=12)
    ax.set_title("Generalization Drift vs Observation Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(15, 215)
    ax.set_ylim(0, 16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "T_sweep_drift.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {FIG_DIR}/T_sweep_drift.png")
    plt.close(fig)


def fig4_summary_table(results):
    """Results table."""
    sorted_results = sorted(results, key=lambda x: (-x["T_bin"], -x["obs_frac"], x["seed"]))
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.35 * len(sorted_results)))
    ax.axis("off")

    headers = ["T (bins)", "k/N", "Seed", "Uniformity", "Circularity", "Drift (deg)", "MSE", "Pass?"]
    rows = []
    cell_colors = []
    for r in sorted_results:
        p = r["milestone_1_pass"]
        row = [
            str(r["T_bin"]),
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

    fig.suptitle("T-Axis Sweep Results", fontsize=13, y=0.98)
    fig.savefig(FIG_DIR / "T_sweep_summary_table.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {FIG_DIR}/T_sweep_summary_table.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} results\n")

    fig1_metrics_vs_T(results)
    fig2_pass_heatmap(results)
    fig3_drift_vs_T(results)
    fig4_summary_table(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
