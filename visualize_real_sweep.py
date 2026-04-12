#!/usr/bin/env python3
"""
Visualize real data sweep results.

Produces:
  1. Metrics vs k/N (per session and averaged)
  2. PCA ring projections grid
  3. Drift vs test angle grid
  4. Comparison with synthetic data thresholds
  5. Pass/fail summary table
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = Path("figs/05_real_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = Path("data/real_sweep_results/summary.json")
SYNTH_SUMMARY = Path("data/sweep_results/summary.json")


def load_results():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def load_synthetic():
    if not SYNTH_SUMMARY.exists():
        return None
    with open(SYNTH_SUMMARY) as f:
        return json.load(f)


def plot_metrics_vs_obs(results, synth_results=None):
    """Three-panel plot: uniformity, circularity, drift vs obs_frac."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    subjects = sorted(set(r.get("subject", "?") for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subject_colors = dict(zip(subjects, colors))

    metrics = [
        ("uniformity", "Uniformity", 0.8),
        ("circularity", "Circularity", 0.7),
        ("drift_mean", "Mean |drift| (°)", 5.0),
    ]

    for ax, (key, label, thresh) in zip(axes, metrics):
        for subj in subjects:
            subj_results = [r for r in results if r.get("subject") == subj]
            obs_fracs = sorted(set(r["obs_frac"] for r in subj_results))
            means = []
            stds = []
            for of in obs_fracs:
                vals = [r[key] for r in subj_results if r["obs_frac"] == of]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
            short_name = subj.replace("sub-", "")
            n_hd = subj_results[0].get("n_hd", "?")
            ax.errorbar(obs_fracs, means, yerr=stds, marker='o', capsize=3,
                       label=f"{short_name} (N={n_hd})",
                       color=subject_colors[subj], linewidth=1.5)

        # Synthetic overlay
        if synth_results is not None:
            s_fracs = sorted(set(r["obs_frac"] for r in synth_results))
            s_means = []
            for of in s_fracs:
                vals = [r[key] for r in synth_results if r["obs_frac"] == of]
                s_means.append(np.mean(vals))
            ax.plot(s_fracs, s_means, 'k--', alpha=0.5, linewidth=2,
                   label="Synthetic (N=100)")

        ax.axhline(thresh, color='red', linestyle=':', alpha=0.5, label=f"Threshold ({thresh})")
        ax.set_xlabel("Observation fraction (k/N)")
        ax.set_ylabel(label)
        if key == "drift_mean":
            ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.05)

    fig.suptitle("Real Data: Ring Score vs Observation Fraction", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "real_sweep_metrics.png", dpi=150)
    print(f"Saved {OUT_DIR / 'real_sweep_metrics.png'}")
    return fig


def plot_pca_grid(results):
    """Grid of PCA ring projections."""
    subjects = sorted(set(r.get("subject", "?") for r in results))
    obs_fracs = sorted(set(r["obs_frac"] for r in results), reverse=True)

    # Use first seed for visualization
    fig, axes = plt.subplots(len(subjects), len(obs_fracs),
                              figsize=(3 * len(obs_fracs), 3 * len(subjects)))
    if len(subjects) == 1:
        axes = axes[np.newaxis, :]

    arr_dir = Path("data/real_sweep_results")

    for i, subj in enumerate(subjects):
        for j, of in enumerate(obs_fracs):
            ax = axes[i, j]
            subj_results = [r for r in results
                           if r.get("subject") == subj and r["obs_frac"] == of]
            if not subj_results:
                ax.set_visible(False)
                continue

            tag = subj_results[0]["tag"]
            arr_path = arr_dir / f"{tag}.npz"
            if not arr_path.exists():
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha='center')
                continue

            d = np.load(arr_path)
            pca = d["pca_proj"]
            theta = d["theta_final"]

            ax.scatter(pca[:, 0], pca[:, 1], c=theta, cmap='hsv', s=3, alpha=0.6)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            r = subj_results[0]
            status = "PASS" if r["milestone_1_pass"] else "FAIL"
            ax.set_title(f"u={r['uniformity']:.2f} c={r['circularity']:.2f}\n"
                        f"d={r['drift_mean']:.1f}° {status}", fontsize=8)

            if j == 0:
                short = subj.replace("sub-", "")
                ax.set_ylabel(f"{short}\n(N={r.get('n_hd', '?')})", fontsize=9)
            if i == 0:
                ax.set_xlabel(f"k/N={of}", fontsize=9)

    fig.suptitle("PCA Projections of Autonomous Fixed Points (Real Data)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "real_sweep_pca_grid.png", dpi=150)
    print(f"Saved {OUT_DIR / 'real_sweep_pca_grid.png'}")
    return fig


def plot_pass_rates(results, synth_results=None):
    """Two-panel pass rate chart: full M1 (left) and ring shape only (right)."""
    subjects = sorted(set(r.get("subject", "?") for r in results))
    obs_fracs = sorted(set(r["obs_frac"] for r in results), reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(obs_fracs))
    width = 0.2
    offsets = np.linspace(-width * (len(subjects) - 1) / 2,
                           width * (len(subjects) - 1) / 2, len(subjects))

    for panel_idx, (ax, pass_fn, title) in enumerate([
        (ax1, lambda r: r["milestone_1_pass"],
         "Full Milestone 1 (u>0.8, c>0.7, drift<5°)"),
        (ax2, lambda r: r["uniformity"] > 0.8 and r["circularity"] > 0.7,
         "Ring Shape Only (u>0.8, c>0.7)"),
    ]):
        for k, subj in enumerate(subjects):
            rates = []
            for of in obs_fracs:
                subset = [r for r in results if r.get("subject") == subj and r["obs_frac"] == of]
                if subset:
                    rates.append(np.mean([pass_fn(r) for r in subset]))
                else:
                    rates.append(0)
            short = subj.replace("sub-", "")
            n_hd = [r for r in results if r.get("subject") == subj][0].get("n_hd", "?")
            ax.bar(x + offsets[k], rates, width, label=f"{short} (N={n_hd})", alpha=0.8)

        # Synthetic overlay (use full M1 for both panels since synthetic drift is always <5°)
        if synth_results is not None:
            s_fracs_set = sorted(set(r["obs_frac"] for r in synth_results), reverse=True)
            s_rates = []
            for of in obs_fracs:
                if of in s_fracs_set:
                    subset = [r for r in synth_results if r["obs_frac"] == of]
                    s_rates.append(np.mean([r["milestone_1_pass"] for r in subset]))
                else:
                    s_rates.append(np.nan)
            ax.plot(x, s_rates, 'k--o', label="Synthetic (N=100)", linewidth=2, markersize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{of:.2f}" for of in obs_fracs])
        ax.set_xlabel("Observation fraction (k/N)")
        ax.set_ylabel("Pass rate")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9)
        ax.set_title(title)

    fig.suptitle("Pass Rate: Real vs Synthetic", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "real_sweep_pass_rates.png", dpi=150)
    print(f"Saved {OUT_DIR / 'real_sweep_pass_rates.png'}")
    return fig


def plot_drift_grid(results):
    """Drift vs test angle for each condition."""
    subjects = sorted(set(r.get("subject", "?") for r in results))
    obs_fracs = sorted(set(r["obs_frac"] for r in results), reverse=True)

    fig, axes = plt.subplots(len(subjects), len(obs_fracs),
                              figsize=(3 * len(obs_fracs), 2.5 * len(subjects)))
    if len(subjects) == 1:
        axes = axes[np.newaxis, :]

    arr_dir = Path("data/real_sweep_results")

    for i, subj in enumerate(subjects):
        for j, of in enumerate(obs_fracs):
            ax = axes[i, j]
            subj_results = [r for r in results
                           if r.get("subject") == subj and r["obs_frac"] == of]
            if not subj_results:
                ax.set_visible(False)
                continue

            tag = subj_results[0]["tag"]
            arr_path = arr_dir / f"{tag}.npz"
            if not arr_path.exists():
                continue

            d = np.load(arr_path)
            ax.plot(np.degrees(d["test_angles"]), d["drift_deg"], 'b.-', markersize=3)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(5, color='red', linestyle=':', alpha=0.3)
            ax.axhline(-5, color='red', linestyle=':', alpha=0.3)
            ax.set_ylim(-30, 30)

            if i == len(subjects) - 1:
                ax.set_xlabel("Test angle (°)")
            if j == 0:
                short = subj.replace("sub-", "")
                ax.set_ylabel(f"{short}\nDrift (°)")
            if i == 0:
                ax.set_title(f"k/N={of}")

    fig.suptitle("Generalization Drift (Real Data)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "real_sweep_drift_grid.png", dpi=150)
    print(f"Saved {OUT_DIR / 'real_sweep_drift_grid.png'}")
    return fig


def print_summary(results):
    """Print formatted summary table."""
    print(f"\n{'subject':<10} {'obs':>5} {'seed':>5} {'N':>4} {'unif':>6} {'circ':>6} "
          f"{'drift':>7} {'mse':>8} {'pass':>5}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: (x.get("subject", ""), -x["obs_frac"], x["seed"])):
        s = r.get("subject", "?").replace("sub-", "")
        print(f"{s:<10} {r['obs_frac']:5.2f} {r['seed']:5d} {r.get('n_hd', 0):4d} "
              f"{r['uniformity']:6.3f} {r['circularity']:6.3f} {r['drift_mean']:6.2f}° "
              f"{r['mse']:8.4f} {'YES' if r['milestone_1_pass'] else 'NO':>5}")


def main():
    results = load_results()
    synth = load_synthetic()

    print(f"Loaded {len(results)} real data results")
    if synth:
        print(f"Loaded {len(synth)} synthetic results for comparison")

    print_summary(results)

    plot_metrics_vs_obs(results, synth)
    plot_pca_grid(results)
    plot_drift_grid(results)
    plot_pass_rates(results, synth)

    plt.close("all")
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
