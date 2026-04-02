"""
Reusable plotting helpers for ring attractor visualisation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .network import SimResult


def polar_snapshot(
    ax: plt.Axes,
    rates: np.ndarray,
    angles: np.ndarray,
    title: str = "",
) -> None:
    """Draw a polar bar chart of neuron firing rates."""
    width = 2 * np.pi / len(angles)
    r_norm = rates / (rates.max() + 1e-12)
    colors = plt.cm.inferno(r_norm)
    ax.bar(angles, rates, width=width, bottom=0, color=colors, edgecolor="none")
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
    ax.set_title(title, fontsize=9, pad=8)


def rate_heatmap(
    ax: plt.Axes,
    result: SimResult,
    *,
    cmap: str = "inferno",
    cue_off_step: int | None = None,
) -> plt.cm.ScalarMappable:
    """
    Plot a neuron × time heatmap of firing rates.

    Neurons are sorted by preferred angle.  Returns the image for
    colorbar attachment.
    """
    order = np.argsort(result.angles)
    im = ax.imshow(
        result.rates[:, order].T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Integration step")
    ax.set_ylabel("Neuron (sorted by pref. angle)")
    if cue_off_step is not None:
        ax.axvline(cue_off_step, color="cyan", ls="--", lw=1, label="cue off")
        ax.legend(fontsize=8, loc="upper right")
    return im


def theta_confidence_plot(
    axes: tuple[plt.Axes, plt.Axes],
    result: SimResult,
    target: float | None = None,
    cue_off_step: int | None = None,
) -> None:
    """Plot decoded θ and confidence on two vertically-stacked axes."""
    ax_theta, ax_conf = axes

    ax_theta.plot(np.degrees(result.theta), lw=0.5, color="tab:blue", alpha=0.8)
    if target is not None:
        ax_theta.axhline(
            np.degrees(target), color="red", ls="--", lw=1,
            label=f"target={np.degrees(target):.0f}°",
        )
        ax_theta.legend(fontsize=8)
    ax_theta.set_ylabel("Decoded θ (°)")
    ax_theta.set_ylim(-200, 200)

    ax_conf.plot(result.confidence, lw=0.8, color="tab:green")
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_xlabel("Integration step")
    ax_conf.set_ylim(0, 1)

    if cue_off_step is not None:
        for ax in (ax_theta, ax_conf):
            ax.axvline(cue_off_step, color="cyan", ls="--", lw=0.8)


def circ_error(theta_hat: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Signed circular distance in degrees."""
    d = theta_hat - theta_true
    return np.degrees(np.arctan2(np.sin(d), np.cos(d)))
