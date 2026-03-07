"""
Visualization utilities fro the ring attractor model.
"""

import matplotlib.pyplot as plt
import numpy as np

from ring_attractor import RingAttractor


def plot_ring_state(
    attractor: RingAttractor,
    ax=None,
    bottom: float = 10,
    cmap: str = "viridis",
):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    width = 2 * np.pi / attractor.ring_size
    norm_rates = _normalize_rates(attractor.neuron_rates)
    colors = plt.cm.get_cmap(cmap)(norm_rates)

    height = 1
    ax.bar(attractor.neuron_angles, height=height, width=width, bottom=bottom, color=colors, edgecolor="none")
    ax.set_ylim(0, bottom + height)
    ax.set_yticks([])
    ax.set_xticks([])

    return ax


def _normalize_rates(rates: np.ndarray) -> np.ndarray:
    max_rate = np.max(rates)
    min_rate = np.min(rates)

    if max_rate > min_rate:
        return (rates - min_rate) / (max_rate - min_rate)

    return np.zeros_like(rates)
