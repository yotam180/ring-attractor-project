"""
Visualization utilities fro the ring attractor model.
"""

import matplotlib.pyplot as plt
import numpy as np

from ring_attractor import RingAttractor


def plot_ring_state(
    attractor: RingAttractor,
    bottom: float = 10,
    cmap: str = "viridis",
):
    _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    width = 2 * np.pi / attractor.ring_size

    # Normalize the rates to [0, 1]
    norm_rates = attractor.neuron_rates / np.max(attractor.neuron_rates)
    colors = plt.cm.get_cmap(cmap)(norm_rates)

    height = 1
    ax.bar(attractor.neuron_angles, height=height, width=width, bottom=bottom, color=colors, edgecolor="none")
    ax.set_ylim(0, bottom + height)
    ax.set_yticks([])

    return ax
