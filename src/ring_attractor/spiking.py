"""
Spike generation and processing pipeline.

Converts continuous firing rates into Poisson spikes, bins them
temporally, and applies causal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d


@dataclass
class SpikeData:
    """Container for processed spike data."""

    spikes: np.ndarray    # (T, N)       raw Poisson spike counts
    binned: np.ndarray    # (T_bin, N)   temporally binned counts
    smoothed: np.ndarray  # (T_bin, N)   causal-smoothed counts


class SpikeProcessor:
    """
    Pipeline: rates → Poisson spikes → temporal binning → causal smoothing.

    Parameters
    ----------
    dt : float
        Integration timestep (for Poisson rate calculation).
    rate_scale : float
        Multiplier converting firing rate to Poisson lambda.
    bin_factor : int
        Number of integration steps per time bin.
    smoothing_window : int
        Width of the causal boxcar kernel (in bins).
    """

    def __init__(
        self,
        dt: float = 0.01,
        rate_scale: float = 100.0,
        bin_factor: int = 50,
        smoothing_window: int = 3,
    ):
        self.dt = dt
        self.rate_scale = rate_scale
        self.bin_factor = bin_factor
        self.smoothing_window = smoothing_window

    def process(self, rates: np.ndarray, seed: int | None = None) -> SpikeData:
        """Run the full pipeline on a (T, N) rate matrix."""
        spikes = self.generate_spikes(rates, seed)
        binned = self.bin_spikes(spikes)
        smoothed = self.smooth_bins(binned)
        return SpikeData(spikes=spikes, binned=binned, smoothed=smoothed)

    def generate_spikes(
        self, rates: np.ndarray, seed: int | None = None
    ) -> np.ndarray:
        """Poisson spike counts from a (T, N) rate matrix."""
        rng = np.random.default_rng(seed)
        lam = np.clip(rates * self.rate_scale * self.dt, 0, None)
        return rng.poisson(lam).astype(np.int32)

    def bin_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Sum every *bin_factor* rows.  (T, N) → (T // bf, N)."""
        T, N = spikes.shape
        T_bin = T // self.bin_factor
        trimmed = spikes[: T_bin * self.bin_factor]
        return trimmed.reshape(T_bin, self.bin_factor, N).sum(axis=1)

    def smooth_bins(self, bins: np.ndarray) -> np.ndarray:
        """Causal boxcar smoothing along the time axis."""
        w = self.smoothing_window
        kernel = np.ones(w) / w
        smoothed = convolve1d(
            bins.astype(np.float64),
            kernel,
            axis=0,
            mode="constant",
            cval=0.0,
            origin=-(w // 2),
        )
        # Avoid ramp-up artefact at the start.
        smoothed[:w] = bins[0]
        return smoothed
