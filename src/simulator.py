"""
Ring attractor simulator.

Implements rate dynamics on a ring of N neurons with cosine connectivity,
integrated via forward Euler.

Dynamics (discrete time):
    h[t+1] = h[t] + α(-h[t] + φ(W h[t] + I_ext[t] + σ ξ[t]))

Nonlinearity:
    φ(x) = tanh(steepness × max(0, x))

    This combines a hard threshold at 0 (silent below) with tanh saturation
    (bounded above).  The steepness parameter controls the f-I curve slope:
    higher steepness → sharper, more concentrated bumps.

Firing rates (non-negative by construction since φ ≥ 0):
    r[t] = φ(...)  (already non-negative)

Weight matrix (cosine kernel, normalised by N):
    W_ij = (J0 + J1 cos(θ_i - θ_j)) / N

Design notes:
  - Plain ReLU has no stable nonzero equilibrium (see plans/relu_instability_proof.md).
  - Plain tanh works but produces very wide bumps (FWHM ~150°) because the peak
    input is deep in the saturation regime of tanh.
  - The threshold-saturating φ(x) = tanh(s × ReLU(x)) gives sharp bumps
    (FWHM ~90°) because silent neurons stay at exactly 0, and the steep
    transition concentrates activity near the peak.
"""

import numpy as np
from scipy.ndimage import convolve1d


# ---------------------------------------------------------------------------
# Weight matrix
# ---------------------------------------------------------------------------

def make_weights(N: int, J0: float = -2.0, J1: float = 4.0) -> np.ndarray:
    """Cosine-kernel connectivity, shape (N, N), divided by N."""
    angles = 2 * np.pi * np.arange(N) / N
    dtheta = angles[:, None] - angles[None, :]
    return (J0 + J1 * np.cos(dtheta)) / N


# ---------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------

def phi(x: np.ndarray, steepness: float = 4.0) -> np.ndarray:
    """Threshold-saturating activation: tanh(steepness × ReLU(x))."""
    return np.tanh(steepness * np.maximum(x, 0.0))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(
    N: int = 100,
    J0: float = -2.0,
    J1: float = 4.0,
    alpha: float = 0.01,
    sigma: float = 0.1,
    steepness: float = 4.0,
    T: int = 5000,
    cue_angles: list[float] | None = None,
    cue_onset: int = 0,
    cue_duration: int = 2000,
    cue_amplitude: float = 3.0,
    init_rates: np.ndarray | None = None,
    init_noise_scale: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Run the ring attractor for T integration steps.

    Parameters
    ----------
    N : int
        Number of neurons on the ring.
    J0, J1 : float
        Inhibition baseline and excitation amplitude for cosine kernel.
    alpha : float
        Leak rate  (= dt / tau in continuous-time language).  Controls how
        fast the dynamics evolve per step.  Default 0.01 gives full bump
        formation within ~2000 cue steps.
    sigma : float
        Additive Gaussian noise amplitude.
    steepness : float
        Slope of the threshold-saturating nonlinearity φ(x) = tanh(s·ReLU(x)).
        Higher → sharper bump but wider FWHM.
        steepness=4 gives FWHM ≈ 90°, confidence ≈ 0.90.
    T : int
        Number of integration steps.
    cue_angles : list of float, optional
        Angles (radians) at which to inject a cosine cue.
    cue_onset : int
        Step at which the cue starts.
    cue_duration : int
        Number of steps the cue is active.
    cue_amplitude : float
        Peak amplitude of the cosine cue.
    init_rates : ndarray (N,), optional
        Initial hidden state.  If None, small Gaussian noise.
    init_noise_scale : float
        Std of initial noise (used when init_rates is None).
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys
        rates      : (T, N)   firing rates  (non-negative, ∈ [0, 1])
        angles     : (N,)     preferred angle of each neuron
        theta      : (T,)     decoded bump angle per step
        confidence : (T,)     decoding confidence per step  (0 = flat, ~0.90 = good bump)
        weights    : (N, N)   connectivity matrix
    """
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(N) / N
    W = make_weights(N, J0, J1)

    # Initial state
    if init_rates is not None:
        h = np.array(init_rates, dtype=np.float64)
    else:
        h = init_noise_scale * rng.standard_normal(N)

    # Storage
    rates = np.empty((T, N), dtype=np.float64)
    theta = np.empty(T, dtype=np.float64)
    confidence = np.empty(T, dtype=np.float64)

    for t in range(T):
        # Compute firing rates via nonlinearity
        r = phi(h, steepness)
        rates[t] = r
        th, conf = _decode_theta(r, angles)
        theta[t] = th
        confidence[t] = conf

        # External cue input
        I_ext = np.zeros(N, dtype=np.float64)
        if cue_angles is not None and cue_onset <= t < cue_onset + cue_duration:
            for ca in cue_angles:
                I_ext += cue_amplitude * np.cos(angles - ca)

        # Euler step: hidden state update
        noise = sigma * rng.standard_normal(N)
        total_input = W @ h + I_ext + noise
        dh = (-h + phi(total_input, steepness)) * alpha
        h = h + dh

    return dict(
        rates=rates,
        angles=angles,
        theta=theta,
        confidence=confidence,
        weights=W,
    )


# ---------------------------------------------------------------------------
# Theta decoding  (population vector)
# ---------------------------------------------------------------------------

def _decode_theta(rates: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    """Return (angle, confidence) using the population vector method."""
    z = np.sum(rates * np.exp(1j * angles))
    total = rates.sum() + 1e-12
    return float(np.angle(z)), float(np.abs(z) / total)


def decode_theta(rates: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised decoding over a (T, N) rate matrix."""
    z = rates @ np.exp(1j * angles)
    total = rates.sum(axis=1) + 1e-12
    return np.angle(z), np.abs(z) / total


# ---------------------------------------------------------------------------
# Spike generation + processing
# ---------------------------------------------------------------------------

def generate_spikes(
    rates: np.ndarray,
    dt: float = 0.01,
    rate_scale: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Poisson spikes from rate matrix (T, N).  Returns int spike counts."""
    rng = np.random.default_rng(seed)
    lam = np.clip(rates * rate_scale * dt, 0, None)
    return rng.poisson(lam).astype(np.int32)


def bin_spikes(spikes: np.ndarray, bin_factor: int = 50) -> np.ndarray:
    """Temporal binning: sum every `bin_factor` rows.  (T, N) → (T//bf, N)."""
    T, N = spikes.shape
    T_bin = T // bin_factor
    return spikes[: T_bin * bin_factor].reshape(T_bin, bin_factor, N).sum(axis=1)


def smooth_bins(bins: np.ndarray, window: int = 3) -> np.ndarray:
    """Causal boxcar smoothing along the time axis."""
    kernel = np.ones(window) / window
    smoothed = convolve1d(
        bins.astype(np.float64), kernel, axis=0,
        mode="constant", cval=0.0, origin=-(window // 2),
    )
    smoothed[:window] = bins[0]
    return smoothed


# ---------------------------------------------------------------------------
# Convenience: full pipeline  (simulate → spike → bin → smooth)
# ---------------------------------------------------------------------------

def simulate_trial(
    theta_target: float,
    T_cue: int = 2000,
    T_total: int = 7500,
    bin_factor: int = 50,
    smoothing_window: int = 3,
    dt: float = 0.01,
    rate_scale: float = 100.0,
    seed: int | None = None,
    **sim_kwargs,
) -> dict:
    """
    Run one trial: cue at theta_target for T_cue steps, then free-run.

    Returns dict with raw results plus:
        spikes_binned  : (T_bin, N) binned spike counts
        rates_smooth   : (T_bin, N) smoothed binned rates
    """
    res = simulate(
        T=T_total,
        cue_angles=[theta_target],
        cue_onset=0,
        cue_duration=T_cue,
        seed=seed,
        **sim_kwargs,
    )
    spikes = generate_spikes(res["rates"], dt=dt, rate_scale=rate_scale, seed=seed)
    binned = bin_spikes(spikes, bin_factor)
    smoothed = smooth_bins(binned, smoothing_window)
    res["spikes_binned"] = binned
    res["rates_smooth"] = smoothed
    return res
