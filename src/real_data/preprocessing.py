"""
Preprocess real HD cell data into the same format as synthetic ring attractor dataset.

Output .npz contains:
    trajectories: (n_trials, T, N_hd)  standardised firing rates
    neuron_angles: (N_hd,)  preferred directions in radians
    mean, std: (N_hd,)  per-neuron normalisation params
    groups: (n_trials,)  all zeros (no A/B distinction for real data)
    trial_hd_angles: (n_trials,)  mean head direction during each trial
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .loading import SessionData


def bin_spikes(
    spike_times_list: list[np.ndarray],
    t_start: float,
    t_stop: float,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin spike trains into firing rate matrix.

    Returns:
        rates: (n_units, n_bins) firing rates in Hz
        t_bins: (n_bins,) bin centre times
    """
    bins = np.arange(t_start, t_stop, dt)
    t_bins = bins[:-1] + dt / 2
    n_units = len(spike_times_list)
    n_bins = len(t_bins)

    rates = np.zeros((n_units, n_bins), dtype=np.float32)
    for i, spikes in enumerate(spike_times_list):
        mask = (spikes >= t_start) & (spikes < t_stop)
        counts, _ = np.histogram(spikes[mask], bins=bins)
        rates[i] = counts / dt

    return rates, t_bins


def compute_preferred_directions(
    spike_times_list: list[np.ndarray],
    hd_angle: np.ndarray,
    hd_times: np.ndarray,
    t_start: float,
    t_stop: float,
    n_angle_bins: int = 100,
    smooth_sigma_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute tuning curves and preferred directions for HD cells.

    Returns:
        pref_dirs: (n_units,) preferred direction in radians
        tuning_curves: (n_units, n_angle_bins) smoothed tuning curves in Hz
    """
    # HD data within epoch
    hd_mask = (hd_times >= t_start) & (hd_times <= t_stop)
    angles = hd_angle[hd_mask]
    times = hd_times[hd_mask]

    # Remove NaNs
    valid = ~np.isnan(angles)
    angles = angles[valid]
    times = times[valid]

    dt_hd = float(np.median(np.diff(times)))

    angle_edges = np.linspace(0, 2 * np.pi, n_angle_bins + 1)
    bin_centres = (angle_edges[:-1] + angle_edges[1:]) / 2

    # Occupancy
    occ, _ = np.histogram(angles, bins=angle_edges)
    occ_sec = occ.astype(np.float64) * dt_hd
    occ_sec = np.maximum(occ_sec, 1e-6)

    n_units = len(spike_times_list)
    tuning_curves = np.zeros((n_units, n_angle_bins))

    for i, spikes in enumerate(spike_times_list):
        mask = (spikes >= t_start) & (spikes <= t_stop)
        spike_angles = np.interp(spikes[mask], times, angles)
        counts, _ = np.histogram(spike_angles, bins=angle_edges)
        tuning_curves[i] = counts / occ_sec

    # Smooth with wrapped Gaussian
    sigma_bins = smooth_sigma_deg / (360.0 / n_angle_bins)
    tuning_curves = gaussian_filter1d(tuning_curves, sigma=sigma_bins, axis=1, mode="wrap")

    # Preferred direction = circular mean weighted by tuning curve
    z = tuning_curves @ np.exp(1j * bin_centres)
    pref_dirs = np.angle(z) % (2 * np.pi)

    return pref_dirs, tuning_curves


def smooth_rates(rates: np.ndarray, kernel_width: int = 3) -> np.ndarray:
    """Causal boxcar smoothing along time axis. rates: (n_units, n_bins)."""
    if kernel_width <= 1:
        return rates
    kernel = np.ones(kernel_width) / kernel_width
    smoothed = np.zeros_like(rates)
    for i in range(rates.shape[0]):
        # Causal: pad on the left
        padded = np.concatenate([np.full(kernel_width - 1, rates[i, 0]), rates[i]])
        smoothed[i] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def segment_trials(
    rates: np.ndarray,
    trial_length: int = 200,
) -> np.ndarray:
    """
    Segment continuous rate matrix into non-overlapping trials.

    Args:
        rates: (N, T_total) firing rates
        trial_length: bins per trial

    Returns:
        trials: (n_trials, trial_length, N) — note transposed to (trial, time, neuron)
    """
    N, T_total = rates.shape
    n_trials = T_total // trial_length
    # Trim to exact multiple
    trimmed = rates[:, :n_trials * trial_length]
    # Reshape: (N, n_trials, trial_length) → (n_trials, trial_length, N)
    reshaped = trimmed.reshape(N, n_trials, trial_length)
    return reshaped.transpose(1, 2, 0).astype(np.float32)


def compute_trial_hd_change(
    hd_angle: np.ndarray,
    hd_times: np.ndarray,
    t_bins: np.ndarray,
    trial_length: int = 200,
) -> np.ndarray:
    """Compute total absolute angular change within each trial (degrees)."""
    # Interpolate HD to rate bin times
    hd_binned = np.interp(t_bins, hd_times, hd_angle)
    n_trials = len(t_bins) // trial_length
    changes = np.zeros(n_trials)
    for i in range(n_trials):
        seg = hd_binned[i * trial_length:(i + 1) * trial_length]
        diffs = np.angle(np.exp(1j * np.diff(seg)))
        changes[i] = np.degrees(np.abs(np.cumsum(diffs)[-1]))
    return changes


def get_trial_hd_angles(
    hd_angle: np.ndarray,
    hd_times: np.ndarray,
    t_bins: np.ndarray,
    trial_length: int = 200,
) -> np.ndarray:
    """Get mean head direction for each trial segment."""
    n_trials = len(t_bins) // trial_length
    trial_angles = np.zeros(n_trials)

    for i in range(n_trials):
        t0 = t_bins[i * trial_length]
        t1 = t_bins[(i + 1) * trial_length - 1]
        mask = (hd_times >= t0) & (hd_times <= t1)
        angles = hd_angle[mask]
        angles = angles[~np.isnan(angles)]
        if len(angles) > 0:
            # Circular mean
            trial_angles[i] = np.angle(np.exp(1j * angles).mean()) % (2 * np.pi)
    return trial_angles


def prepare_dataset(
    session: SessionData,
    dt: float = 0.02,
    trial_length: int = 200,
    smooth_kernel: int = 5,
    max_hd_change_deg: float | None = 45.0,
    output_path: str | Path | None = None,
) -> dict:
    """
    Full preprocessing pipeline: session → .npz dataset.

    Returns dict with all arrays (and saves to output_path if given).
    """
    # Extract HD cell spike trains
    hd_indices = np.where(session.is_hd)[0]
    hd_spike_times = [session.spike_times[i] for i in hd_indices]
    n_hd = len(hd_spike_times)

    print(f"  {session.subject}: {n_hd} HD cells, "
          f"wake_square={session.ws_start:.0f}s–{session.ws_stop:.0f}s "
          f"({(session.ws_stop - session.ws_start)/60:.1f} min)")

    # Bin spikes
    rates, t_bins = bin_spikes(hd_spike_times, session.ws_start, session.ws_stop, dt)
    print(f"  Rate matrix: {rates.shape} (neurons × time bins)")

    # Smooth
    rates = smooth_rates(rates, smooth_kernel)

    # Compute preferred directions
    pref_dirs, tuning_curves = compute_preferred_directions(
        hd_spike_times, session.hd_angle, session.hd_times,
        session.ws_start, session.ws_stop,
    )

    # Sort neurons by preferred direction (like synthetic data)
    order = np.argsort(pref_dirs)
    rates = rates[order]
    pref_dirs = pref_dirs[order]
    tuning_curves = tuning_curves[order]

    # Segment into trials
    trials = segment_trials(rates, trial_length)
    n_trials = trials.shape[0]
    print(f"  Trials: {n_trials} × {trial_length} bins × {n_hd} neurons")

    # Get trial head directions
    trial_hd = get_trial_hd_angles(
        session.hd_angle, session.hd_times, t_bins, trial_length,
    )

    # Filter for stable trials (head direction doesn't change too much)
    if max_hd_change_deg is not None:
        hd_changes = compute_trial_hd_change(
            session.hd_angle, session.hd_times, t_bins, trial_length,
        )
        stable_mask = hd_changes[:n_trials] < max_hd_change_deg
        trials = trials[stable_mask]
        trial_hd = trial_hd[:n_trials][stable_mask]
        n_trials = trials.shape[0]
        print(f"  Stable trials (HD change < {max_hd_change_deg}°): {n_trials}")

    # Standardise (per-neuron z-score)
    mean = trials.mean(axis=(0, 1))    # (N,)
    std = trials.std(axis=(0, 1))      # (N,)
    std = np.maximum(std, 1e-6)
    trajectories = (trials - mean) / std

    print(f"  Standardised: mean≈{trajectories.mean():.4f}, std≈{trajectories.std():.4f}")

    result = {
        "trajectories": trajectories.astype(np.float32),
        "neuron_angles": pref_dirs.astype(np.float64),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "groups": np.zeros(n_trials, dtype=np.int64),
        "trial_hd_angles": trial_hd[:n_trials].astype(np.float64),
        "tuning_curves": tuning_curves.astype(np.float64),
        "subject": session.subject,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), **{
            k: v for k, v in result.items() if isinstance(v, np.ndarray)
        })
        print(f"  Saved: {output_path}")

    return result
