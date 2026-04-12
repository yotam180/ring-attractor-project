"""
Load NWB files from DANDI:000939.

Each file contains spike-sorted units from mouse postsubiculum with
head-direction cell classifications, behavioral tracking, and epoch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class SessionData:
    """All data needed from one NWB session."""
    subject: str
    n_units: int
    n_hd: int

    # Per-unit spike times (list of arrays)
    spike_times: list[np.ndarray]

    # Unit classifications
    is_hd: np.ndarray            # (n_units,) bool
    is_excitatory: np.ndarray    # (n_units,) bool

    # Head direction tracking (wake epochs only)
    hd_angle: np.ndarray         # (n_samples,) radians 0–2π
    hd_times: np.ndarray         # (n_samples,) seconds

    # Epoch boundaries
    epoch_names: list[str]
    epoch_starts: np.ndarray
    epoch_stops: np.ndarray

    # wake_square convenience
    ws_start: float = 0.0
    ws_stop: float = 0.0


def load_session(nwb_path: str | Path) -> SessionData:
    """Load key data from a single NWB file."""
    nwb_path = Path(nwb_path)
    subject = nwb_path.parent.name  # e.g. "sub-A3701"

    with h5py.File(nwb_path, "r") as f:
        # Unit classifications
        is_hd = f["units/is_head_direction"][:].astype(bool)
        is_exc = f["units/is_excitatory"][:].astype(bool)
        n_units = len(is_hd)

        # Spike times (ragged array via index pointer)
        all_spikes = f["units/spike_times"][:]
        spike_idx = f["units/spike_times_index"][:]
        spike_times = []
        prev = 0
        for idx in spike_idx:
            spike_times.append(all_spikes[prev:idx])
            prev = idx

        # Head direction
        hd_angle = f["processing/behavior/CompassDirection/head-direction/data"][:]
        hd_times = f["processing/behavior/CompassDirection/head-direction/timestamps"][:]

        # Epochs
        epoch_starts = f["intervals/epochs/start_time"][:]
        epoch_stops = f["intervals/epochs/stop_time"][:]
        raw_tags = f["intervals/epochs/tags"][:]
        epoch_names = [
            t.decode() if isinstance(t, bytes) else str(t)
            for t in raw_tags
        ]

    # Find wake_square
    ws_start, ws_stop = 0.0, 0.0
    for i, name in enumerate(epoch_names):
        if name == "wake_square":
            ws_start = float(epoch_starts[i])
            ws_stop = float(epoch_stops[i])
            break

    return SessionData(
        subject=subject,
        n_units=n_units,
        n_hd=int(is_hd.sum()),
        spike_times=spike_times,
        is_hd=is_hd,
        is_excitatory=is_exc,
        hd_angle=hd_angle,
        hd_times=hd_times,
        epoch_names=epoch_names,
        epoch_starts=epoch_starts,
        epoch_stops=epoch_stops,
        ws_start=ws_start,
        ws_stop=ws_stop,
    )


def list_sessions(data_dir: str | Path = "000939") -> list[dict]:
    """Scan all sessions and return summary info (subject, n_hd, nwb_path)."""
    data_dir = Path(data_dir)
    sessions = []
    for sub_dir in sorted(data_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        nwb_files = list(sub_dir.glob("*.nwb"))
        if not nwb_files:
            continue
        nwb_path = nwb_files[0]
        # Quick read: just get HD cell count
        with h5py.File(nwb_path, "r") as f:
            is_hd = f["units/is_head_direction"][:].astype(bool)
        sessions.append({
            "subject": sub_dir.name,
            "n_hd": int(is_hd.sum()),
            "n_units": len(is_hd),
            "nwb_path": str(nwb_path),
        })
    return sorted(sessions, key=lambda s: -s["n_hd"])
