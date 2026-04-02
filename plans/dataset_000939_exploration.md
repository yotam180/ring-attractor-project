# Dataset 000939: Large-scale Recordings of Head-Direction Cells in Mouse Postsubiculum

## Overview

**Citation:** Duszkiewicz, Skromne Carrasco, Peyrache (2024). *Large-scale recordings of head-direction cells in mouse postsubiculum.* DANDI:000939/0.240528.1542  
**DOI:** https://doi.org/10.48324/dandi.000939/0.240528.1542  
**Associated paper:** Clark, Abbott, Sompolinsky (2025). *Symmetries and Continuous Attractors in Disordered Neural Circuits.* bioRxiv https://doi.org/10.1101/2025.01.26.634933  
**License:** CC-BY-4.0  
**Format:** Neurodata Without Borders (NWB), HDF5  

---

## What This Dataset Is

This dataset contains **64-channel linear probe (silicon probe) extracellular electrophysiology recordings** from the **postsubiculum** (dorsal presubiculum) of freely-moving mice. The postsubiculum is a cortical region where the majority of excitatory neurons are narrowly tuned to the animal's head direction — these are the canonical **head-direction (HD) cells** of the mammalian navigation system.

**Why HD cells are relevant to this project:** HD cells are the textbook example of a ring attractor circuit. The population maintains a single "bump" of activity that tracks the animal's current heading continuously around 2π. This means:
- The ground-truth neural dynamics should live on a 1D ring manifold
- A correctly trained RNN should recover this ring attractor structure
- The dataset provides real neural activity (not simulated) where we can test whether partial observation causes RNN models to mis-classify the underlying mechanism

---

## Dataset Statistics

| Property | Value |
|---|---|
| Number of subjects | 31 mice |
| Total neurons recorded | 1,533 (pooled across mice) |
| Units per session | 42–185 (mean: 86.8) |
| Head-direction cells per session | 21–117 (mean: 49.5) |
| Species | *Mus musculus* (house mouse) |
| Brain region | Postsubiculum (dorsal presubiculum) |
| Probe type | Cambridge Neurotech H5 (64-channel linear) |
| Sessions with optogenetics | 9 |
| Sessions with triangle epoch | 20 |
| Sessions with square epoch only | 2 |

---

## Session Structure (Epochs)

Every session has **2–4 epochs** in the following order:

```
home_cage (sleep pre)  →  wake_square  →  home_cage (sleep post)  →  [wake_triangle]
```

| Epoch | Duration (example sub-A3701) | Description |
|---|---|---|
| `home_cage` (pre) | ~89 min | Pre-exploration sleep; used for sleep scoring |
| `wake_square` | ~42 min | Open field exploration in square arena |
| `home_cage` (post) | ~63 min | Post-exploration sleep |
| `wake_triangle` | ~40 min | (Optional) Open field in triangular arena — present in 20/31 mice |

Head-direction and position tracking is only available during **wake epochs** (the tracking timestamps span only the wake periods). Sleep scoring (REM/NREM) is applied to home cage epochs.

---

## NWB File Contents (Per Subject)

Each NWB file (`sub-XXXXX_behavior+ecephys[+ogen].nwb`) contains:

### 1. `units/` — Spike-sorted neural data
| Field | Shape (example) | Description |
|---|---|---|
| `id` | (102,) | Unit IDs |
| `spike_times` | (6,043,225,) | All spike timestamps (seconds), concatenated |
| `spike_times_index` | (102,) | Pointer array: unit `i` spans `spike_times[index[i-1]:index[i]]` |
| `is_head_direction` | (102,) uint8 | Boolean: classified as HD cell |
| `is_excitatory` | (102,) uint8 | Boolean: excitatory (broad-waveform) neuron |
| `is_fast_spiking` | (102,) uint8 | Boolean: fast-spiking interneuron |
| `waveform_mean` | (102, 40, 64) float64 | Mean spike waveform: 102 units × 40 samples × 64 channels |
| `trough_to_peak` | (102,) | Waveform trough-to-peak duration (waveform shape metric) |
| `electrode_index` | (102,) | Which electrode channel each unit was detected on |

**Important:** `is_excitatory` and `is_head_direction` are not the same thing. HD cells ≈ excitatory cells in postsubiculum; the paper specifically analyzes HD-classified excitatory units.

### 2. `processing/behavior/CompassDirection/head-direction/`
| Field | Shape | Description |
|---|---|---|
| `data` | (494,315,) float64 | Head direction angle (radians, 0–2π) |
| `timestamps` | (494,315,) float64 | Corresponding timestamps (seconds) |

- Sampling rate: **~100 Hz** (during wake epochs)
- Used to compute tuning curves: for each neuron, bin spikes by head direction to get firing rate as function of θ

### 3. `processing/behavior/Position/position/`
| Field | Shape | Description |
|---|---|---|
| `data` | (494,315, 2) float64 | X, Y position in arena (meters) |
| `timestamps` | (494,315,) float64 | Timestamps (seconds) |

### 4. `processing/ecephys/LFP/LFP/`
| Field | Shape | Description |
|---|---|---|
| `data` | (17,647,830, 64) int16 | Raw LFP across all 64 channels |
| `electrodes` | (64,) | Channel IDs |

- Sampling rate: **~2000 Hz**
- Covers the entire recording (sleep + wake)

### 5. `processing/ecephys/pseudoEMG/`
- Downsampled EMG-proxy signal derived from high-frequency LFP power; used for sleep/wake scoring

### 6. `processing/behavior/accelerometer/`
- 3-axis accelerometer data; high sampling rate (~30 kHz equivalent); useful for motion detection

### 7. `intervals/epochs/` — Session epoch boundaries
- 4 rows: start/stop times and string tags (`home_cage`, `wake_square`, `wake_triangle`)

### 8. `intervals/nrem/` and `intervals/rem/`
- Sleep-scored intervals (NREM: ~85 bouts, REM: ~7 bouts per session typical)
- Used to analyze persistent ring manifold structure during sleep

### 9. `general/extracellular_ephys/electrodes/`
- 64 electrode metadata: x/y/z positions on probe, which channels are faulty (`is_faulty`)

---

## How the Paper Uses This Data (Clark et al. 2025)

Clark et al. use this dataset as the primary real-data test case for their theory of **heterogeneous continuous attractors**:

1. **Tuning curve extraction:** For each neuron, compute occupancy-normalized spike rates in 100 angle bins during `wake_square`. Apply cross-validated Gaussian kernel smoothing (bandwidth selected per neuron by maximizing cross-partition Poisson log-likelihood). Normalize each tuning curve to unit mean.

2. **Key finding:** Head-direction tuning curves show substantial heterogeneity (multi-peaked, variable widths) — violating classical ring-attractor predictions of identical shift-invariant tuning curves.

3. **Network construction (Setting 1):** Using the tuning curves from all 31 mice (N = 1,533 neurons pooled), they solve for a synaptic weight matrix J via least-squares optimization with regularization, such that the tuning curves are fixed points of the dynamics:
   - τ ∂ₜxᵢ(t) = −xᵢ(t) + Σⱼ Jᵢⱼ φⱼ(t) + b
   - Optimization minimizes flow off the tuning-curve manifold
   - τ = 50 ms

4. **Resulting network:** Despite disordered weights, the network exhibits quasi-continuous-attractor dynamics — bump states that drift and can be updated by angular velocity inputs.

---

## Relevance to Our Project

Our project studies how **partial observation** (neuron dropout + time truncation) affects an RNN's ability to recover the **correct dynamical class** (ring attractor vs. discrete fixed points). This dataset is ideal because:

### Why It's the Right Dataset

| Property | Why It Matters |
|---|---|
| **Ground truth is ring attractor** | HD cells are the canonical ring attractor system in mammals. We know what mechanism we should recover. |
| **Real neural heterogeneity** | Unlike teacher simulations, real HD cells have diverse tuning — the paper shows the network still has ring dynamics despite this. This is our Phase 5 (real data validation). |
| **Multiple dropout levels already available** | Sessions range from 21–117 HD cells. We can stratify by session as a natural "neuron count" axis. |
| **Long recordings with sleep** | Multi-hour sessions mean we can test time-window truncation directly. Sleep epochs (where position is absent) provide a natural test of dynamics without behavioral correlate. |
| **Optogenetic sessions** | 9 sessions with optogenetic manipulation — potential perturbation trials analogous to the "update" trials in our dataset plan. |
| **Well-characterized cell types** | `is_head_direction`, `is_excitatory`, `is_fast_spiking` labels allow clean subsetting. |

### How to Apply the Project Pipeline to This Data

**Phase 5 application (from `project_overview.md`):**

```
Step 1: Load NWB file for one session
Step 2: Extract spike trains for all HD cells during wake_square epoch
Step 3: Bin into firing rates (dt=10ms or 20ms bins, smooth)
Step 4: Apply neuron dropout (subsample k of N HD cells)
Step 5: Truncate to time window [0, T]
Step 6: Train student RNN on subsampled, windowed firing rates
Step 7: Analyze student RNN: fixed points, perturbation responses, ring score
Step 8: Compare ring score vs (k/N, T) — the mechanistic recovery curve
```

**Key prediction to test:** There exists a threshold in (k/N, T)-space below which the student RNN's ring score drops sharply even though its predictive fit (MSE on held-out neurons/time) remains good.

### Alignment with `dataset_generation.md` Plan

Our simulated teacher dataset (Phase 3) should mirror the structure of real recordings:

| Simulated (Phase 3) | Real data (Phase 5) |
|---|---|
| Noise-driven ring exploration | home_cage / sleep epochs |
| Single-cue memory trials | wake_square (animal explores, HD bump moves continuously) |
| Perturbation trials | Optogenetic sessions (sub-A1808, A1813, etc.) |
| Controlled neuron fraction | Natural variation across 31 sessions (21–117 HD cells) |

---

## Python Code to Explore This Dataset

### Installation
```bash
pip install h5py numpy matplotlib scipy
# Optional for full NWB API:
pip install pynwb
```

### Load and inspect a session
```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_session(path):
    """Load key data from a NWB file. Returns a dict."""
    data = {}
    with h5py.File(path, 'r') as f:
        # Unit classifications
        data['n_units'] = len(f['units/id'][:])
        data['is_hd'] = f['units/is_head_direction'][:].astype(bool)
        data['is_excitatory'] = f['units/is_excitatory'][:].astype(bool)
        data['is_fast_spiking'] = f['units/is_fast_spiking'][:].astype(bool)
        
        # Spike times (ragged array via index pointer)
        all_spikes = f['units/spike_times'][:]
        spike_idx = f['units/spike_times_index'][:]
        data['spike_times'] = []
        prev = 0
        for idx in spike_idx:
            data['spike_times'].append(all_spikes[prev:idx])
            prev = idx
        
        # Head direction (wake epochs only)
        data['hd_angle'] = f['processing/behavior/CompassDirection/head-direction/data'][:]
        data['hd_times'] = f['processing/behavior/CompassDirection/head-direction/timestamps'][:]
        
        # Position
        data['position'] = f['processing/behavior/Position/position/data'][:]
        data['pos_times'] = f['processing/behavior/Position/position/timestamps'][:]
        
        # Epochs
        data['epoch_starts'] = f['intervals/epochs/start_time'][:]
        data['epoch_stops'] = f['intervals/epochs/stop_time'][:]
        data['epoch_tags'] = [t.decode() for t in f['intervals/epochs/tags'][:]]
        
        # Sleep scoring
        data['nrem_start'] = f['intervals/nrem/start_time'][:]
        data['nrem_stop'] = f['intervals/nrem/stop_time'][:]
        data['rem_start'] = f['intervals/rem/start_time'][:]
        data['rem_stop'] = f['intervals/rem/stop_time'][:]
        
        # Waveforms
        data['waveforms'] = f['units/waveform_mean'][:]  # (n_units, 40 samples, 64 channels)
        
    return data
```

### Extract firing rates in time bins
```python
def bin_spike_trains(spike_times_list, t_start, t_stop, dt=0.02, unit_mask=None):
    """
    Bin spike trains into a firing rate matrix.
    
    Args:
        spike_times_list: list of arrays, one per unit
        t_start, t_stop: time window (seconds)
        dt: bin size in seconds (default 20 ms)
        unit_mask: boolean array; if given, only include True units
    
    Returns:
        rates: (n_units, n_bins) array of spike counts (divide by dt for Hz)
        t_bins: (n_bins,) bin center times
    """
    if unit_mask is not None:
        spike_times_list = [s for s, m in zip(spike_times_list, unit_mask) if m]
    
    bins = np.arange(t_start, t_stop + dt, dt)
    t_bins = (bins[:-1] + bins[1:]) / 2
    n_bins = len(t_bins)
    n_units = len(spike_times_list)
    
    rates = np.zeros((n_units, n_bins))
    for i, spikes in enumerate(spike_times_list):
        mask = (spikes >= t_start) & (spikes < t_stop)
        counts, _ = np.histogram(spikes[mask], bins=bins)
        rates[i] = counts
    
    return rates / dt, t_bins  # Convert to Hz
```

### Compute tuning curves (as in Clark et al.)
```python
def compute_tuning_curves(spike_times_list, hd_angle, hd_times, t_start, t_stop,
                           n_bins=100, unit_mask=None, smooth_sigma_deg=10):
    """
    Compute occupancy-normalized tuning curves.
    
    Returns:
        tuning_curves: (n_units, n_bins) firing rate as function of head direction
        angle_bins: (n_bins,) bin centers in radians
    """
    from scipy.ndimage import gaussian_filter1d
    
    if unit_mask is not None:
        spike_times_list = [s for s, m in zip(spike_times_list, unit_mask) if m]
    
    n_units = len(spike_times_list)
    angle_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    # Time mask for this epoch
    time_mask = (hd_times >= t_start) & (hd_times <= t_stop)
    angles = hd_angle[time_mask]
    times = hd_times[time_mask]
    dt_hd = np.median(np.diff(times))
    
    # Occupancy: time spent at each angle
    occ, _ = np.histogram(angles, bins=angle_bins)
    occ = occ * dt_hd  # Convert to seconds
    occ = np.maximum(occ, 1e-6)  # Avoid division by zero
    
    # Spike counts per angle bin per unit
    tuning_curves = np.zeros((n_units, n_bins))
    for i, spikes in enumerate(spike_times_list):
        mask = (spikes >= t_start) & (spikes <= t_stop)
        # Interpolate head direction at spike times
        spike_angles = np.interp(spikes[mask], times, angles)
        counts, _ = np.histogram(spike_angles, bins=angle_bins)
        tuning_curves[i] = counts / occ  # Firing rate in Hz
    
    # Smooth (Gaussian kernel in angle space; sigma in bins)
    sigma_bins = smooth_sigma_deg / (360 / n_bins)
    tuning_curves = gaussian_filter1d(tuning_curves, sigma=sigma_bins, axis=1, mode='wrap')
    
    return tuning_curves, bin_centers
```

### Visualize tuning curves
```python
def plot_tuning_curves(tuning_curves, angle_bins, title="HD Tuning Curves", 
                       sort_by_peak=True, max_show=50):
    """Plot a heatmap of tuning curves sorted by preferred direction."""
    tc = tuning_curves[:max_show]
    if sort_by_peak:
        order = np.argmax(tc, axis=1).argsort()
        tc = tc[order]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tc, aspect='auto', cmap='hot',
                   extent=[0, 360, tc.shape[0], 0])
    ax.set_xlabel('Head direction (degrees)')
    ax.set_ylabel('Neuron (sorted by pref. dir.)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Firing rate (Hz)')
    return fig
```

### Full exploration pipeline (one session)
```python
import os

SESSION = '000939/sub-A3701/sub-A3701_behavior+ecephys.nwb'

# 1. Load
data = load_session(SESSION)
print(f"HD cells: {data['is_hd'].sum()}")

# 2. Find wake_square epoch
for i, tag in enumerate(data['epoch_tags']):
    if tag == 'wake_square':
        t_start = data['epoch_starts'][i]
        t_stop = data['epoch_stops'][i]
        print(f"wake_square: {t_start:.0f}s - {t_stop:.0f}s ({(t_stop-t_start)/60:.1f} min)")

# 3. Compute tuning curves for HD cells only
tc, angle_bins = compute_tuning_curves(
    data['spike_times'], data['hd_angle'], data['hd_times'],
    t_start, t_stop, unit_mask=data['is_hd']
)
print(f"Tuning curves: {tc.shape}")  # (n_hd_cells, 100)

# 4. Bin firing rates
rates, t_bins = bin_spike_trains(
    data['spike_times'], t_start, t_stop, dt=0.02, unit_mask=data['is_hd']
)
print(f"Rate matrix: {rates.shape}")  # (n_hd_cells, n_time_bins)

# 5. Plot
fig = plot_tuning_curves(tc, angle_bins)
plt.tight_layout()
plt.savefig('tuning_curves_A3701.png', dpi=150)
```

---

## Session Inventory

Sessions are stored in `000939/sub-XXXXX/`. Key session types:

| Group | Subjects | Notes |
|---|---|---|
| Optogenetics (ogen) | A1808, A1813, A1815, A1821, A1824, A6211, A6215, A6216, A9701 | File ends in `+ogen.nwb`; has optogenetic stimulation epoch |
| Standard (square only) | A3705, A5801 | Only square wake epoch |
| Standard (square + triangle) | All others | Two different arena shapes |

Sessions with the **most HD cells** (best for RNN training):
- Verify with: `python3 -c "import h5py; ... is_head_direction.sum() ..."` across all subjects

---

## Key Numbers for Project Planning

| Quantity | Value | Implication |
|---|---|---|
| N_total per session | 42–185 units | Max teacher size if using real data |
| N_HD per session | 21–117 units | Our "observed" population |
| Wake epoch duration | ~40 min (square) | T_max for time-window experiments |
| Sampling rate (spikes → rates) | 50–100 Hz (10–20ms bins) | Governs temporal resolution |
| Tuning curve bins | 100 (standard) | Angular resolution |
| Across-mouse variability | All 31 show consistent mean/SD profile | Supports pooling OR individual session analysis |

---

## Data Quality Notes

- Head-direction data has some `NaN` values (tracking artifacts); these should be masked before computing tuning curves
- `is_faulty` channel flag should be checked before using specific electrode channels for LFP analysis
- The paper uses **cross-validated smoothing** per neuron for tuning curves; a simpler fixed-sigma Gaussian is acceptable for our purposes
- Sleep scoring uses pseudoEMG + theta/delta ratio heuristics; the provided NREM/REM intervals are already scored and can be used directly

---

## Connection to Phase 5 of the Project

From `project_overview.md`:
> *"Real data application – apply steps 2-5 to a real dataset and review the inferred mechanical classification as a function of the applied data filtering aggressiveness."*

This dataset is **the** real dataset for Phase 5. The protocol:

1. For each of the 31 sessions, extract HD cell firing rates during `wake_square`
2. Grid over neuron subsampling fractions: p ∈ {0.2, 0.3, 0.5, 0.7, 1.0}
3. Grid over time windows: T ∈ {5, 10, 20, 40} minutes
4. For each (session, p, T): train student RNN, compute ring score
5. Report ring score as function of (p, T) per session AND pooled

Expected result: confirms (or challenges) the Phase 3 simulated results — the same threshold behavior should appear in real data.

**Alternative use:** Use only a small number of sessions (e.g., the 5 with most HD cells) for Phase 5 given computational constraints, and report results as "consistent across sessions."
