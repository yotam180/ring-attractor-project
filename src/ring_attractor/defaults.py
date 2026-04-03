"""
Shared default parameters for the ring attractor simulator and spike pipeline.

This module is the **single source of truth** for all numerical defaults.
Both the simulator classes and the dataset generation / evaluation scripts
import from here, ensuring that ``RingAttractor()`` with no arguments
always matches the parameters used to generate training data.
"""

import numpy as np

# ── Simulator ────────────────────────────────────────────────────────────
N: int = 100                        # neurons on the ring
J0: float = -2.0                    # inhibition baseline  (cosine kernel)
J1: float = 4.0                     # excitation amplitude  (cosine kernel)
STEEPNESS: float = 4.0              # slope of φ(x) = tanh(s · ReLU(x))
ALPHA: float = 0.01                 # leak rate  (= dt / τ)
SIGMA: float = 0.1                  # additive Gaussian noise amplitude

# ── Cue ──────────────────────────────────────────────────────────────────
CUE_AMPLITUDE: float = 3.0          # peak of the Gaussian cue
CUE_SIGMA: float = np.radians(20)   # width (std, rad) of cue envelope

# ── Spike processing ─────────────────────────────────────────────────────
DT: float = 0.01                    # integration timestep
RATE_SCALE: float = 100.0           # Poisson λ multiplier
BIN_FACTOR: int = 50                # integration steps per time bin
SMOOTHING_WINDOW: int = 3           # causal boxcar kernel width (bins)

# ── Dataset generation ───────────────────────────────────────────────────
N_ANGLES: int = 72                  # 5-degree spacing
T_CUE: int = 2000                   # integration steps with external cue
T_SETTLE: int = 500                 # post-cue settling
T_RECORD: int = 10_000              # integration steps recorded per trial
SIGMA_PERTURB_FRAC: float = 0.5     # perturbation σ as fraction of peak rate
