"""
PyTorch Dataset for ring attractor training data.

Handles per-trial teacher-forcing cutoff randomisation: each __getitem__
call draws a fresh K ~ Uniform(k_min, k_max).  The returned input tensor
contains observed rates for t < K and zeros for t >= K.  A binary mask
marks the autonomous phase (where loss is computed).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RingAttractorDataset(Dataset):
    """
    Returns (input, target, mask) per trial.

        input:  (T, input_dim) — observed neurons for t < K, zeros after
        target: (T, N)         — full standardised trajectory
        mask:   (T,)           — 0 during teacher-forcing, 1 during autonomous
    """

    def __init__(
        self,
        trajectories: np.ndarray,   # (n_trials, T, N) standardised
        k_min: int = 5,
        k_max: int = 20,
        observed_idx: np.ndarray | None = None,
    ):
        self.trajectories = torch.from_numpy(
            trajectories.astype(np.float32)
        )
        self.k_min = k_min
        self.k_max = k_max

        N = trajectories.shape[-1]
        if observed_idx is not None:
            self.observed_idx = torch.from_numpy(observed_idx.astype(np.int64))
        else:
            self.observed_idx = torch.arange(N)
        self.input_dim = len(self.observed_idx)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = self.trajectories[idx]               # (T, N)
        T = traj.shape[0]

        K = torch.randint(self.k_min, self.k_max + 1, (1,)).item()

        # Input: observed neurons for first K steps, zeros after
        x = torch.zeros(T, self.input_dim)
        x[:K] = traj[:K][:, self.observed_idx]

        # Mask: 1 = autonomous (compute loss), 0 = teacher-forced (no loss)
        mask = torch.zeros(T)
        mask[K:] = 1.0

        return x, traj, mask
