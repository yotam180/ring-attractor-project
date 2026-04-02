"""
RNN architectures for ring attractor recovery.

Two variants:
  - VanillaRateRNN: full recurrent weight matrix W_hh
  - LowRankRateRNN: W_hh = UV^T / H  (rank <= R)

Both support a teacher-forcing -> autonomous transition: the input tensor
contains observed rates for t < K and zeros for t >= K.  The model processes
the full sequence; the bias b is always active so that recurrent dynamics
are well-defined even when x = 0.

Dynamics (both variants):
    h[t] = (1-alpha) h[t-1] + alpha tanh(W_rec h[t-1] + W_xh x[t] + b)
    y[t] = W_hy h[t]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class VanillaRateRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 alpha: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.W_hy = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.W_xh.weight)
        xavier_normal_(self.W_hh.weight)
        xavier_normal_(self.W_hy.weight)
        nn.init.zeros_(self.W_hy.bias)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None,
                noise_std: float = 0.0):
        """
        Args:
            x:  (B, T, input_dim) — zeros after teacher-forcing cutoff
            h0: (B, hidden_dim) or None -> zeros
            noise_std: if > 0, add Gaussian noise to h at each step (training only)

        Returns:
            y:     (B, T, output_dim)
            h_all: (B, T, hidden_dim)
        """
        B, T, _ = x.shape
        h = h0 if h0 is not None else x.new_zeros(B, self.hidden_dim)

        ys = []
        hs = []
        for t in range(T):
            pre = self.W_xh(x[:, t]) + self.W_hh(h) + self.bias
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(pre)
            if noise_std > 0 and self.training:
                h = h + noise_std * torch.randn_like(h)
            ys.append(self.W_hy(h))
            hs.append(h)

        return torch.stack(ys, dim=1), torch.stack(hs, dim=1)

    @torch.no_grad()
    def run_autonomous(self, h0: torch.Tensor, T: int):
        """Run T autonomous steps (zero input) from h0.  (B, H) -> (B, T, N)."""
        h = h0
        ys = []
        hs = [h]
        for _ in range(T):
            pre = self.W_hh(h) + self.bias
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(pre)
            ys.append(self.W_hy(h))
            hs.append(h)
        return torch.stack(ys, dim=1), torch.stack(hs, dim=1)

    def get_recurrent_weights(self) -> np.ndarray:
        return self.W_hh.weight.detach().cpu().numpy()


class LowRankRateRNN(nn.Module):
    """W_hh = U V^T / H  with U, V in R^{H x R}."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 rank: int = 2, alpha: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha

        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        # Init so UV^T/H has singular values ~ 1.5 (matching vanilla Xavier scale).
        # Each column of U,V needs norm ~ sqrt(target_sigma * H).
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) * np.sqrt(1.5))
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * np.sqrt(1.5))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.W_hy = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.W_xh.weight)
        xavier_normal_(self.W_hy.weight)
        nn.init.zeros_(self.W_hy.bias)

    def _recurrent(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.V) @ self.U.T / self.hidden_dim

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None,
                noise_std: float = 0.0):
        B, T, _ = x.shape
        h = h0 if h0 is not None else x.new_zeros(B, self.hidden_dim)

        ys = []
        hs = []
        for t in range(T):
            pre = self.W_xh(x[:, t]) + self._recurrent(h) + self.bias
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(pre)
            if noise_std > 0 and self.training:
                h = h + noise_std * torch.randn_like(h)
            ys.append(self.W_hy(h))
            hs.append(h)

        return torch.stack(ys, dim=1), torch.stack(hs, dim=1)

    @torch.no_grad()
    def run_autonomous(self, h0: torch.Tensor, T: int):
        h = h0
        ys = []
        hs = [h]
        for _ in range(T):
            pre = self._recurrent(h) + self.bias
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(pre)
            ys.append(self.W_hy(h))
            hs.append(h)
        return torch.stack(ys, dim=1), torch.stack(hs, dim=1)

    def get_recurrent_weights(self) -> np.ndarray:
        with torch.no_grad():
            return ((self.U @ self.V.T) / self.hidden_dim).cpu().numpy()

    def get_singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.get_recurrent_weights(), compute_uv=False)


def create_model(model_type: str, input_dim: int, hidden_dim: int,
                 output_dim: int, **kwargs) -> nn.Module:
    if model_type == "vanilla":
        return VanillaRateRNN(input_dim, hidden_dim, output_dim,
                              alpha=kwargs.get("alpha", 0.5))
    elif model_type == "lowrank":
        return LowRankRateRNN(input_dim, hidden_dim, output_dim,
                              rank=kwargs.get("rank", 2),
                              alpha=kwargs.get("alpha", 0.5))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
