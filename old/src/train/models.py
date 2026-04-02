"""
RNN model architectures for ring attractor reconstruction.

Two architectures:
  - VanillaRateRNN: Standard RNN with full recurrent weights
  - LowRankRateRNN: RNN with low-rank constraint on recurrent weights
"""

import torch
import torch.nn as nn
import numpy as np


class VanillaRateRNN(nn.Module):
    """
    Standard rate-based RNN.

    Dynamics:
        h[t] = (1 - alpha) * h[t-1] + alpha * tanh(W_xh @ x[t] + W_hh @ h[t-1])
        y[t] = W_hy @ h[t]

    Args:
        input_dim: Dimension of input (observed neurons)
        hidden_dim: Dimension of hidden state
        output_dim: Dimension of output (full population)
        alpha: Leak rate (0 < alpha <= 1), controls integration timescale
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.W_xh = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hy = nn.Linear(hidden_dim, output_dim)

        self.nonlinearity = torch.tanh

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            h0: Optional initial hidden state (batch, hidden_dim)

        Returns:
            y: Output tensor (batch, time, output_dim)
            h_all: Hidden states (batch, time, hidden_dim)
        """
        B, T, _ = x.shape

        if h0 is None:
            h = torch.zeros(B, self.hidden_dim, device=x.device)
        else:
            h = h0

        ys = []
        hs = []

        for t in range(T):
            pre = self.W_xh(x[:, t]) + self.W_hh(h)
            h = (1 - self.alpha) * h + self.alpha * self.nonlinearity(pre)
            y = self.W_hy(h)

            ys.append(y.unsqueeze(1))
            hs.append(h.unsqueeze(1))

        y_out = torch.cat(ys, dim=1)
        h_all = torch.cat(hs, dim=1)

        return y_out, h_all

    def get_recurrent_weights(self) -> np.ndarray:
        """Return the recurrent weight matrix as numpy array."""
        return self.W_hh.weight.detach().cpu().numpy()


class LowRankRateRNN(nn.Module):
    """
    Low-rank rate-based RNN.

    The recurrent weight matrix is factorized as W_hh = U @ V.T / hidden_dim,
    constraining it to have rank <= rank parameter.

    This is motivated by the fact that ring attractors have rank-2 connectivity
    (cosine kernel = sum of two outer products).

    Args:
        input_dim: Dimension of input (observed neurons)
        hidden_dim: Dimension of hidden state
        output_dim: Dimension of output (full population)
        rank: Rank of the recurrent weight matrix
        alpha: Leak rate (0 < alpha <= 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rank: int = 2,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha

        self.W_xh = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) / np.sqrt(hidden_dim))
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) / np.sqrt(hidden_dim))
        self.W_hy = nn.Linear(hidden_dim, output_dim)

        self.nonlinearity = torch.tanh

    def _recurrent_map(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank recurrent transformation."""
        return (h @ self.V) @ self.U.T / self.hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            h0: Optional initial hidden state (batch, hidden_dim)

        Returns:
            y: Output tensor (batch, time, output_dim)
            h_all: Hidden states (batch, time, hidden_dim)
        """
        B, T, _ = x.shape

        if h0 is None:
            h = torch.zeros(B, self.hidden_dim, device=x.device)
        else:
            h = h0

        ys = []
        hs = []

        for t in range(T):
            pre = self.W_xh(x[:, t]) + self._recurrent_map(h)
            h = (1 - self.alpha) * h + self.alpha * self.nonlinearity(pre)
            y = self.W_hy(h)

            ys.append(y.unsqueeze(1))
            hs.append(h.unsqueeze(1))

        y_out = torch.cat(ys, dim=1)
        h_all = torch.cat(hs, dim=1)

        return y_out, h_all

    def get_recurrent_weights(self) -> np.ndarray:
        """Return the full recurrent weight matrix as numpy array."""
        with torch.no_grad():
            W = (self.U @ self.V.T) / self.hidden_dim
            return W.cpu().numpy()

    def get_singular_values(self) -> np.ndarray:
        """Return singular values of the recurrent weight matrix."""
        W = self.get_recurrent_weights()
        return np.linalg.svd(W, compute_uv=False)


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create RNN models.

    Args:
        model_type: "vanilla" or "lowrank"
        input_dim: Input dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific arguments (alpha, rank, etc.)

    Returns:
        Instantiated model
    """
    if model_type == "vanilla":
        return VanillaRateRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            alpha=kwargs.get("alpha", 0.1),
        )
    elif model_type == "lowrank":
        return LowRankRateRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rank=kwargs.get("rank", 2),
            alpha=kwargs.get("alpha", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
