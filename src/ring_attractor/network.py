"""
Ring attractor network and simulator.

Core class: RingAttractor — holds network parameters (weights, angles,
nonlinearity) and runs simulations via .simulate().

Dynamics (discrete time, with divisive normalisation):
    g[t] = 1 + γ · mean(r[t])
    h[t+1] = h[t] + α(-h[t] + φ((W h[t] + I_ext[t] + σ ξ[t]) / g[t]))

Nonlinearity:
    φ(x) = tanh(steepness × max(0, x))

The divisive normalisation factor g creates competition between neurons:
as total activity rises, the effective input to every neuron is scaled
down, which sharpens the bump and produces a properly peaked profile
instead of a flat-topped mesa.

Weight matrix (cosine kernel, normalised by N):
    W_ij = (J0 + J1 cos(θ_i - θ_j)) / N
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """Container for simulation output."""

    rates: np.ndarray  # (T, N)  non-negative firing rates ∈ [0, 1]
    angles: np.ndarray  # (N,)    preferred angle of each neuron
    theta: np.ndarray  # (T,)    decoded bump angle per step
    confidence: np.ndarray  # (T,)    decoding confidence per step
    weights: np.ndarray  # (N, N)  connectivity matrix


# ---------------------------------------------------------------------------
# Ring attractor
# ---------------------------------------------------------------------------


class RingAttractor:
    """
    Ring attractor network with cosine connectivity.

    Parameters
    ----------
    N : int
        Number of neurons on the ring.
    J0, J1 : float
        Inhibition baseline / excitation amplitude for the cosine kernel.
    steepness : float
        Slope of φ(x) = tanh(s · ReLU(x)).  Controls bump sharpness.
    alpha : float
        Leak rate (= dt/τ).  Controls integration speed.
    sigma : float
        Additive Gaussian noise amplitude.
    gamma : float
        Divisive normalisation strength.  The recurrent input is divided
        by (1 + γ · mean(r)), which sharpens the bump by creating
        activity-dependent gain control.  γ=0 disables it.
    """

    def __init__(
        self,
        N: int = 100,
        J0: float = -2.0,
        J1: float = 4.0,
        steepness: float = 4.0,
        alpha: float = 0.01,
        sigma: float = 0.1,
        gamma: float = 2.0,
    ):
        self.N = N
        self.J0 = J0
        self.J1 = J1
        self.steepness = steepness
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma

        self.angles = 2 * np.pi * np.arange(N) / N
        self.weights = self._make_weights()

    # -- weight matrix -----------------------------------------------------

    def _make_weights(self) -> np.ndarray:
        dtheta = self.angles[:, None] - self.angles[None, :]
        return (self.J0 + self.J1 * np.cos(dtheta)) / self.N

    # -- nonlinearity ------------------------------------------------------

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Threshold-saturating activation: tanh(steepness × ReLU(x))."""
        return np.tanh(self.steepness * np.maximum(x, 0.0))

    # -- simulation --------------------------------------------------------

    def simulate(
        self,
        T: int = 5000,
        cue_angles: list[float] | None = None,
        cue_onset: int = 0,
        cue_duration: int = 2000,
        cue_amplitude: float = 3.0,
        cue_sigma: float = np.radians(20),
        init_rates: np.ndarray | None = None,
        init_noise_scale: float = 0.01,
        seed: int | None = None,
    ) -> SimResult:
        """
        Run the ring attractor for *T* integration steps.

        Parameters
        ----------
        T : int
            Number of steps.
        cue_angles : list[float] | None
            Angles (rad) at which to inject a Gaussian cue.
        cue_onset, cue_duration : int
            When the cue starts and how long it lasts.
        cue_amplitude : float
            Peak amplitude of the Gaussian cue.
        cue_sigma : float
            Width (std, rad) of the Gaussian cue envelope.
        init_rates : ndarray (N,) | None
            Initial hidden state.  ``None`` → small Gaussian noise.
        init_noise_scale : float
            Std of initial noise when *init_rates* is None.
        seed : int | None
            Random seed.

        Returns
        -------
        SimResult
        """
        rng = np.random.default_rng(seed)
        W = self.weights
        angles = self.angles
        N = self.N
        alpha = self.alpha
        sigma = self.sigma

        # initial state
        if init_rates is not None:
            h = np.array(init_rates, dtype=np.float64)
        else:
            h = init_noise_scale * rng.standard_normal(N)

        # pre-compute cue envelope per cue angle (constant across time)
        cue_envelopes: list[np.ndarray] = []
        if cue_angles is not None:
            for ca in cue_angles:
                d = np.angle(np.exp(1j * (angles - ca)))
                cue_envelopes.append(
                    cue_amplitude * np.exp(-(d**2) / (2 * cue_sigma**2))
                )

        # storage
        rates = np.empty((T, N), dtype=np.float64)
        theta = np.empty(T, dtype=np.float64)
        confidence = np.empty(T, dtype=np.float64)

        for t in range(T):
            r = self.phi(h)
            rates[t] = r
            th, conf = decode_theta_single(r, angles)
            theta[t] = th
            confidence[t] = conf

            # external cue
            I_ext = np.zeros(N, dtype=np.float64)
            if cue_envelopes and cue_onset <= t < cue_onset + cue_duration:
                for env in cue_envelopes:
                    I_ext += env

            # Euler step with divisive normalisation
            noise = sigma * rng.standard_normal(N)
            total_input = W @ h + I_ext + noise
            if self.gamma > 0:
                gain = 1.0 + self.gamma * r.mean()
                total_input = total_input / gain
            dh = (-h + self.phi(total_input)) * alpha
            h = h + dh

        return SimResult(
            rates=rates,
            angles=angles,
            theta=theta,
            confidence=confidence,
            weights=W,
        )


# ---------------------------------------------------------------------------
# Theta decoding  (population vector)
# ---------------------------------------------------------------------------


def decode_theta_single(rates: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    """Decode (angle, confidence) from a single rate vector."""
    z = np.sum(rates * np.exp(1j * angles))
    total = rates.sum() + 1e-12
    return float(np.angle(z)), float(np.abs(z) / total)


def decode_theta(
    rates: np.ndarray, angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised decoding over a (T, N) rate matrix."""
    z = rates @ np.exp(1j * angles)
    total = rates.sum(axis=1) + 1e-12
    return np.angle(z), np.abs(z) / total
