from abc import abstractmethod
import numpy as np

from nonlinearity import NonlinearityFunction


class RingAttractor:
    def __init__(
        self,
        ring_size: int,
        init_strategy: "RingAttractorInitStrategy",
        nonlinearity_function: NonlinearityFunction,
        rng: np.random.Generator,
        init_noise_scale: float = 0.01,  # TODO: Pass this in a better way
    ):
        self.ring_size = ring_size
        self.rng = rng

        self.neuron_angles = self._initialize_neuron_angles()
        self.neuron_rates = self._initialize_neuron_rates(init_noise_scale)
        self.shape = self.neuron_rates.shape

        self.nonlinearity_function = nonlinearity_function
        self.init_strategy = init_strategy

        self.weights = init_strategy.compute_weights(self)

    def decode_theta(self) -> tuple[float, float]:
        """
        TODO: Find the correct citation for this method of decoding the theta. This takes the current
            ring state and returns the "predicted" theta.
            According to ChatGPT, this is the standard measure from circular statistics (Fisher 1993,
            Mardia & Jupp 2000).

        We return the angle and magnitude of the complex result, to indicate both the direction
        and the confidence in the prediction. (confidence = 0 -> random distribution in ring, confidence = 1 -> pure bump)
        """
        z = np.sum(self.neuron_rates * np.exp(1j * self.neuron_angles))

        return np.angle(z), np.abs(z)

    def _initialize_neuron_angles(self) -> np.ndarray:
        return 2 * np.pi * (np.arange(self.ring_size) / self.ring_size)

    def _initialize_neuron_rates(self, init_noise_scale: float = 0.01) -> np.ndarray:
        noise = self.rng.standard_normal(self.ring_size).astype(np.float32)
        return init_noise_scale * noise


class RingAttractorSimulator:
    def __init__(
        self,
        attractor: RingAttractor,
        dt: float,
        tau: float,  # TODO: Give tau + sigma better coding-oriented names
        sigma: float,  # TODO: What is this noise factor?
        rng: np.random.Generator,
    ):
        self.attractor = attractor
        self.dt = dt
        self.tau = tau
        self.sigma = sigma
        self.rng = rng

    def simulate_single_step(self, external_input: np.ndarray) -> np.ndarray:
        """
        External input is the input for every neuron in self.attractor
        """
        noise = self._create_noise()
        input_total = self.attractor.weights @ self.attractor.neuron_rates + external_input + noise * self.sigma

        rate_diff = -self.attractor.neuron_rates + self.attractor.nonlinearity_function.apply(input_total)
        new_rates = self.attractor.neuron_rates + rate_diff * self.dt / self.tau

        return new_rates

    def perform_single_step(self, external_input: np.ndarray) -> np.ndarray:
        new_rates = self.simulate_single_step(external_input)
        self.attractor.neuron_rates = new_rates
        return new_rates

    def _create_noise(self) -> np.ndarray:
        noise = self.rng.standard_normal(self.attractor.shape)
        return noise.astype(self.attractor.neuron_rates.dtype)


class RingAttractorInitStrategy:
    @abstractmethod
    def compute_weights(self, attractor: "RingAttractor") -> np.ndarray:
        raise NotImplementedError


class CosineKernelInitStrategy(RingAttractorInitStrategy):
    def __init__(self, j0: float, j1: float):
        self.j0 = j0
        self.j1 = j1

    def compute_weights(self, attractor: "RingAttractor") -> np.ndarray:
        """
        Initialize the weights using a cosine kernel that makes close neurons have positive weight
        and distant neurons have negative weight.

        Based on:
            W_ij = J0 + J1 * cos(theta_i - theta_j)

        J1 > 0 - local excitation
        J0 < 0 - global inhibition (pulls neurons towards 0)

        TODO: What other kernel types can we try here?
        TODO: What values for J0 and J1 are good? I understand -2.0 and 4.0 are okay
            but we need to see if any of the papers we have use different ones...
        """

        angles = attractor.neuron_angles
        theta_diffs = angles[:, None] - angles[None, :]
        cosine_diffs = np.cos(theta_diffs)
        weights = self.j0 + self.j1 * cosine_diffs

        return weights
