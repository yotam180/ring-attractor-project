import numpy as np


class SpikeGenerator:
    def __init__(
        self,
        dt: float,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.dt = dt
        self.rng = rng

    def generate_spikes(self, rates: np.ndarray) -> np.ndarray:
        """
        Since Poisson is "memoryless" we don't need to keep track of previous spikes.
        If we had to that would complicate the interface of this class just a bit, we
        might want to think how to adjust it anyway...
        """
        clamped = np.clip(rates, 0, None)  # Should we set a max value?
        spikes = self.rng.poisson(clamped * self.dt)

        return spikes


class SpikeProcessor:
    def __init__(self, dt: float, bin_factor: int):
        self.dt = dt
        self.bin_factor = bin_factor

    def process_spikes(self, spikes: np.ndarray) -> np.ndarray:
        # TODO: ChatGPT wrote this line I'm not sure this actually does the job?
        return spikes.reshape(-1, self.bin_factor).sum(axis=1) * self.dt
