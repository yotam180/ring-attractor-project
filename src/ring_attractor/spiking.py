import numpy as np

from ring_attractor.network import RingAttractorSimulator


class SpikeGenerator:
    def __init__(
        self,
        dt: float,
        rate_scale: float = 1.0,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.dt = dt
        self.rate_scale = rate_scale
        self.rng = rng

    def generate_spikes(self, rates: np.ndarray) -> np.ndarray:
        """
        Since Poisson is "memoryless" we don't need to keep track of previous spikes.
        If we had to that would complicate the interface of this class just a bit, we
        might want to think how to adjust it anyway...
        """
        clamped = np.clip(rates * self.rate_scale, 0, None)
        return self.rng.poisson(clamped * self.dt)


class SpikeGeneratorSimulator:
    def __init__(self, generator: SpikeGenerator, simulator: RingAttractorSimulator):
        self.generator = generator
        self.simulator = simulator
        self.network_size = simulator.attractor.ring_size
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated recordings."""
        self.spikes = np.zeros((0, self.network_size), dtype=np.int32)
        self.neuron_rates = np.zeros((0, self.network_size), dtype=np.float32)
        self.decoded_angle = np.zeros(0)
        self.decoding_confidence = np.zeros(0)

    def perform_steps(self, cue: np.ndarray) -> None:
        timesteps = cue.shape[0]

        spikes = np.zeros((timesteps, self.network_size), dtype=np.int32)
        rates = np.zeros((timesteps, self.network_size), dtype=np.float32)
        thetas = np.zeros(timesteps)
        confs = np.zeros(timesteps)

        for t in range(timesteps):
            # Snapshot BEFORE stepping so spikes[t] and rates[t] are aligned.
            rates[t] = self.simulator.attractor.neuron_rates
            spikes[t] = self.generator.generate_spikes(rates[t])

            angle, conf = self.simulator.attractor.decode_theta()
            thetas[t] = angle
            confs[t] = conf
            self.simulator.perform_single_step(cue[t])

        # Append the new data to the existing simulator history
        self.spikes = np.concatenate([self.spikes, spikes])
        self.neuron_rates = np.concatenate([self.neuron_rates, rates])
        self.decoded_angle = np.concatenate([self.decoded_angle, thetas])
        self.decoding_confidence = np.concatenate([self.decoding_confidence, confs])


class SpikeProcessor:
    def __init__(self, dt: float, bin_factor: int):
        self.dt = dt
        self.bin_factor = bin_factor

    def bin_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """
        Bin spike counts along the time axis.

        Input shape:  (t - timestamps, n - neurons)
        Output shape: (t // bin_factor - number of bins, n - neurons)
        """
        t, n = spikes.shape
        t_bin = t // self.bin_factor
        return spikes[: t_bin * self.bin_factor].reshape(t_bin, self.bin_factor, n).sum(axis=1)
