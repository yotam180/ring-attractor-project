import numpy as np


class NeuronDropout:
    """
    Randomly select a subset of neurons to keep and mask out the rest.

    The neuron mask is fixed at construction time (determined by keep_fraction),
    so repeated calls to reduce() always drop the same neurons.

    This is important for consistency across training runs so that the student
    always sees data in the same N-dimensional format, with the same columns zeroed out.
    """

    def __init__(self, network_size: int, keep_fraction: float, rng: np.random.Generator = None):
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"invalid keep_fraction")

        self.network_size = network_size
        self.keep_fraction = keep_fraction
        self.rng = rng if rng is not None else np.random.default_rng()

        n_keep = max(1, round(network_size * keep_fraction))
        all_indices = np.arange(network_size)
        self.kept_indices = np.sort(self.rng.choice(all_indices, size=n_keep, replace=False))

        self.mask = np.zeros(network_size, dtype=bool)
        self.mask[self.kept_indices] = True

    @property
    def neurons_kept(self) -> int:
        return int(self.mask.sum())

    @property
    def neurons_dropped(self) -> int:
        return self.network_size - self.neurons_kept

    def reduce(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        out[:, ~self.mask] = 0.0
        return out
