from abc import abstractmethod, ABC
from typing import Callable
import numpy as np


class NonlinearityFunction(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLUNonlinearity(NonlinearityFunction):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class SoftplusNonlinearity(NonlinearityFunction):
    def __init__(self, beta: float = 1.0):
        # TODO: Document what beta does
        self.beta = beta

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(self.beta * x)) / self.beta


class TanhNonlinearity(NonlinearityFunction):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
