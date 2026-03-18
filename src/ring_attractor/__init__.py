"""
Ring attractor network simulation.

Components:
  - RingAttractor: The network state and weights
  - RingAttractorSimulator: Euler integration of dynamics
  - CosineKernelInitStrategy: Standard ring attractor connectivity
  - SpikeGenerator: Poisson spike generation from rates
  - SpikeProcessor: Binning and smoothing
"""

from ring_attractor.network import (
    RingAttractor,
    RingAttractorSimulator,
    RingAttractorInitStrategy,
    CosineKernelInitStrategy,
)
from ring_attractor.nonlinearity import (
    NonlinearityFunction,
    ReLUNonlinearity,
    SoftplusNonlinearity,
    TanhNonlinearity,
)
from ring_attractor.spiking import (
    SpikeGenerator,
    SpikeGeneratorSimulator,
    SpikeProcessor,
)

__all__ = [
    "RingAttractor",
    "RingAttractorSimulator",
    "RingAttractorInitStrategy",
    "CosineKernelInitStrategy",
    "NonlinearityFunction",
    "ReLUNonlinearity",
    "SoftplusNonlinearity",
    "TanhNonlinearity",
    "SpikeGenerator",
    "SpikeGeneratorSimulator",
    "SpikeProcessor",
]
