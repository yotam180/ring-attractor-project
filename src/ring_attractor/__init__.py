"""
Ring attractor simulator package.

Quick-start::

    from src.ring_attractor import RingAttractor, SpikeProcessor

    ring = RingAttractor()
    result = ring.simulate(T=7500, cue_angles=[1.0], cue_duration=2000, seed=42)
    result.rates        # (T, N) firing rates
    result.theta        # (T,)   decoded angle
    result.confidence   # (T,)   decoding confidence

    sp = SpikeProcessor()
    data = sp.process(result.rates, seed=42)
    data.smoothed       # (T_bin, N) smoothed binned spike counts
"""

from . import defaults
from .network import RingAttractor, SimResult, decode_theta
from .spiking import SpikeProcessor, SpikeData

__all__ = [
    "defaults",
    "RingAttractor",
    "SimResult",
    "SpikeProcessor",
    "SpikeData",
    "decode_theta",
]
