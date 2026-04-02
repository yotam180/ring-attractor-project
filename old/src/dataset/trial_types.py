"""
Trial type definitions for ring attractor dataset generation.

Three trial families as specified in plans/dataset_generation.md:
  A) Noise-only exploration trials
  B) Single-cue memory trials  
  C) Perturbation/update trials (held-out for mechanistic validation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TrialType(Enum):
    NOISE_ONLY = "noise_only"
    SINGLE_CUE = "single_cue"
    PERTURBATION = "perturbation"


@dataclass
class TrialConfig:
    """Configuration for a single trial."""

    trial_type: TrialType
    total_steps: int
    network_size: int

    # For cue trials
    theta_target: float | None = None
    cue_amplitude: float = 2.0
    cue_onset: int = 0
    cue_duration: int = 200

    # For perturbation trials
    perturbation_theta: float | None = None
    perturbation_onset: int | None = None
    perturbation_duration: int = 100
    perturbation_amplitude: float = 1.5


class CueScheduleBuilder(ABC):
    """Base class for building cue schedules."""

    @abstractmethod
    def build(self, config: TrialConfig, neuron_angles: np.ndarray) -> np.ndarray:
        """
        Build the cue schedule for a trial.

        Args:
            config: Trial configuration
            neuron_angles: Preferred angles for each neuron (shape: N,)

        Returns:
            Cue schedule array of shape (total_steps, network_size)
        """
        raise NotImplementedError


class NoiseOnlyCueBuilder(CueScheduleBuilder):
    """
    Type A: Noise-only trials.

    No external cue — the network explores state space driven only by noise.
    Used to test whether the student invents discretization under dropout.
    """

    def build(self, config: TrialConfig, neuron_angles: np.ndarray) -> np.ndarray:
        return np.zeros((config.total_steps, config.network_size), dtype=np.float32)


class SingleCueCueBuilder(CueScheduleBuilder):
    """
    Type B: Single-cue memory trials.

    Brief cue at theta_target, then delay period where bump persists and diffuses.
    The canonical ring attractor maintenance test.
    """

    def build(self, config: TrialConfig, neuron_angles: np.ndarray) -> np.ndarray:
        if config.theta_target is None:
            raise ValueError("theta_target required for single-cue trials")

        schedule = np.zeros((config.total_steps, config.network_size), dtype=np.float32)

        cue_start = config.cue_onset
        cue_end = min(cue_start + config.cue_duration, config.total_steps)

        cue_pattern = config.cue_amplitude * np.cos(neuron_angles - config.theta_target)
        schedule[cue_start:cue_end] = cue_pattern

        return schedule


class PerturbationCueBuilder(CueScheduleBuilder):
    """
    Type C: Perturbation/update trials.

    Initial cue sets the bump, then a second cue (perturbation) tests updating.
    Used for held-out mechanistic validation.
    """

    def build(self, config: TrialConfig, neuron_angles: np.ndarray) -> np.ndarray:
        if config.theta_target is None:
            raise ValueError("theta_target required for perturbation trials")
        if config.perturbation_theta is None:
            raise ValueError("perturbation_theta required for perturbation trials")
        if config.perturbation_onset is None:
            raise ValueError("perturbation_onset required for perturbation trials")

        schedule = np.zeros((config.total_steps, config.network_size), dtype=np.float32)

        # Initial cue
        cue_start = config.cue_onset
        cue_end = min(cue_start + config.cue_duration, config.total_steps)
        cue_pattern = config.cue_amplitude * np.cos(neuron_angles - config.theta_target)
        schedule[cue_start:cue_end] = cue_pattern

        # Perturbation cue
        pert_start = config.perturbation_onset
        pert_end = min(pert_start + config.perturbation_duration, config.total_steps)
        pert_pattern = config.perturbation_amplitude * np.cos(
            neuron_angles - config.perturbation_theta
        )
        schedule[pert_start:pert_end] = pert_pattern

        return schedule


def get_cue_builder(trial_type: TrialType) -> CueScheduleBuilder:
    """Factory function to get the appropriate cue builder."""
    builders = {
        TrialType.NOISE_ONLY: NoiseOnlyCueBuilder(),
        TrialType.SINGLE_CUE: SingleCueCueBuilder(),
        TrialType.PERTURBATION: PerturbationCueBuilder(),
    }
    return builders[trial_type]


@dataclass
class TrialData:
    """Container for a single trial's data."""

    trial_type: TrialType
    config: TrialConfig

    # Raw simulation outputs (at integration resolution)
    spikes_int: np.ndarray  # (T_int, N) integer spike counts
    rates_true: np.ndarray  # (T_int, N) ground-truth firing rates
    theta_hat: np.ndarray  # (T_int,) decoded bump angle
    confidence: np.ndarray  # (T_int,) decoding confidence
    cue_schedule: np.ndarray  # (T_int, N) external input

    # Processed outputs (at bin resolution)
    spikes_bin: np.ndarray  # (T_bin, N) binned spike counts
    rates_smooth: np.ndarray  # (T_bin, N) smoothed rate estimates
    theta_hat_bin: np.ndarray  # (T_bin,) decoded angle at bin resolution
