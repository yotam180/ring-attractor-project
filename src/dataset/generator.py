"""
Dataset generation for ring attractor experiments.

Generates multi-trial datasets with three trial families:
  A) Noise-only exploration
  B) Single-cue memory
  C) Perturbation/update

Compatible with the RNN training pipeline.
"""

from dataclasses import dataclass, field
import numpy as np

from ring_attractor.network import (
    RingAttractor,
    RingAttractorSimulator,
    CosineKernelInitStrategy,
)
from ring_attractor.nonlinearity import ReLUNonlinearity, NonlinearityFunction
from ring_attractor.spiking import SpikeGenerator, SpikeGeneratorSimulator, SpikeProcessor

from dataset.trial_types import (
    TrialType,
    TrialConfig,
    TrialData,
    get_cue_builder,
)


@dataclass
class DatasetConfig:
    """Configuration for the entire dataset."""

    # Network parameters
    network_size: int = 100
    j0: float = -2.0
    j1: float = 4.0

    # Simulation parameters
    dt: float = 0.01  # Integration timestep (dimensionless time units)
    tau: float = 100.0  # Neural time constant
    sigma: float = 0.1  # Noise level

    # Spike generation
    rate_scale: float = 100.0  # Scale rates before Poisson sampling

    # Binning/smoothing
    bin_factor: int = 50  # Integration steps per bin
    smoothing_window: int = 3  # Bins for causal smoothing

    # Trial structure
    steps_per_trial: int = 5000  # Integration steps per trial
    cue_duration: int = 200  # Steps with cue ON
    cue_amplitude: float = 2.0

    # Perturbation trials
    perturbation_delay: int = 1500  # Steps after cue before perturbation
    perturbation_duration: int = 150
    perturbation_amplitude: float = 1.5

    # Trial counts per type
    n_noise_trials: int = 10
    n_single_cue_trials: int = 30
    n_perturbation_trials: int = 10

    # Train/test split
    test_fraction: float = 0.2

    # Random seed
    seed: int = 42

    @property
    def bins_per_trial(self) -> int:
        return self.steps_per_trial // self.bin_factor

    @property
    def dt_bin(self) -> float:
        return self.dt * self.bin_factor

    @property
    def total_trials(self) -> int:
        return self.n_noise_trials + self.n_single_cue_trials + self.n_perturbation_trials


@dataclass
class GeneratedDataset:
    """Container for the complete generated dataset."""

    config: DatasetConfig

    # Per-trial data lists
    trials: list[TrialData] = field(default_factory=list)

    # Stacked arrays for RNN training (created by finalize())
    X_cue: np.ndarray | None = None  # (n_trials, T_bin, N) cue schedule
    Y_rates: np.ndarray | None = None  # (n_trials, T_bin, N) target rates
    Y_theta: np.ndarray | None = None  # (n_trials, T_bin) target angles

    # Train/test indices
    train_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None

    # Trial metadata
    trial_types: list[TrialType] = field(default_factory=list)
    theta_targets: list[float | None] = field(default_factory=list)

    def finalize(self, rng: np.random.Generator) -> None:
        """Stack trial data into arrays and create train/test split."""
        n_trials = len(self.trials)
        if n_trials == 0:
            raise ValueError("No trials to finalize")

        T_bin = self.trials[0].rates_smooth.shape[0]
        N = self.config.network_size

        # Stack arrays
        self.X_cue = np.zeros((n_trials, T_bin, N), dtype=np.float32)
        self.Y_rates = np.zeros((n_trials, T_bin, N), dtype=np.float32)
        self.Y_theta = np.zeros((n_trials, T_bin), dtype=np.float32)

        for i, trial in enumerate(self.trials):
            # Bin the cue schedule to match Y_rates resolution
            cue_binned = trial.cue_schedule[: T_bin * self.config.bin_factor].reshape(
                T_bin, self.config.bin_factor, N
            ).mean(axis=1)
            self.X_cue[i] = cue_binned
            self.Y_rates[i] = trial.rates_smooth
            self.Y_theta[i] = trial.theta_hat_bin

            self.trial_types.append(trial.trial_type)
            self.theta_targets.append(trial.config.theta_target)

        # Train/test split (stratified by trial type would be better, but simple random for now)
        n_test = max(1, int(n_trials * self.config.test_fraction))
        all_idx = np.arange(n_trials)
        rng.shuffle(all_idx)
        self.test_idx = np.sort(all_idx[:n_test])
        self.train_idx = np.sort(all_idx[n_test:])


class DatasetGenerator:
    """
    Generates multi-trial ring attractor datasets.

    Usage:
        config = DatasetConfig(network_size=100, n_single_cue_trials=40)
        generator = DatasetGenerator(config)
        dataset = generator.generate()
        save_dataset(dataset, "my_dataset.npz")
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> GeneratedDataset:
        """Generate the complete dataset."""
        dataset = GeneratedDataset(config=self.config)

        # Generate each trial type
        self._generate_noise_trials(dataset)
        self._generate_single_cue_trials(dataset)
        self._generate_perturbation_trials(dataset)

        # Finalize (stack arrays, create split)
        dataset.finalize(self.rng)

        return dataset

    def _generate_noise_trials(self, dataset: GeneratedDataset) -> None:
        """Generate Type A: noise-only exploration trials."""
        for _ in range(self.config.n_noise_trials):
            config = TrialConfig(
                trial_type=TrialType.NOISE_ONLY,
                total_steps=self.config.steps_per_trial,
                network_size=self.config.network_size,
            )
            trial = self._run_trial(config)
            dataset.trials.append(trial)

    def _generate_single_cue_trials(self, dataset: GeneratedDataset) -> None:
        """Generate Type B: single-cue memory trials with uniform theta coverage."""
        thetas = np.linspace(0, 2 * np.pi, self.config.n_single_cue_trials, endpoint=False)
        self.rng.shuffle(thetas)

        for theta in thetas:
            config = TrialConfig(
                trial_type=TrialType.SINGLE_CUE,
                total_steps=self.config.steps_per_trial,
                network_size=self.config.network_size,
                theta_target=float(theta),
                cue_amplitude=self.config.cue_amplitude,
                cue_onset=0,
                cue_duration=self.config.cue_duration,
            )
            trial = self._run_trial(config)
            dataset.trials.append(trial)

    def _generate_perturbation_trials(self, dataset: GeneratedDataset) -> None:
        """Generate Type C: perturbation trials for mechanistic validation."""
        for _ in range(self.config.n_perturbation_trials):
            theta_init = self.rng.uniform(0, 2 * np.pi)
            theta_pert = self.rng.uniform(0, 2 * np.pi)

            config = TrialConfig(
                trial_type=TrialType.PERTURBATION,
                total_steps=self.config.steps_per_trial,
                network_size=self.config.network_size,
                theta_target=float(theta_init),
                cue_amplitude=self.config.cue_amplitude,
                cue_onset=0,
                cue_duration=self.config.cue_duration,
                perturbation_theta=float(theta_pert),
                perturbation_onset=self.config.cue_duration + self.config.perturbation_delay,
                perturbation_duration=self.config.perturbation_duration,
                perturbation_amplitude=self.config.perturbation_amplitude,
            )
            trial = self._run_trial(config)
            dataset.trials.append(trial)

    def _run_trial(self, config: TrialConfig) -> TrialData:
        """Run a single trial simulation."""
        # Create fresh network for each trial
        attractor = RingAttractor(
            ring_size=self.config.network_size,
            init_strategy=CosineKernelInitStrategy(j0=self.config.j0, j1=self.config.j1),
            nonlinearity_function=ReLUNonlinearity(),
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        simulator = RingAttractorSimulator(
            attractor=attractor,
            dt=self.config.dt,
            tau=self.config.tau,
            sigma=self.config.sigma,
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        spike_gen = SpikeGenerator(
            dt=self.config.dt,
            rate_scale=self.config.rate_scale,
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        sgs = SpikeGeneratorSimulator(generator=spike_gen, simulator=simulator)

        # Build cue schedule
        cue_builder = get_cue_builder(config.trial_type)
        cue_schedule = cue_builder.build(config, attractor.neuron_angles)

        # Run simulation
        sgs.perform_steps(cue_schedule)

        # Process spikes
        processor = SpikeProcessor(
            dt=self.config.dt,
            bin_factor=self.config.bin_factor,
            smoothing_window=self.config.smoothing_window,
        )

        spikes_bin = processor.bin_spikes(sgs.spikes)
        rates_smooth = processor.smooth_bins(spikes_bin) / self.config.dt_bin

        # Downsample theta to bin resolution
        T_bin = spikes_bin.shape[0]
        theta_hat_bin = sgs.decoded_angle[: T_bin * self.config.bin_factor : self.config.bin_factor]

        return TrialData(
            trial_type=config.trial_type,
            config=config,
            spikes_int=sgs.spikes,
            rates_true=sgs.neuron_rates,
            theta_hat=sgs.decoded_angle,
            confidence=sgs.decoding_confidence,
            cue_schedule=cue_schedule,
            spikes_bin=spikes_bin,
            rates_smooth=rates_smooth,
            theta_hat_bin=theta_hat_bin,
        )
