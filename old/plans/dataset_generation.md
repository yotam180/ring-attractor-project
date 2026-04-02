Yes — we can answer the dataset-generation question from the papers you attached, and you do **not** need to re-paste them.

What’s important is that _different papers solve “generate data” in different ways depending on what they want to validate_: some generate **free-running** activity under noise, some generate **trial-structured** input–delay–go trajectories, and some generate **target trajectories** (PSTHs or synthetic sinusoids) that the RNN is trained to reproduce. Your Phase 3 design should pick one of these _on purpose_, because it determines what “mechanistic recovery” even means.

## 1) What your project document actually commits you to

Your proposal defines the _axes of incompleteness_ as (i) neuron dropout and (ii) shortening the observed time window, and it evaluates both **predictive fit** and **mechanistic class recovery** (fixed points / perturbations / ring-like attractor criteria).

So Phase 3’s dataset generation must support _both_:

- training the student RNN to predict held-out data well, and
- later probing the trained student to see whether it really has a **continuous ring**, rather than “discretized bumps”.

## 2) How the “partial observation → wrong mechanism” paper generates data and evaluates models

The most directly relevant reference for Phase 3 is **Qian et al. (Partial observation can induce mechanistic mismatches…)** because it is _exactly_ about partial observation and mechanistic mismatch.

### How they generate teacher data

They study teacher–student RNNs driven by either:

- **known low-dimensional inputs** in a task setting (their “motivating example”), or
- **isotropic Gaussian white noise** to ensure broad exploration of state space (their OU-process formulation).

The key practical point: when they want to remove “task design” confounds and test identifiability, they deliberately use _noise-driven activity_ so the network explores directions broadly.

### How they implement partial observation

They do _exactly what your proposal suggests_: they observe only a subset of neurons for a duration (T). In their analytic setup, observation is literally a projection (x\_{\text{obs}}(t) = P z(t)) over a time window ([0,T]).

### How they evaluate “mechanism recovered”

They do not rely on “loss” alone. They evaluate mechanistic recovery with dynamical-systems diagnostics:

- **eigenspectrum / time constants** (gap structure is the tell for spurious line/plane attractors),
- **flow fields** in PCA space (learned vs ground truth),
- explicit “line attractor score” based on the ratio of the top timescales, and
- showing cases where the student matches observed activity but has **spurious attractor structure**.

### Concrete sim settings (useful for your Phase 1/3 notebooks)

They report a very standard simulation recipe: Euler integration, fixed (\tau), chosen (\Delta t), process noise, and long time series for finite-window experiments.

**Implication for your Phase 3 dataset:**
If you want a clean “ground truth ring” benchmark with controlled omission, you should have a **noise-driven condition** (no cues at all) in addition to any cue-driven trials. Noise-driven data is the simplest way to test whether the student invents discretization when neurons/time are reduced.

## 3) How the ALM “attractor dynamics gate information flow” paper generates training data

This paper is not about ring attractors per se, but it is extremely informative about **trial structure** and **how to train and validate an RNN surrogate** when inputs happen in epochs.

### How they generate training targets

They train RNN units to match **PSTHs** from real neurons on correct trials without distractors, and they include an **external ramping input** to capture a nonselective ramping component.

That is a “dataset generation” strategy: the dataset is _trial-averaged trajectories_, aligned to task epochs (sample → delay → go), and the model is validated by generalizing to conditions it was not trained on (distractors).

### How they validate mechanism

They don’t stop at fit. They reverse engineer dynamics via:

- fixed point / saddle structure,
- robustness to perturbations,
- how attractor separation changes over time / with ramp amplitude.

**Implication for your Phase 3 dataset:**
A good “trial library” is not “one cue then decay.” It should include:

- baseline / pre-cue,
- cue epoch that sets the state,
- delay epoch where state persists and diffuses,
- optional perturbations/distractors used _only for evaluation_ (or used as held-out trial types).

That is exactly the pattern they use: train on a clean subset, test on perturbed variants.

## 4) How the “Distributing task-related activity…” paper generates synthetic datasets

This paper is useful because it shows a controlled way to generate inputs and targets in simulation (again, not ring-specific, but directly applicable).

They explicitly describe:

- **Ornstein–Uhlenbeck** stimulus generation (finite-duration external drive) to trigger patterns,
- synthetic target patterns such as **sinusoids with random phases** (a clean “known ground truth” benchmark),
- and training on recorded-like trajectories (PSTHs) for a subset of neurons.

**Implication for your Phase 3 dataset:**
If you want a rigorous dataset that covers the ring well, the cleanest option is:

- choose (\theta_0 \sim \mathrm{Uniform}[0,2\pi)),
- apply a brief OU-like input “bump” centered at (\theta_0),
- run a delay period with noise (diffusion along the ring),
- optionally add a second cue on some fraction of trials to test updating.

OU-driven finite pulses are a very defensible, publishable choice because they match how other authors create controlled, smooth stochastic inputs.

## 5) A concrete Phase 3 dataset plan that is consistent with your proposal _and_ aligned with the literature

I recommend you generate **three trial families** (all from the same teacher ring network):

### A) Noise-only “exploration” trials (identifiability benchmark)

- No cue; just background noise for (T).
- Purpose: does the student invent discretization under dropout/time truncation?
- This aligns with the OU/noise-driven philosophy in Qian et al.

### B) Single-cue “memory” trials (the canonical ring test)

- Pre-cue baseline (short).
- Cue: a brief localized input that sets bump at (\theta_0).
- Delay: bump persists; noise produces drift/diffusion along the ring.
- Purpose: does the student reproduce _continuous_ ring state evolution, or does it collapse to a few stable angles?

### C) Held-out perturbation/update trials (mechanistic validation)

Pick one (or both):

- **distractor pulse** during delay (input centered at (\theta_d)), or
- **second cue** that should update the bump.
  Train on A+B; test on C (mirrors the “train on clean, test on perturbations” logic).

### Coverage rule for (\theta)

Do **not** constrain “subsequent cues must be near.” If your goal is ring-ness, you actually want broad coverage:

- sample (\theta_0) uniformly across the ring,
- for two-cue trials sample (\theta_2) independently (uniform) on a subset of trials.
  Then you can later stratify performance by (|\Delta\theta|) as an analysis.

This avoids baking in an assumption about smooth sequential motion unless your teacher explicitly includes a velocity input.

## 6) What to report as “ring simulation quality” (what the papers actually use)

If you want to match how these papers justify their simulations, you should report **at least**:

1. **Predictive fit** on observed neurons (what your proposal calls “how well data from the RNN simulates original data”).
2. **Mechanistic tests** on the trained student:
   - fixed point search from many initial conditions (ring should look like a continuum / manifold, not a few points),
   - perturbation response (does it return along a low-dimensional manifold? does it show diffusion?),
   - spectral/time-constant structure (watch for spurious big gaps that signal “fake attractor” explanations).

3. Sensitivity to **subsampling fraction** and **observation window length (T)** — the two knobs your proposal explicitly cares about.

That trio is fully aligned with the “mechanistic mismatch” evaluation philosophy (fit can be excellent while mechanism is wrong).

---

If you want, I can turn the above into a Phase 3 “spec” (exact trial timings, number of trials, train/val/test splits, and the grid over neuron-dropout × time-window), but the key decision you need to make first is this:

**Do you want the student to learn (i) ring _maintenance only_, or (ii) ring maintenance + _controlled movement_ along the ring via an explicit velocity input?**

Your current proposal text is compatible with (i); (ii) requires adding a motion-generating input channel and will change what “good dataset coverage” means.
