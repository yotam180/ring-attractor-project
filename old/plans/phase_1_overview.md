Below is how to think about **Step 1 (“teacher simulation”)** in your proposal: you need a **ground-truth ring attractor network** that (a) produces a moving/maintainable “bump” of activity on a circle (a continuous variable (\theta)), and (b) can emit **spiking data** that you will later partially observe, bin/smooth, and train an RNN on. This is exactly what your proposal calls for: simulate a ring attractor and generate spiking data, then later bucket/smooth with a known (dt).

---

## 1) What you are simulating (conceptually)

A **ring attractor** is a continuous attractor: the network has (approximately) a _continuum_ of equally valid stable states parameterized by an angle (\theta \in [0,2\pi)). In practice that means:

- The population forms a **localized bump** of activity.
- The bump’s **location** around the ring encodes (\theta).
- With perfect symmetry, the bump can sit anywhere (neutral stability along the ring); with noise it **diffuses** slowly; with inputs it **moves**.
- With symmetry-breaking / heterogeneity / insufficient recurrence, the “continuous” attractor can collapse into **discrete fixed points** (“a discretized ring”), exactly the failure mode your proposal wants to detect downstream.

---

## 2) The math: a standard “vanilla” ring model (rate network)

### State variable

- (r(t) \in \mathbb{R}^N): firing **rates** (one per neuron)
- Each neuron has a preferred angle (\theta_i = 2\pi i/N)

### Dynamics (continuous time)

A common baseline is:
[
\tau \frac{dr}{dt} = -r + \phi!\left(W r + I_{\text{ext}}(t) + \sigma,\xi(t)\right)
]

- (\tau): neural time constant
- (\phi(\cdot)): static nonlinearity (ReLU, softplus, tanh, etc.)
- (\xi(t)): white noise (Gaussian), scaled by (\sigma)
- (W\in\mathbb{R}^{N\times N}): recurrent connectivity with **circular symmetry**

### Ring connectivity (“cosine” kernel)

A canonical choice:
[
W_{ij} = J_0 + J_1 \cos(\theta_i-\theta_j)
]
Interpretation:

- (J_1>0): local excitation + distal inhibition structure (effectively “Mexican-hat-ish” on the circle). This is what supports a bump.
- (J_0<0): global inhibition / normalization that prevents runaway activity and encourages a single bump.

**Equivalent implementation trick:** you do _not_ need to build full (W) if (W) is circulant; you can compute (Wr) via circular convolution (FFT). But for first-pass clarity, building (W) explicitly is fine at moderate (N).

### What you measure as “the encoded (\theta)”

A standard readout of bump location:
[
z(t)=\sum_{i=1}^N r_i(t),e^{i\theta_i},\quad
\hat{\theta}(t)=\arg(z(t))
]
This gives you a continuous estimate of the bump’s center.

---

## 3) How to generate spiking data from the rate model

Your proposal explicitly wants “spiking data” from the teacher.
The simplest (and standard for synthetic benchmarks) is an **inhomogeneous Poisson** observation model:

For each neuron (i) at timestep (t_k) with bin width (dt):

- spike count:
  [
  s_i[k] \sim \text{Poisson}(r_i[k];dt)
  ]
  This produces integer spike counts per bin (0,1,2,…). If (dt) is small and rates are reasonable, it’s mostly 0/1.

Then Step 3 of your pipeline (“smooth into time buckets of known (dt)”) is consistent with standard practice in the literature: bin spikes then smooth to get PSTH-like rates.

(Separately, real-data papers do similar bin+smooth operations; e.g., spike rates computed in small bins and smoothed with a causal boxcar filter. )

---

## 4) What you run: a concrete simulation loop

### Data types (practical)

- **Core state arrays**: `numpy.ndarray` of shape `(N,)` for `r`, `(N,N)` for `W` if explicit.
- Use `float32` unless you see numerical issues (saves memory, faster).
- Spikes: `int16` or `int32` counts array `(T, N)`; or store sparse events if (N,T) large.
- Inputs: `(T, N)` if precomputed; or computed on the fly.

### Libraries (sane defaults)

- `numpy`: arrays, RNG
- `scipy` (optional): FFT-based circular convolution, signal filters
- `matplotlib`: sanity plots (bump shape, (\hat{\theta}(t)), raster)
- `numba` (optional): accelerate loop if pure NumPy gets slow
- If you already plan to train the student in PyTorch later, you _can_ implement teacher in PyTorch too—but for Step 1, NumPy is cleaner and less error-prone.

### Integration method

Use forward Euler to start:
[
r_{k+1} = r_k + \frac{dt}{\tau}\left(-r_k + \phi(\cdot)\right)
]
It’s fine for a first step. If you see instability, reduce `dt` or switch to Heun/RK2.

### Minimal experiment structure

You typically want at least these conditions:

1. **Initialize** a bump at a chosen (\theta_0) (either by initial condition or a brief external input “cue”).
2. **Delay / maintenance**: remove cue; bump persists and may diffuse under noise.
3. **Perturb / drive**: apply a transient input that pushes the bump; observe movement.

That maps cleanly onto your project’s later mechanistic checks (fixed points, perturbation responses).

---

## 5) Parameters: what they mean and what happens when you tweak them

Below is the set you actually need to control tightly in Step 1.

### A) Network size and discretization

- **(N)** (neurons around ring): higher (N) = smoother bump, better approximation of continuous manifold, but more compute.
- Too small (N) can itself create “discretized” behavior (a purely numerical artifact you don’t want confounding your results).

### B) Time constants and timestep

- **(\tau)**: sets intrinsic timescale of rate relaxation.
- **(dt)**: integration resolution; must be small relative to (\tau) (rule of thumb: (dt \le \tau/20) to start).
- Your later spike binning (dt\_{\text{bin}}) (for “time buckets”) is a _separate_ choice; don’t confuse integration `dt` with observation bin width. Your proposal distinguishes this bucketing step explicitly.

### C) Recurrent strength: bump existence and stability

- **(J_1)** (structured cosine component):
  - Increase (J_1): bump becomes sharper and more stable (up to saturation / multi-bump regimes depending on (\phi) and inhibition).
  - Decrease (J_1): bump broadens, then disappears → homogeneous state.

- **(J_0)** (global offset / inhibition):
  - More negative (J_0): suppresses global activity, encourages single bump.
  - Too negative: quashes activity entirely.

### D) Nonlinearity (\phi)

- **ReLU**: simple, but can create hard-threshold effects.
- **tanh**: saturating; helps bound activity and can prevent explosions (relevant because saturating nonlinearities change stability behavior compared to linear systems).
- **gain / slope** of (\phi): higher gain can create sharper attractors (and more abrupt transitions).

### E) Noise level (\sigma): drift/diffusion along the ring

- (\sigma = 0): bump sits (nearly) fixed unless you drive it.
- Moderate (\sigma): bump location performs a slow random walk (diffusion), which is often the most realistic “continuous attractor under noise” regime.
- Too large (\sigma): bump dissolves or jumps erratically.

### F) External input (I\_{\text{ext}}(t)): how you control (\theta(t))

A typical cue input is also a cosine/von-Mises bump:
[
I_{\text{cue},i}(t) = A(t)\cos(\theta_i-\theta_{\text{target}})
]

- **Amplitude (A)**: bigger cue pins the bump strongly to (\theta\_{\text{target}}).
- **Duration**: brief cue then off tests true attractor maintenance.
- **Bias input** can intentionally break symmetry to induce discrete fixed points (useful as a controlled “teacher with discretization” condition).

---

## 6) What artifacts you should produce at the end of Step 1 (non-negotiable)

If you finish Step 1 correctly, you should have:

1. **Ground-truth latent trajectory** (\theta(t)) (either commanded by inputs or inferred from the bump).
2. **Full-population spike trains** (s_i[k]) for all (N) neurons.
3. A “teacher dataset object” you can later sub-sample:
   - `spikes: (T, N)` counts
   - `rates: (T, N)` optional (teacher internal)
   - `theta_hat: (T,)`
   - metadata: parameters used, seeds, `dt_int`, `dt_bin`

This sets you up directly for Step 2 (“dropout / partial observation”) and Step 3 (“smooth into buckets”).

---

## 7) Where to read in your attached papers (for the _logic_ of this step)

Even though your ring model is a standard construction, here’s how the attached readings support the _pipeline rationale_ and the failure modes you’re testing:

- **Your proposal’s success criterion** for “did we recover a ring?” is explicitly about whether the bump moves through (\theta) continuously vs settling into discrete points—use that as your operational definition throughout Step 1 and Step 5.
- **Mechanistic mismatch under partial observation**: the core warning is that models can fit observed activity while misidentifying the underlying dynamical mechanism; this is the motivation for carefully controlled teacher simulations and later perturbation-based validation.
- **Dynamical systems reconstruction framing**: trained RNNs are used as surrogate dynamical systems, but interpretability and faithfulness depend on the reconstruction setup—this supports why you start with a known teacher DS.
- **Empirical preprocessing precedent** (useful when you later implement Step 3): spike rates computed in short bins and smoothed with a boxcar filter is a common workflow.

---

## 8) Strong recommendation for your next action

Implement Step 1 in this strict order:

1. **Rate-only ring attractor** (no spikes yet): verify a bump exists, persists, and its decoded (\hat{\theta}(t)) behaves as expected under cue/noise.
2. Add **Poisson spiking** observation on top.
3. Save one clean dataset with **full observability**, with parameter metadata, and a few sanity plots.

If you skip (1), you will waste time later debugging the student training when the teacher never had a real continuous attractor to begin with—exactly the kind of silent failure your proposal warns about.
