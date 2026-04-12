# Phase 5B: Real Data Failure Analysis

## The Problem

We trained RNNs on real head-direction (HD) cell recordings from mouse postsubiculum (DANDI:000939, Duszkiewicz et al. 2024) using the same autonomous-forced learning pipeline that successfully recovered ring attractor dynamics from synthetic data. The pipeline:

1. Teacher-forces the RNN with real neural activity for K time steps
2. Cuts input to zero, lets the RNN run autonomously for T steps
3. Trains on reconstruction loss during both phases (emphasising the autonomous phase via convergence loss)
4. Evaluates: do the autonomous fixed points form a ring? (uniformity, circularity, drift metrics)

**Results across 27 runs (3 sessions, 6 observation fractions, 2 seeds):**
- **0/27 pass full Milestone 1** (uniformity > 0.8, circularity > 0.7, drift < 5°)
- 7/27 pass ring shape (uniformity + circularity) — the geometry is sometimes learned
- **Drift is 17–43° in every single run** — the trained RNN never holds a stable bump
- MSE ≈ 0.97–1.0 everywhere (barely beats trivial predictor)
- No clean observation fraction threshold (unlike synthetic data's sharp cliff at k/N ≈ 0.15–0.20)

## Root Cause: Stationary vs Moving Bumps

The fundamental mismatch is between what the training data contains and what the evaluation requires.

### Synthetic data (works)
- The synthetic ring attractor teacher produces **stationary bumps**: activity is initialised at an angle, and the autonomous dynamics maintain it there indefinitely (drift < 5°)
- During training, the RNN learns: "after teacher-forcing stops, hold this bump in place"
- During evaluation, the autonomous fixed points are stable → ring recovered

### Real data (fails)
- During wake exploration, the animal constantly turns its head, so the HD bump is **always moving**
- The bump movement is driven by **vestibular/motor efference copy inputs** that we do not observe and cannot feed to the RNN
- During training, the RNN sees neural activity where the bump drifts 20–100° over a 4-second trial — so it learns: "the bump should move"
- During autonomous evaluation, the RNN either (a) drifts to a discrete attractor basin (~20–30° away), or (b) produces chaotic wandering
- There are essentially **zero 4-second windows where the head is stationary** (< 5°/s angular velocity): only 8% of individual time bins are below 2°/s, and no contiguous 4-second segments exist below 10°/s

### The missing input problem
The biological HD circuit receives angular velocity input (from vestibular nuclei and motor efference copy) that drives the bump around the ring. Our RNN architecture has:
- **Input** = observed neural firing rates (during teacher-forcing phase)
- **No input** during autonomous phase

We have no way to provide angular velocity as a driving input, and even if we did, the "zero input = hold position" test requires that the RNN learned stationary dynamics — which it can't learn from data where the bump always moves.

## What the Data Looks Like

### Session A3707 (117 HD cells, best session)
- Wake epoch: 30 min, ~180k time bins at 100 Hz
- After stability filtering (max Δθ < 45° per 4s window): 301 trials
- But even "stable" trials have ~15–30° of head rotation
- Angular velocity distribution: mean ~45°/s, median ~35°/s
- Truly stationary periods (|ω| < 5°/s) cover only 17.5% of wake time, never forming a contiguous 4-second segment

### Sleep data available but unlabelled
- A3707 has 78 min of NREM sleep (61 bouts) and 0.6 min of REM
- During NREM, HD cells are known to spontaneously replay ring dynamics without vestibular drive (Peyrache et al. 2015)
- BUT: we have no head direction labels during sleep (tracking camera only works during wake)
- So we can't compute tuning curves or assign neuron angles for sleep data

## Questions for Literature Review

### 1. How do others train RNNs on neural data with unobserved inputs?

Our pipeline assumes the autonomous dynamics alone are sufficient to reveal the attractor. But real neural circuits receive external inputs that we don't record. How do papers like LFADS (Pandarinath et al. 2018), CEBRA (Schneider et al. 2023), or pi-VAE (Zhou & Wei 2020) handle unobserved inputs when training dynamical models on neural recordings?

Specific sub-questions:
- Does LFADS's "inferred input" (controller network) solve the problem of unobserved vestibular drive?
- Can a sequential VAE separate the autonomous manifold from the input-driven dynamics?
- What is the state of the art for inferring attractor structure from neural recordings where the system is being continuously driven by unobserved inputs?

### 2. How have people identified ring attractor dynamics in real HD cell data?

The dataset we're using (DANDI:000939) is from the Clark, Abbott & Sompolinsky (2025) paper. They don't train an RNN — they construct a weight matrix analytically from tuning curves. Other approaches:

- **Manifold/topology methods**: Chaudhuri et al. (2019) used PCA + persistent homology to show the ring manifold persists across wake and sleep in Drosophila HD cells. Gardner et al. (2022) used persistent cohomology on grid cells. Can topology-based methods (persistent homology, cohomological decoding) identify ring structure without training an RNN?
- **Direct weight fitting**: Clark et al. (2025) solve for J such that tuning curves are fixed points of τ dx/dt = -x + Jφ(x). This sidesteps the training problem entirely. Is this approach more appropriate than RNN training for real data?
- **Sleep replay analysis**: Peyrache et al. (2015) showed that HD population vectors during NREM sleep lie on the same ring manifold as during wake, without external drive. Can we use sleep data (where bumps are quasi-stationary) instead of wake data for our autonomous evaluation?

### 3. Can we add angular velocity as an explicit input to our RNN?

We have continuous head direction recordings at 100 Hz, so we can compute angular velocity dθ/dt directly. If we add this as a second input channel:
- During training: RNN receives both neural activity (teacher) and angular velocity
- During autonomous evaluation: set angular velocity to zero, give the RNN no neural input
- The RNN should learn: bump_position = f(recurrent_state, angular_velocity)
- At zero velocity input, the ring of fixed points should emerge

Questions:
- Has anyone done this for HD cells specifically?
- What's the right way to represent angular velocity as input to the RNN? (scalar dθ/dt? sine/cosine components? binned like population vector?)
- Does this change the scientific claim? (We'd be testing whether the RNN + known input can reveal the ring, not whether pure autonomous dynamics reveal it)

### 4. Can we use the sleep epochs for evaluation?

Key idea: train on wake data (with angular velocity input), but evaluate ring structure on NREM sleep data where:
- The bump is quasi-stationary or slowly drifting (~2–5°/s instead of 30–100°/s during wake)
- No external vestibular drive (the animal isn't moving)
- The ring manifold is known to persist (Peyrache et al. 2015)

Problems:
- No head direction labels during sleep → can't compute standard metrics
- Population vector decoding from wake tuning curves could assign angles to sleep activity
- Is this a valid approach? What are the pitfalls?

### 5. Would shorter time windows help?

Our current trial length is 200 bins × 20ms = 4 seconds. During 4s, the head rotates 15–100°. Could we:
- Use much shorter windows (e.g., 500ms = 25 bins) where the bump barely moves?
- Use overlapping windows instead of non-overlapping?
- What's the minimum window length for the convergence loss to be meaningful?

### 6. Is there a fundamentally different training objective we should use?

Instead of autonomous-forced learning (predict autonomous continuation), could we:
- Train on one-step prediction only (next-bin firing rate), then analyze the learned recurrent weights for ring structure?
- Use a Koopman operator / DMD approach on the neural data directly?
- Apply SINDy (sparse identification of nonlinear dynamics) to the population vectors?
- Use a switching linear dynamical system (SLDS) to separate stationary from moving-bump regimes?

## Summary of the Core Issue

```
SYNTHETIC (works):         REAL DATA (fails):
                           
Input: bump at angle θ     Input: neural activity (bump always moving)
  ↓                          ↓  
Teacher-force K steps      Teacher-force K steps
  ↓                          ↓
Zero input (autonomous)    Zero input (autonomous)
  ↓                          ↓
RNN holds bump at θ        RNN drifts to discrete basin (~20-30° away)
  ↓                          ↓
Fixed points form ring ✓   Fixed points form polygon, high drift ✗
                           
WHY: synthetic data has    WHY: real data has moving bumps, 
stationary bumps           driven by unobserved vestibular input
```

The RNN is being asked to maintain stationary dynamics from training data that is inherently non-stationary, and the driving input that causes the non-stationarity is unobserved.

## Our Dataset Details (for context)

- **Source**: DANDI:000939 — Duszkiewicz, Skromne Carrasco, Peyrache (2024)
- **Associated paper**: Clark, Abbott, Sompolinsky (2025) — "Symmetries and Continuous Attractors in Disordered Neural Circuits"
- **Brain region**: Mouse postsubiculum (dorsal presubiculum)
- **Sessions used**: A3707 (117 HD cells), A3716 (93 HD cells), A3711 (90 HD cells)
- **Recording**: Silicon probe, 64-channel, freely moving mice in square arena
- **Available data**: Spike times, head direction (wake only), position, NREM/REM intervals, LFP
- **RNN architecture**: Vanilla rate RNN, h[t] = (1-α)h[t-1] + α·tanh(W_hh·h[t-1] + W_xh·x[t] + b), H=128
- **Training**: 3000 epochs, cosine LR schedule, convergence weight=5.0, K=30 teacher steps + 170 autonomous steps per trial

## Potential Paths Forward (to investigate)

1. **Angular velocity input** — add dθ/dt as RNN input, evaluate at zero velocity
2. **Sleep training/evaluation** — use NREM epochs where bumps are quasi-stationary
3. **Manifold methods** — PCA + persistent homology directly on population vectors (no RNN)
4. **LFADS-style approach** — variational sequential model that infers unobserved inputs
5. **Analytical weight fitting** — follow Clark et al.'s approach (solve for J from tuning curves)
6. **Ultra-short windows** — 500ms trials where the bump barely moves
7. **Hybrid**: train with velocity input on wake data, evaluate autonomously on sleep data
