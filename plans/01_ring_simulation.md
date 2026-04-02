# 01 — Ring Attractor Simulator: Build Log

**Date:** 2026-04-02
**Status:** Complete — simulator validated, ready for dataset generation

---

## Goal

Build a ring attractor simulator that **actually works**: given noise, a bump forms; given a bump, it persists indefinitely. The old simulator (in `old/src/`) used ReLU and the bump dissolved over time.

---

## What We Built

New `src/` directory with 4 files:

| File | Purpose |
|---|---|
| `src/simulator.py` | Core simulator: weights, dynamics, decoding, spike pipeline |
| `src/validate_simulator.py` | 5 automated tests proving attractor properties |
| `src/visualize_ring_attractor.py` | 6-figure visualization suite (saves to `figs/`) |
| `src/demo_ring_attractor.py` | Notebook-ready demo (cell markers with `# %%`) |

Plus supporting documents:

| File | Purpose |
|---|---|
| `plans/relu_instability_proof.md` | Analytical proof that ReLU fails |
| `figs/01–06_*.png` | Visualization figures |

---

## Iteration History

### Attempt 1: ReLU (original parameters from revised_training_strategy.md)

**Parameters:** N=100, J0=−2, J1=4, φ=ReLU, dt=0.01, τ=100 (α=dt/τ=0.0001)

**Result:** All 5 validation tests FAILED.

The bump formed during the cue phase but **decayed to zero** during free-running. Even without noise (σ=0), the peak rate dropped from 0.86 to 0.001 over 100k steps.

**Diagnosis:** The old code had this exact same problem. The "confidence metric fix" (normalizing by total activity) made the display look better but masked the underlying dynamics bug — the bump was genuinely dissolving.

**Root cause (proven analytically in `plans/relu_instability_proof.md`):**

The cosine kernel W with /N normalization has eigenvalues:
- Uniform mode: J0 = −2
- Cos/sin modes: J1/2 = 2
- All others: 0

Near r=0: eigenvalue 2 > 1 → bumps try to grow (zero state is unstable).

But at finite amplitude, a half-cosine bump has effective gain:

$$G = \frac{J_0}{\pi} + \frac{J_1}{4} = \frac{-2}{\pi} + 1 = 0.363$$

Since G < 1, the self-consistency equation A = G·A has only the solution A = 0. **No nonzero equilibrium exists.**

This was proven for arbitrary bump width α: the self-consistency function F(α) < 0 for all α ∈ (0, π], not just for the half-cosine case.

The critical J1 for a half-cosine equilibrium would be J1_crit = 4(1 − J0/π) ≈ 6.55, but even then the equilibrium is marginally stable (any amplitude is a fixed point — no restoring force).

**The fundamental issue:** ReLU has constant gain (derivative = 1) in the active region. A piecewise-linear system can't have a unique, stable, finite-amplitude equilibrium — the amplitude either decays, diverges, or drifts neutrally.

### Attempt 2: Plain tanh

**Parameters:** Same as above but φ=tanh, α=0.01 (increased from 0.0001 for faster convergence)

**Result:** All 5 tests PASSED. Bump forms, persists for 100k+ steps, all 36 angles work.

**Problem discovered:** The bump was a **wide plateau**, not a sharp peak.
- FWHM = 151° (42% of neurons above half-max)
- Confidence = 0.73
- At 50° from peak, rate was still 88% of peak
- 50/100 neurons active (the entire hemisphere)

**Diagnosis:** The equilibrium peak input to tanh was ~2.3, deep in the saturation regime (tanh(2.3) = 0.98). Since tanh is nearly flat above x ≈ 1.5, all neurons with input > 1.5 had essentially the same output rate. The bump looked like a mesa/cliff, not a peaked hill.

**Why the kernel shape doesn't help:** We tested von Mises (κ=1–16) and Gaussian (σ=0.3–0.8) kernels. Result: with the same effective eigenvalue (~1.4), all kernels gave similar FWHM (~140°). The bump width is dominated by tanh saturation, not kernel shape.

**Why adjusting J0/J1 doesn't help:** Increasing coupling strength drives more neurons deeper into saturation → wider plateau. E.g., J0=−20, J1=30 gave FWHM=176°. Decreasing coupling brings the eigenvalue below 1 → no bump at all.

### Attempt 3: tanh(steepness × ReLU(x))  ← Final version

**Parameters:** φ(x) = tanh(4 · max(0, x)), α=0.01, cue_amplitude=3.0

**Result:** All 5 tests PASSED with much better metrics.

| Metric | Plain tanh | tanh(4·ReLU) |
|---|---|---|
| FWHM | 151° | 112° |
| Confidence | 0.73 | 0.84 |
| Active neurons | 50/100 | 31/100 |
| Silent neurons | 0 (all slightly active) | ~50 (exactly zero) |
| Peak rate | 0.98 | 0.84 |
| Angle error (36 angles) | 0.2° | 1.3° |
| Long-term stability | Stable (ratio 1.00) | Stable (ratio 1.00) |

**Why this works:** The nonlinearity combines:
1. **Hard threshold at 0** — neurons with negative recurrent input are completely silent, not just attenuated. This eliminates the "long tail" of weakly active neurons that widened the plain-tanh bump.
2. **Steep transition** — steepness=4 means the input-to-rate curve rises sharply near 0, concentrating the active population.
3. **Saturation at 1** — tanh caps the output, preventing amplitude explosion (the ReLU failure mode).

Biologically, this is a standard **f-I curve** (firing rate vs. input current): below threshold → silent; above threshold → rate increases then saturates. This is more realistic than either plain ReLU (no saturation) or plain tanh (no threshold, negative "rates").

**Steepness tradeoff:** We scanned steepness ∈ {2, 3, 4, 5, 6, 8, 10}:
- s=3: FWHM=76°, conf=0.91, but peak rate only 0.21 (weak signal)
- s=4: FWHM=90°, conf=0.90 (best balance) — used as default
- s=5: FWHM=94°, conf=0.89, peak=0.95
- s≥6: approaches the plain-tanh regime (wider bumps)

(Note: with noise σ=0.1 and shorter cue, FWHM increases to ~112° in practice.)

---

## Other Parameter Changes from Revised Training Strategy

| Parameter | Revised strategy | What we use | Why |
|---|---|---|---|
| Nonlinearity | ReLU | tanh(4·ReLU) | ReLU has no stable equilibrium (see proof) |
| α (= dt/τ) | 0.0001 (dt=0.01, τ=100) | 0.01 | Original was 100× too slow; bump barely formed in 2000 cue steps |
| Cue amplitude | 2.0 | 3.0 | Compensates for the threshold in the nonlinearity |
| Cue duration | 2000 steps | 2000 steps | Unchanged |

The weight matrix (cosine kernel, J0=−2, J1=4, /N normalization) and all spike processing parameters (bin_factor=50, smoothing_window=3, rate_scale=100) are unchanged.

---

## Validation Results (Final)

| Test | Description | Key metrics | Result |
|---|---|---|---|
| 1. Bump formation | Cue at 60° → free-run | conf=0.84, err=0.9°, FWHM=112° | PASS |
| 2. Long-term maintenance | 100k steps after cue removal | conf=0.84, peak ratio=1.00 | PASS |
| 3. Spontaneous from noise | No cue, just σ=0.1 noise | conf=0.84 (late phase) | PASS |
| 4. Perturbation recovery | Large noise added to clean bump | conf: 0.46→0.84 | PASS |
| 5. Multiple angles | 36 evenly-spaced cues | mean err=1.3°, conf=0.84 | PASS |

---

## Metrics Explained

**Confidence R:** The population vector magnitude, normalized by total activity.

$$R = \frac{|\sum_i r_i e^{i\theta_i}|}{\sum_i r_i}$$

R=1 means all activity at one angle (delta bump). R=0 means uniform activity (no bump). R≈0.84 is a peaked bump covering ~25% of the ring.

**Peak rate ratio:** Mean peak firing rate in the last 10k steps divided by mean peak in the first 10k steps after cue. Ratio=1.00 means the bump amplitude is perfectly constant — no decay, no growth.

**FWHM (Full Width at Half Maximum):** The angular span of neurons firing above 50% of the peak rate. Lower = sharper bump. Our 112° means 31/100 neurons are in the "active core."

**Circular std of θ:** Measures how much the decoded angle wanders over time (noise-driven diffusion along the ring manifold). Small values (~1–4°) indicate the bump position is stable.

---

## Figures (in `figs/`)

1. `01_single_trial.png` — Heatmap + decoded θ/confidence + polar snapshots for one cued trial
2. `02_multi_angle.png` — 36 angles all accurately recovered (mean error 1.3°)
3. `03_long_term_stability.png` — 100k steps: peak rate, confidence, θ all stable
4. `04_noise_bump_formation.png` — Bump emerges spontaneously without any cue
5. `05_perturbation_recovery.png` — Large noise cleaned up, bump restored
6. `06_spike_pipeline.png` — Full rates → spikes → binned → smoothed pipeline

---

## Next Steps

1. **Dataset generation** — Use the simulator to produce Group A (bump maintenance) and Group B (noisy convergence) training data, following `plans/revised_training_strategy.md` Section 3.2.
2. **RNN training** — Autonomous student RNN with teacher-forcing initialization.
3. The revised training strategy should be updated to reflect: (a) the nonlinearity change (tanh(4·ReLU) instead of ReLU), (b) α=0.01 instead of dt/τ=0.0001, (c) cue_amplitude=3.0.
