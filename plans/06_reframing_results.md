# 06 — Reframing the Results

**Date:** 2026-04-10
**Status:** Conclusions from paper-writing session — use to build Results section

---

## 1. The Original Hypothesis That Didn't Hold

H2 from the project proposal predicted:

> "Below some threshold in the number of neurons, the RNN's predictive accuracy (MSE) stays high, although the dynamical class reproduction ability will be decreased."

This was inspired by Qian et al. (2024), who showed that partial observation can induce mechanistic misidentification even when the student fits well.

**Our synthetic data does not show this.** On the k/N axis, MSE and structural metrics (drift, uniformity) degrade at the same threshold (~k/N = 0.15–0.20). There is no regime where the model fits well but learns the wrong dynamics. Both break together.

**Why:** Our teacher has a symmetric (normal) cosine kernel. Qian et al.'s result relies on non-normal teacher connectivity. With a normal matrix, partial observation creates a straightforward information bottleneck — the student can't reconstruct 100 neurons from 10, so both prediction and mechanism suffer.

---

## 2. What We Actually Found

### 2.1 Two axes, two different failure modes

The project's real contribution is characterizing **how** ring attractor recovery fails along two independent observation axes, revealing qualitatively different failure modes:

**k/N axis (spatial undersampling) — global failure:**
- Sharp cliff between k/N = 0.15 and 0.25.
- Below the cliff, everything breaks: MSE rises ~3×, drift exceeds threshold, uniformity becomes unreliable.
- The failure is unsurprising in hindsight: with 10–15 neurons out of 100, the teacher-forcing phase provides too little information for the student to infer the full population state.
- **Failure mode:** the model can't reconstruct the data, so it can't learn the dynamics either.

**T axis (temporal undersampling) — mechanistic misidentification:**
- Sharp cliff between T = 50 and T = 100 time bins.
- Below the cliff, the ring *geometry* is preserved (uniformity 0.90–0.96, circularity 0.81–0.96) but the *dynamics* are wrong (drift 6–14°).
- The student learns a ring-shaped manifold with discrete attracting regions at the 72 training angles — a polygon, not a continuous ring.
- **Failure mode:** with ~30 autonomous steps per trial, the model sees enough bump shapes to learn the geometry but not enough temporal evolution to learn neutral stability.

This distinction — spatial undersampling causes global failure while temporal undersampling causes selective mechanistic misidentification — is the central finding.

### 2.2 The threshold is sharp, not gradual

On both axes, the transition from success to failure is cliff-like:
- k/N axis: 100% pass at 0.20, dropping to 50% at 0.15 and 40% at 0.10.
- T axis: 100% pass at T = 100, 0% pass at T = 50.

There is no regime of graceful degradation. The system either recovers the ring or it doesn't.

### 2.3 MSE is not a reliable proxy for mechanistic correctness

While the original H2 (systematic decoupling) didn't hold, there is a weaker but real finding: within any given observation regime, MSE does not predict pass/fail on a per-seed basis. At k/N = 0.10:

| Seed | MSE | Drift | Pass? |
|------|-----|-------|-------|
| 7 | 0.061 (best) | 6.20° | FAIL |
| 99 | 0.085 | 4.02° | PASS |
| 123 | 0.122 (worst) | 3.63° | PASS |

The seed with the lowest MSE fails; the seed with the highest MSE passes. This means: even within the degraded regime, reconstruction quality does not tell you whether the model learned the right mechanism. Structural evaluation (fixed-point analysis, drift tests) remains necessary.

---

## 3. Revised Paper Framing

**Old thesis (don't use):** "RNN-based DSR of a ring attractor exhibits low MSE but wrong mechanism below observation thresholds."

**New thesis:** "We characterize the observation requirements for RNN-based ring attractor recovery and find sharp thresholds on both the neuron count (k/N) and recording duration (T) axes. The two axes produce qualitatively different failure modes: spatial undersampling causes global reconstruction failure, while temporal undersampling causes selective mechanistic misidentification — the model learns the correct attractor geometry but with discrete rather than continuous dynamics."

### What this means for the Results section structure

**3.1 Baseline** — full observation recovers the ring. Validates the pipeline.

**3.2 k/N axis** — sharp cliff at 0.15–0.25. Present drift as the primary metric (cleanest cliff). Note that MSE co-degrades — this is NOT the "low MSE, wrong mechanism" story. Frame it as: below a spatial sampling threshold, the information bottleneck is too severe for the student to learn anything correctly.

**3.3 T axis** — sharp cliff at 50–100 bins. Here the failure mode IS mechanistic misidentification: geometry preserved, dynamics wrong. This is the more interesting result. Show the contrast: uniformity/circularity pass while drift fails. If possible, show a drift-vs-angle plot comparing T = 100 (flat, continuous ring) vs T = 50 (sawtooth, polygon).

**3.4 MSE is necessary but not sufficient** — reframe the predictive/mechanistic relationship honestly. MSE correlates with mechanistic quality across conditions (both degrade at the same threshold) but does not predict it within conditions (per-seed decorrelation at k/N = 0.10). Structural evaluation is always necessary.

---

## 4. Comparison to Qian et al.

Our results complement rather than replicate Qian et al. (2024):

| | Qian et al. | This work |
|---|---|---|
| Teacher topology | Line attractor | Ring attractor (S¹) |
| Teacher connectivity | Non-normal | Normal (symmetric cosine kernel) |
| Observation axis | Neuron count only | Neuron count AND time |
| Key failure mode | Spurious attractors from non-normality | Discrete polygon from temporal undersampling |
| MSE/mechanism decoupling | Yes (strong, systematic) | No on k/N axis; partial on T axis |

The difference in MSE/mechanism decoupling likely traces to the teacher's connectivity structure. Non-normal matrices have directions that are amplified transiently but stable long-term — partial observation can miss these transient directions, causing the student to learn spurious structure while still fitting the observed data. Our symmetric cosine kernel doesn't have this property.

This is itself a finding worth noting: **the severity of mechanistic misidentification under partial observation depends on the teacher's connectivity structure, not just on the observation fraction.**

---

## 5. Figures Needed for the Revised Results

| Figure | Purpose | Status |
|---|---|---|
| Drift + MSE vs k/N (dual panel) | Show co-degradation on k/N axis | Generated (paper_drift_vs_mse.png), will improve after more-seeds sweep |
| PCA at selected k/N values | Visual evidence of ring collapse | Generated (paper_pca_selected.png) |
| T-axis: uniformity/circularity vs drift | Show geometry-preserved-but-dynamics-wrong | Needs creation — key figure for the new framing |
| Drift-vs-angle at T=100 vs T=50 | Show continuous ring vs polygon (sawtooth pattern) | Data exists in sweep_results_T/, needs figure |
| MSE vs drift scatter (per-seed, k/N=0.10) | Show per-seed decorrelation | Needs creation — small panel or inset |

---

## 6. Open Questions

1. **More seeds will help.** 2 seeds per condition is thin. The 8-seed sweep (run_sweep_moreseeds.py) will give proper error bars and more convincing pass rates. Run this before finalizing figures.

2. **Should we redefine the pass criteria?** The uniformity threshold at 0.8 is arbitrary and the baseline (k/N = 1.0) barely clears it at 0.844. Consider either (a) lowering the threshold, (b) dropping uniformity as a pass criterion and using only drift + circularity, or (c) presenting results without binary pass/fail and instead showing continuous metric values with the threshold as a reference line.

3. **The k/N = 0.75 seed 123 failure** (uniformity 0.736, everything else fine) is a seed artifact. With more seeds we can confirm this is noise, not a real threshold.

4. **Do we want a 2D phase diagram (k/N × T)?** The T-sweep only covers 2 k/N values (1.0 and 0.25) × 4 T values. A fuller grid would be compelling but expensive to run.
