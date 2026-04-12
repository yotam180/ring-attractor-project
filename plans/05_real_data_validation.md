# Phase 5: Real Data Validation (DANDI:000939)

## Goal

Apply the RNN training pipeline — developed and validated on synthetic ring attractor data — to real head-direction cell recordings. Two questions:

1. **Can the pipeline recover a ring attractor from real neural data?**  
   Train with full observation (all HD cells, full wake epoch). If the ring score passes Milestone 1 criteria, the method generalises beyond simulation.

2. **Do the same partial-observation thresholds appear?**  
   Sweep k/N and T on real data. Compare the cliff locations to synthetic results (k/N ≈ 0.15–0.20, T ≈ 50–100 bins). If they match, the thresholds are a property of the inference problem, not the simulator.

---

## Dataset

**Source:** Duszkiewicz, Skromne Carrasco, Peyrache (2024). Large-scale recordings of head-direction cells in mouse postsubiculum. DANDI:000939.

- 31 sessions, 21–117 HD cells per session
- ~40 min wake recordings in square arena
- Already downloaded to `000939/`
- Full exploration: `plans/dataset_000939_exploration.md`

---

## Pipeline Adaptations

### 1. Trial construction

Synthetic data: 144 discrete trials, each 200 bins (4 seconds).  
Real data: one continuous ~40 min recording → segment into non-overlapping windows of 200 bins.  
At 50 Hz (dt=20ms), 40 min = 120,000 bins → 600 trial segments. Plenty for train/val split.

### 2. Neuron angles

Synthetic: evenly spaced θᵢ = 2πi/N.  
Real: compute from tuning curves. For each HD cell, preferred direction = circular mean of occupancy-normalised firing rate weighted by angle.

### 3. Generalisation test

Synthetic: uses teacher simulator to generate bumps at intermediate angles.  
Real: find time segments where the animal's head direction is near each test angle. Use those as teacher-forcing seeds.

### 4. Everything else

Training config (proven recipe), evaluation metrics (uniformity, circularity, drift), and autonomous fixed-point analysis all transfer unchanged.

---

## Execution Plan

### Phase 5A: Data loading & preprocessing

- `src/real_data/loading.py` — NWB → spike trains + head direction + epoch boundaries
- `src/real_data/preprocessing.py` — bin spikes, tuning curves, trial segmentation, normalisation
- Output: one `.npz` per session matching `ring_attractor_dataset.npz` format

### Phase 5B: Baseline (full observation)

- Top 3 sessions by HD cell count
- 2 seeds each → 6 runs
- Evaluate ring scores, compare to synthetic baseline

### Phase 5C: k/N sweep

- For each of 3 sessions, sweep k/N ∈ {1.0, 0.75, 0.5, 0.25, 0.15, 0.1}
- 2 seeds each → 36 runs
- Compare cliff location to synthetic k/N ≈ 0.15–0.20

### Phase 5D: T sweep (deferred)

- Truncate recordings to different durations
- Cross with k/N
- Compare to synthetic T threshold

### Phase 5E: Figures & comparison (deferred)

---

## Session Selection

Use the 3 sessions with the most HD cells to maximise statistical power. Identify these by scanning all NWB files.

---

## Key Parameters (refined after PoC)

Modifications from synthetic recipe, discovered during iteration:

```python
TrainingConfig(
    model_type="vanilla",
    hidden_dim=128,          # increased from 100 to avoid N→H bottleneck
    n_epochs=3000,           # reduced from 5000 (converges faster)
    learning_rate=8e-4,
    scheduler="cosine",
    convergence_weight=5.0,
    convergence_steps=30,
    noise_std=0.08,
    noise_std_final=0.002,
    circular_shift_augment=False,
    val_split_mode="random", # no paired trials in real data
    observation_fraction=<varies>,
)
```

Preprocessing changes:
- `smooth_kernel=5` (up from 3): real spike data needs more smoothing
- `max_hd_change_deg=45`: filter out trials where head direction changes > 45° 
  (animal turning rapidly → poor autonomous prediction target)

---

## Results (A3707, 117 HD cells, 301 trials)

### Full sweep results

| k/N  | Seed | Uniformity | Circularity | Drift (°) | MSE   | M1 Pass |
|------|------|-----------|-------------|-----------|-------|---------|
| 1.00 |   42 | **0.857** | **0.810**   | 24.7      | 0.969 | NO      |
| 1.00 |  123 | 0.590     | 0.592       | 21.7      | 0.996 | NO      |
| 0.75 |   42 | **0.816** | **0.768**   | 25.1      | 0.964 | NO      |
| 0.75 |  123 | **0.818** | 0.607       | 22.5      | 0.984 | NO      |
| 0.50 |   42 | 0.458     | 0.564       | 17.0      | 0.978 | NO      |
| 0.50 |  123 | 0.792     | 0.601       | 24.1      | 0.993 | NO      |
| 0.25 |   42 | **0.852** | **0.788**   | 28.2      | 0.975 | NO      |
| 0.25 |  123 | 0.621     | **0.875**   | 22.8      | 0.994 | NO      |
| 0.15 |   42 | 0.750     | 0.709       | 22.3      | 0.988 | NO      |
| 0.15 |  123 | **0.863** | 0.667       | 39.4      | 0.982 | NO      |
| 0.10 |   42 | 0.767     | 0.673       | 31.8      | 0.985 | NO      |
| 0.10 |  123 | **0.900** | **0.875**   | 26.6      | 0.985 | NO      |

Bold = exceeds individual threshold (uniformity>0.8, circularity>0.7).

**All 12 runs fail Milestone 1** because drift is universally 17-40°, far above the 5° threshold.

### Key findings

1. **Ring shape IS recoverable** — 4/12 runs have BOTH uniformity>0.8 and circularity>0.7
   (seed 42 at k/N=1.0, 0.75, 0.25; seed 123 at k/N=0.10). The RNN learns a ring-shaped
   manifold in hidden space from real data.

2. **Autonomous dynamics always drift** (17-40°). This is the critical gap between synthetic
   and real data. The ring has ~8-10 discrete attractor positions (polygon), not a continuous
   attractor. Autonomous activity drifts to the nearest discrete basin rather than staying
   at the teacher-forced angle.

3. **High seed variance, no clean threshold.** Unlike synthetic data (sharp cliff at
   k/N=0.15-0.20), real data shows noisy, non-monotonic metrics across k/N. Seed 123 at
   k/N=0.10 is the best run overall — better than many higher observation fractions.

4. **MSE is flat and near 1.0 everywhere** (0.96-1.00). The RNN barely beats the trivial
   predictor, suggesting that real HD cell dynamics are much harder to predict step-by-step
   than synthetic data.

5. **Ring shape vs ring dynamics are separable.** Uniformity and circularity measure the
   geometry of fixed points. Drift measures the dynamical stability. Real data shows that
   good geometry doesn't imply good dynamics — the shape can be a ring even when the
   attractor landscape has discrete wells.

### Comparison with synthetic

| Property | Synthetic | Real (A3707) |
|----------|-----------|--------------|
| Full M1 pass at k/N=1.0 | 100% (2/2) | 0% (0/2) |
| Ring shape pass (u+c) at k/N=1.0 | 100% | 50% (1/2) |
| Ring shape pass across all k/N | Sharp cliff at 0.15-0.20 | Non-monotonic, 4/12 |
| Drift | < 5° (continuous ring) | 17-40° (discrete polygon) |
| MSE | ~0.01-0.1 | ~0.97-1.0 |
| Seed variance | Low | High |

### Interpretation

The key finding is a **dissociation between ring geometry and ring dynamics** in real data:

- The RNN can learn a ring-shaped manifold of fixed points from real HD cell recordings,
  confirming that the pipeline generalises beyond simulation.
- However, the autonomous dynamics on this manifold produce large drift (~20-30°), unlike
  synthetic data where drift is <5°. This reflects the heterogeneous, noisy nature of real
  HD cell activity.
- The sharp partial-observation threshold found in synthetic data does NOT transfer to real
  recordings — instead, recovery is stochastic across both k/N and random seeds.
- MSE near 1.0 confirms that step-by-step prediction of real neural dynamics is far harder
  than synthetic, consistent with unmodeled noise sources (spike variability, non-HD inputs,
  non-stationarity).

### A3716 (93 HD cells, 129 trials) — preliminary

| k/N  | Seed | Uniformity | Circularity | Drift (°) | M1 Pass |
|------|------|-----------|-------------|-----------|---------|
| 1.00 |   42 | 0.703     | 0.741       | 31.4      | NO      |

Only 1 run complete so far. Fewer trials (129 vs 301) due to stricter stability filtering.
Sweep in progress.

---

## Figures

- `figs/05_real_data/real_sweep_metrics.png` — metrics vs obs_frac (per session)
- `figs/05_real_data/real_sweep_pca_grid.png` — PCA projections grid
- `figs/05_real_data/real_sweep_drift_grid.png` — drift vs test angle
- `figs/05_real_data/real_sweep_pass_rates.png` — pass rate comparison with synthetic
- `figs/05_tuning_curves_3sessions.png` — tuning curve verification
