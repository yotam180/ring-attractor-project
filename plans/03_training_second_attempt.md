# 03 — Training Second Attempt: Changes, Goals, and Execution Plan

**Date:** 2026-04-03
**Status:** Milestone 1 PASS — ready for Milestone 2
**Prerequisite reading:** `plans/02_training.md` (training guide), `plans/02_training_execution.md` (first attempt results)

---

## 0. What this document is

This is the plan for the **second training attempt**. The first attempt (documented in `02_training_execution.md`) achieved Milestone 1 but relied on circular shift augmentation which is incompatible with partial observation. After a critical review, we made three code fixes, discovered the augmentation confound, increased training angles from 36 to 72, and verified the baseline passes without augmentation. See `03_training_execution.md` for the full experimental log.

---

## 1. What changed and why

### 1.1 Removed: Divisive normalization from the simulator

**What it was:** The teacher simulator had divisive normalization: `total_input / (1 + γ·mean(r))` with γ=2.

**Why we removed it:** The student RNN (`h[t] = (1-α)h[t-1] + α·tanh(W·h[t-1] + b)`) cannot replicate this nonlocal operation. It created a teacher-student mismatch that the student could never compensate for.

**Impact:** Negligible. Bump FWHM, confidence, and stability are unchanged. All 5 validation tests still pass.

### 1.2 Fixed: Shared simulator parameters

Created `src/ring_attractor/defaults.py` as single source of truth for all parameters. Both `RingAttractor`, `SpikeProcessor`, and `generate_dataset.py` import from it. Prevents the evaluation from silently using different parameters than training.

### 1.3 Fixed: evaluation.py `observed_idx` bug

`evaluate_predictions` and `generalization_test` used `trajectories[:, :, :input_dim]` (first k columns) instead of `trajectories[:, :, observed_idx]` (the actual observed neurons). Invisible at k/N=1.0, would produce completely wrong results for partial observation. Fixed by threading `observed_idx` through both functions and `full_evaluation`.

### 1.4 Dropped circular shift augmentation, increased N_ANGLES to 72

**The problem:** Circular shift augmentation is incompatible with partial observation (rolling a random subset of neurons doesn't correspond to rotating the bump). Since augmentation is disabled for k/N < 1.0, any partial observation experiment would have a confound: degradation could be from fewer neurons OR from loss of augmentation.

**Investigation:** We ran a no-augmentation control at k/N=1.0 with 36 angles. It failed drift on all 3 seeds tested (5.07°, 5.44°, 5.82° — all above the 5° threshold). Uniformity and circularity passed every time.

**Fix:** Increased `N_ANGLES` from 36 to 72 (5° spacing instead of 10°). This gives the model denser angular coverage directly in the training data, replacing what augmentation provided. With 72 angles and no augmentation: 2/2 seeds pass all Milestone 1 criteria.

**Consequence:** The dataset now has 144 trials (72 angles × 2 groups) instead of 72. Circular shift augmentation is no longer used. There is no augmentation confound in Milestone 2.

---

## 2. Milestone 1 exit criteria (unchanged)

| Metric | Threshold | What it tests |
|---|---|---|
| Uniformity | > 0.8 | Converged states cover the full ring |
| Circularity | > 0.7 | Ring is a circle in PCA, not a line or blob |
| Mean generalization drift | < 5° | Ring is continuous, not a polygon |

---

## 3. Final winning configuration

### Training recipe

```python
from src.train import train, TrainingConfig, full_evaluation

result = train("data/ring_attractor_dataset.npz", TrainingConfig(
    model_type="vanilla",
    n_epochs=5000,
    learning_rate=8e-4,
    scheduler="cosine",
    convergence_weight=5.0,
    convergence_steps=30,
    noise_std=0.08,
    noise_std_final=0.002,
    circular_shift_augment=False,   # not needed with 72 angles
))
```

### Parameter summary

| Component | Parameter | Value |
|---|---|---|
| **Teacher** | N | 100 |
| | J0, J1 | -2.0, 4.0 |
| | steepness | 4.0 (φ = tanh(4·ReLU)) |
| | alpha | 0.01 |
| | sigma | 0.1 |
| **Dataset** | N_ANGLES | **72** (5° spacing) |
| | Trials | 144 (72 A + 72 B) |
| | Time bins | 200 per trial |
| **Student** | Architecture | VanillaRateRNN |
| | Hidden dim | 100 |
| | alpha | 0.5 |
| | Nonlinearity | tanh |
| **Training** | Epochs | 5000 |
| | LR | 8e-4, cosine → 1e-5 |
| | Noise | 0.08 → 0.002 |
| | Conv. weight | 5.0 (first 30 steps) |
| | Augmentation | **None** |

---

## 4. Baseline characterization

### T_auto sensitivity (ring structure stability over time)

| T_auto | Uniformity | Circularity | Spread |
|---|---|---|---|
| 100 | 0.967 | 0.729 | 8.371 |
| 500 | 0.964 | 0.700 | 8.393 |
| 2000 | 0.965 | 0.699 | 8.394 |
| 5000 | 0.965 | 0.699 | 8.394 |

The ring is fully stable through T=5000. No decay.

### Drift vs T_gen (angle stability over time)

| T_gen | Mean drift | Max drift |
|---|---|---|
| 100 | 2.32° | 5.32° |
| 500 | 3.56° | 6.08° |
| 2000 | 3.70° | 6.50° |

Drift plateaus by T=1000. The ring holds intermediate angles indefinitely.

---

## 5. Milestone 2: Partial observation sweep

Now that the baseline is solid and augmentation-free, proceed to partial observation.

```python
for obs_frac in [1.0, 0.75, 0.5, 0.25, 0.1]:
    torch.manual_seed(42)
    result = train("data/ring_attractor_dataset.npz", TrainingConfig(
        model_type="vanilla",
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=30,
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=False,
        observation_fraction=obs_frac,
        checkpoint_dir=f"checkpoints/obs_{int(obs_frac*100):03d}",
    ))
    report = full_evaluation(
        result['model'], "data/ring_attractor_dataset.npz",
        result['device'], result['val_idx'], result['observed_idx'],
    )
```

**What to record for each observation fraction:**
- All three Milestone 1 metrics (uniformity, circularity, drift)
- PCA scatter plot of converged states
- Drift-vs-angle plot
- Eigenvalue spectrum
- At what k/N does the ring break? What breaks first?

**Run at least 2 seeds per observation fraction** for reproducibility.

---

## 6. Codebase map (current state)

```
src/
  ring_attractor/
    defaults.py             # Single source of truth (N_ANGLES=72)
    network.py              # No divisive normalization
    spiking.py              # Imports from defaults
    __init__.py             # Exports defaults module

  train/
    models.py               # VanillaRateRNN, LowRankRateRNN
    dataset.py              # Teacher-forcing cutoff logic
    training.py             # --scheduler CLI arg added
    evaluation.py           # observed_idx bug FIXED
    __init__.py

  generate_dataset.py       # Uses defaults, no gamma

data/
  ring_attractor_dataset.npz  # 144 trials, 72 angles, no div. norm
```
