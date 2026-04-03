# 03 — Training Second Attempt: Execution Log

**Date:** 2026-04-03
**Status:** Milestone 1 PASS — 72 angles, no augmentation, 2/2 seeds pass

---

## Result Summary

After investigating the augmentation confound and increasing training angles from 36 to 72, the vanilla RNN passes all Milestone 1 criteria **without any data augmentation**:

| Metric | Seed 42 | Seed 123 | Threshold |
|---|---|---|---|
| Uniformity | 0.846 | 0.841 | > 0.8 |
| Circularity | 0.862 | 0.893 | > 0.7 |
| Drift | 3.55° | 3.74° | < 5° |

This is the clean baseline for Milestone 2 (partial observation). No augmentation confound.

---

## Phase 1: Initial training (36 angles, with augmentation)

First, we retrained on the regenerated dataset (no divisive normalization, 36 angles) using the proven recipe with circular shift augmentation.

### Attempt 1: Wrong scheduler (plateau instead of cosine)

The CLI defaulted to plateau scheduler. LR dropped to 1e-5 by epoch 1050, wasting 80% of training.

- Uniformity: 0.759 (FAIL), Circularity: 0.836, Drift: 4.05°
- **Added `--scheduler` CLI argument** to prevent this.

### Attempt 2: Cosine scheduler, seed 42 — PASS

| Metric | Value |
|---|---|
| Uniformity | 0.924 |
| Circularity | 0.821 |
| Drift | 3.32° |
| MSE | 0.023 |
| Best val loss | 0.020 |

### Attempt 3: Cosine scheduler, seed 123 — PASS

| Metric | Value |
|---|---|
| Uniformity | 0.886 |
| Circularity | 0.721 |
| Drift | 3.56° |
| Best val loss | 0.024 |

Both seeds pass with augmentation. But augmentation is incompatible with partial observation...

---

## Phase 2: Augmentation confound investigation

### The problem

Circular shift augmentation rolls the neuron dimension, exploiting rotational symmetry. But for partial observation (k < N neurons), rolling a random subset doesn't correspond to rotating the bump. The augmentation is automatically disabled when `input_dim != output_dim`.

This means partial observation experiments would differ from the baseline in TWO ways: fewer neurons AND no augmentation. Any degradation could be from either cause.

### No-augmentation control (36 angles, k/N=1.0)

Trained 3 seeds without augmentation to isolate the effect:

| Seed | Uniformity | Circularity | Drift | Pass? |
|---|---|---|---|---|
| 42 | 0.886 | 0.901 | **5.07°** | FAIL |
| 7 | 0.893 | 0.954 | **5.44°** | FAIL |
| 123 | 0.892 | 0.914 | **5.82°** | FAIL |

**Pattern:** Uniformity and circularity always pass comfortably. Drift consistently lands in 5-6° — the model learns a slight polygon tendency with only 36 training angles (10° spacing). Augmentation was masking this by providing continuous angular coverage.

### Conclusion

Circular shift augmentation is doing real work: ~1.7° improvement in drift. Without it and with only 36 angles, the model can't learn a sufficiently continuous ring. We need denser angular coverage in the training data itself.

---

## Phase 3: Baseline characterization

Before changing N_ANGLES, we characterized the augmented model's stability.

### T_auto sensitivity (ring lifetime)

Using the seed 42 augmented model, tested autonomous fixed-point analysis at different time horizons:

| T_auto | Uniformity | Circularity | Spread |
|---|---|---|---|
| 100 | 0.967 | 0.729 | 8.371 |
| 200 | 0.965 | 0.707 | 8.385 |
| 500 | 0.964 | 0.700 | 8.393 |
| 1000 | 0.964 | 0.700 | 8.394 |
| 2000 | 0.965 | 0.699 | 8.394 |
| 5000 | 0.965 | 0.699 | 8.394 |

**The ring is perfectly stable through T=5000.** Metrics plateau by T=500 and don't decay. The eigenvalues < 1 only contract the radial direction (toward the ring), not the angular direction (around it).

### Drift vs T_gen (angle stability)

| T_gen | Mean drift | Max drift |
|---|---|---|
| 100 | 2.32° | 5.32° |
| 200 | 3.04° | 5.81° |
| 500 | 3.56° | 6.08° |
| 1000 | 3.70° | 6.48° |
| 2000 | 3.70° | 6.50° |

**Drift plateaus at ~3.7° by T=1000.** The ring holds intermediate angles indefinitely — strong evidence of a genuine continuous attractor.

---

## Phase 4: 72 angles, no augmentation — final configuration

Increased `N_ANGLES` from 36 to 72 (5° spacing) in `src/ring_attractor/defaults.py`. Regenerated dataset (144 trials). Trained without augmentation.

### Seed 42 — PASS

```
Epoch  500: train=0.167, val=0.109, lr=7.8e-04
Epoch 1000: train=0.095, val=0.080, lr=7.2e-04
Epoch 2000: train=0.059, val=0.058, lr=5.3e-04
Epoch 3000: train=0.044, val=0.038, lr=2.8e-04
Epoch 4000: train=0.037, val=0.034, lr=8.6e-05
Epoch 5000: train=0.034, val=0.031, lr=1.0e-05
```

| Metric | Value |
|---|---|
| Uniformity | 0.846 |
| Circularity | 0.862 |
| Drift | 3.55° |

### Seed 123 — PASS

| Metric | Value |
|---|---|
| Uniformity | 0.841 |
| Circularity | 0.893 |
| Drift | 3.74° |

**2/2 seeds pass.** All metrics have comfortable margins. No augmentation needed.

---

## Key Decisions and Rationale

1. **Cosine scheduler is essential.** Plateau scheduler kills LR too early (uniformity 0.759 FAIL vs 0.924 PASS).

2. **Divisive normalization removal helped.** Lower MSE (0.023 vs 0.033), better drift (3.3° vs 4.7° in first round).

3. **72 training angles replace circular shift augmentation.** Same drift performance (~3.5°), no confound for partial observation. Dataset doubles from 72 to 144 trials (still tiny at 7.9 MB).

4. **evaluation.py observed_idx bug fixed.** Would have produced wrong results for any partial observation experiment.

---

## Final Configuration

```python
TrainingConfig(
    model_type="vanilla",
    hidden_dim=100,
    alpha=0.5,
    n_epochs=5000,
    learning_rate=8e-4,
    scheduler="cosine",
    convergence_weight=5.0,
    convergence_steps=30,
    noise_std=0.08,
    noise_std_final=0.002,
    circular_shift_augment=False,   # not needed with 72 angles
    batch_size=32,
    clip_grad=1.0,
)
```

---

## Next Steps

1. **Milestone 2: Partial observation sweep.** Train at k/N = {1.0, 0.75, 0.5, 0.25, 0.1}, 2+ seeds each.
2. **Record:** uniformity, circularity, drift, PCA plots, eigenvalue spectra per condition.
3. **Find the threshold:** at what k/N does the ring break, and what breaks first?
