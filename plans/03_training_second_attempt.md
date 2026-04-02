# 03 — Training Second Attempt: Changes, Goals, and Execution Plan

**Date:** 2026-04-03
**Status:** Ready for execution
**Prerequisite reading:** `plans/02_training.md` (training guide), `plans/02_training_execution.md` (first attempt results)

---

## 0. What this document is

This is the plan for the **second training attempt**. The first attempt (documented in `02_training_execution.md`) achieved Milestone 1 — a vanilla RNN that passes all three ring attractor criteria. But during a critical review of the full pipeline, we identified three issues that need fixing before we can trust the results and move to Milestone 2 (partial observation). This document describes those fixes, what changed in the codebase, and how to re-train from scratch.

**If you haven't read `02_training.md`**, read Sections 1-5 first. The training approach, model architecture, loss function, and evaluation metrics are all the same. This document only covers **what changed and why**.

---

## 1. What changed and why

### 1.1 Removed: Divisive normalization from the simulator

**What it was:** The teacher simulator had a divisive normalization mechanism:
```
g[t] = 1 + γ · mean(r[t])
total_input = total_input / g[t]
```
with γ=2. This divided all recurrent input by a factor proportional to mean network activity, creating competition between neurons that sharpened the activity bump.

**Why we removed it:** The student RNN has dynamics `h[t] = (1-α)h[t-1] + α·tanh(W·h[t-1] + b)` — there is no equivalent of divisive normalization in this architecture. Divisive normalization is a *nonlocal* operation (it depends on the mean activity across all neurons), and a standard RNN cannot replicate it through its local recurrent weights. This created a model mismatch between teacher and student that the student could never fully compensate for.

**Impact on bump quality:** Minimal. The validation tests still pass with identical metrics:

| Metric | With div. norm (γ=2) | Without (γ=0) |
|---|---|---|
| FWHM | 112° | 112° |
| Confidence | 0.84 | 0.838 |
| Peak rate | 0.84 | 0.995 |
| Angle error (36 angles) | 1.3° | 1.3° |
| Long-term stability | 1.00 | 1.00 |

The bump shape is essentially unchanged because the tanh(4·ReLU) nonlinearity already provides the threshold and saturation that shape the bump. Divisive normalization was redundant.

**What this means for training:** The training data will be slightly different (bump profiles at the fine scale), so **the dataset must be regenerated** and **the model must be retrained from scratch**. The old checkpoint (`checkpoints/best_augmented.pt`) is invalid.

### 1.2 Fixed: Shared simulator parameters (no more defaults drift)

**The problem:** The generalization test in `evaluation.py` created `RingAttractor()` with no arguments, relying on the class defaults matching the parameters used in dataset generation. But `generate_dataset.py` defined its own copy of every parameter. If anyone changed one file without the other, the evaluation would silently use a different dynamical system than the training data came from.

**The fix:** Created `src/ring_attractor/defaults.py` — a single source of truth for all simulator and spike processing parameters. Both the `RingAttractor` class and `generate_dataset.py` now import from this file. The `evaluation.py` generalization test calls `RingAttractor()` with no args, which automatically uses the same defaults.

**What to know:** If you ever need to change a simulator parameter (e.g., J0, J1, steepness), change it in `src/ring_attractor/defaults.py` and it propagates everywhere. You should not hardcode simulator parameters elsewhere.

### 1.3 Fixed: Circular shift augmentation under partial observation

**The problem:** When `observation_fraction < 1.0`, the input tensor has shape `(B, T, k)` and the target has shape `(B, T, N)` where `k < N`. The old code rolled both by `torch.randint(0, k)` — but rolling k observed neurons by some shift doesn't correspond to rotating the bump the same way rolling all N neurons does. This would produce **inconsistent input-target pairs** in Milestone 2.

**The fix:** Circular shift augmentation now only activates when `input_dim == output_dim` (i.e., full observation). When partial observation is used, the augmentation is silently skipped.

**What this means for Milestone 2:** Circular shift augmentation will NOT be available during partial observation training. This is a real limitation — the model will only see the 36 discrete training angles. We may need an alternative augmentation strategy for partial observation (e.g., training on more angles, or adding small angle jitter to the cue). But that's a Milestone 2 problem — for now, the bug is fixed.

---

## 2. Training goals (unchanged)

The goal is **Milestone 1: Full-observation ring recovery**. Train a student RNN on the teacher's output data with k/N = 1.0 and demonstrate that the RNN has learned a continuous ring attractor.

### Milestone 1 exit criteria

| Metric | Threshold | What it tests |
|---|---|---|
| Uniformity | > 0.8 | Converged states cover the full ring, not a single point |
| Circularity | > 0.7 | The ring is a circle in PCA, not a line or blob |
| Mean generalization drift | < 5° | The ring is continuous, not a polygon of 36 discrete points |

Once all three pass, move to Milestone 2 (partial observation). Do not over-optimize.

---

## 3. Step-by-step execution plan

### Step 1: Regenerate the dataset

The old dataset was generated with divisive normalization. Regenerate:

```bash
python src/generate_dataset.py
```

This produces `data/ring_attractor_dataset.npz` (72 trials, 200 time bins, 100 neurons). The script prints validation diagnostics — check that:
- Group A angle errors are < 5° mean
- Group A confidence is > 0.8
- Group B shows confidence recovery (early < late)

### Step 2: Train with the proven recipe

The winning configuration from the first attempt should be the starting point. All the training infrastructure (noise annealing, convergence weighting, circular shift augmentation, cosine LR schedule) is already implemented in `src/train/training.py`.

**From Python:**
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
    circular_shift_augment=True,
))
```

**From the command line:**
```bash
python -m src.train.training data/ring_attractor_dataset.npz \
    --model vanilla --epochs 5000 --lr 8e-4 \
    --conv-weight 5 --conv-steps 30 \
    --noise 0.08 --noise-final 0.002 --circular-shift
```

**What to expect:**
- Training takes ~10-20 minutes on MPS (Apple Silicon) or GPU, longer on CPU.
- Loss should drop from ~1.0 to ~0.05 over 5000 epochs.
- Gradient norms should stay in [0.1, 2.0]. If consistently hitting 1.0 (the clip ceiling), the model is struggling.
- Best validation loss should be < 0.05.
- The best model is auto-saved to `checkpoints/best.pt`.

### Step 3: Evaluate

```python
report = full_evaluation(
    result['model'], "data/ring_attractor_dataset.npz",
    result['device'], result['val_idx'], result['observed_idx'],
)
```

This prints all metrics and a MILESTONE 1: PASS/FAIL verdict. See `02_training.md` Section 5 for how to interpret each metric.

### Step 4: Verify with additional seeds

The first attempt only verified one random seed. Run at least 2 more seeds to confirm the result is reproducible:

```python
import torch

for seed in [42, 123, 7]:
    torch.manual_seed(seed)
    result = train("data/ring_attractor_dataset.npz", TrainingConfig(
        model_type="vanilla",
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=30,
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=True,
    ))
    report = full_evaluation(
        result['model'], "data/ring_attractor_dataset.npz",
        result['device'], result['val_idx'], result['observed_idx'],
    )
    print(f"Seed {seed}: pass={report['milestone_1_pass']}")
```

**Expectation:** At least 2/3 seeds should pass Milestone 1. If only 1/3 passes, the recipe needs tuning before moving to Milestone 2.

---

## 4. If the proven recipe no longer works

Removing divisive normalization changes the training data slightly. The old recipe *should* still work (the data is similar), but if it doesn't, here's the tuning priority:

1. **Try more epochs** (7000-10000). The simpler dynamics might need more time to converge.
2. **Try lower noise** (`noise_std=0.05`). Without divisive normalization, the bump profile is marginally different and may need less regularization.
3. **Try higher convergence weight** (`convergence_weight=8.0`). If the model is learning identity-like dynamics.
4. **Try alpha=0.3**. Slower leak rate = more stable dynamics.

See `02_training.md` Section 6 for the full troubleshooting guide — it still applies.

---

## 5. Evaluation metrics — quick reference

These are identical to the first attempt. Reproduced here for self-containedness.

### 5.1 Autonomous fixed-point analysis (the core test)

Initialize 300 random hidden states, run autonomously for 500 steps, see where they converge.

- **Spread** (mean ||h_final||): >> 0 means non-trivial fixed points. ~0 means origin collapse.
- **Uniformity** (1 - |mean(exp(iθ))|): ~1 means angles uniformly distributed. ~0 means single point.
- **Circularity** (min/max std ratio in PCA): ~1 means circle. ~0 means line or blob.

### 5.2 Generalization test (ring vs. polygon)

Test 36 intermediate angles (5°, 15°, ..., 355°) that the model never trained on. Teacher-force each angle, then run autonomously for 500 steps. Measure angular drift.

- **True continuous ring:** drift ≈ 0 for all test angles.
- **Discrete fixed points (polygon):** sawtooth drift pattern — each test angle drifts toward the nearest training angle.
- **Mean |drift| < 5°** is the pass criterion.

### 5.3 Eigenvalue analysis

Compute Jacobian eigenvalues at fixed points on the ring.

- **Ring signature:** ~2 eigenvalues near |λ|=1 (neutral ring direction), all others < 1 (attracting).
- **All |λ| < 1:** isolated fixed points, no ring.
- **Any |λ| > 1:** unstable dynamics.

### 5.4 Visual diagnostics

```python
import matplotlib.pyplot as plt
import numpy as np

# PCA of converged hidden states (should be a rainbow circle)
rs = report['ring_score']
plt.figure(figsize=(6, 6))
plt.scatter(rs.pca_proj[:, 0], rs.pca_proj[:, 1], c=rs.theta_final, cmap='hsv', s=5)
plt.colorbar(label='decoded angle')
plt.title('PCA of converged hidden states')
plt.axis('equal')
plt.savefig('figs/ring_pca.png')

# Drift vs angle (should be flat near zero, NOT sawtooth)
gen = report['generalization']
plt.figure(figsize=(8, 4))
plt.plot(np.degrees(gen.test_angles), gen.drift_deg, 'o-')
plt.axhline(0, color='k', ls='--')
plt.xlabel('Test angle (deg)')
plt.ylabel('Drift (deg)')
plt.title(f'Generalization drift (mean |drift| = {gen.mean_abs_drift_deg:.1f}°)')
plt.savefig('figs/drift_vs_angle.png')
```

---

## 6. Codebase map (current state)

```
src/
  ring_attractor/
    defaults.py             # NEW — single source of truth for all parameters
    network.py              # MODIFIED — removed divisive normalization (gamma)
    spiking.py              # MODIFIED — imports defaults for parameter values
    __init__.py             # MODIFIED — exports defaults module
    plotting.py             # Unchanged

  train/
    models.py               # Unchanged — VanillaRateRNN, LowRankRateRNN
    dataset.py              # Unchanged — teacher-forcing cutoff logic
    training.py             # MODIFIED — circular shift only when input_dim == output_dim
    evaluation.py           # Unchanged — all metrics + Milestone 1 pass/fail
    __init__.py             # Unchanged

  generate_dataset.py       # MODIFIED — uses defaults module, no gamma
  demo_ring_attractor.py    # MODIFIED — no gamma
  validate_simulator.py     # Unchanged (uses RingAttractor() defaults)

data/
  ring_attractor_dataset.npz  # MUST REGENERATE — old data used divisive normalization

checkpoints/
  best_augmented.pt           # INVALID — trained on old data, do not use
```

---

## 7. Summary of all parameter values

### Teacher simulator (from `src/ring_attractor/defaults.py`)

| Parameter | Value | Description |
|---|---|---|
| N | 100 | Neurons on the ring |
| J0 | -2.0 | Inhibition baseline (cosine kernel) |
| J1 | 4.0 | Excitation amplitude (cosine kernel) |
| steepness | 4.0 | Slope of φ(x) = tanh(s · ReLU(x)) |
| alpha | 0.01 | Leak rate (dt/τ) |
| sigma | 0.1 | Additive Gaussian noise |
| cue_amplitude | 3.0 | Peak of Gaussian cue |

### Student RNN

| Parameter | Value | Description |
|---|---|---|
| Architecture | VanillaRateRNN | Full recurrent matrix W_hh |
| Hidden dim | 100 | Matches N |
| alpha | 0.5 | Leak rate per time bin (= 50 × teacher's 0.01) |
| Nonlinearity | tanh | Standard (different from teacher's tanh(4·ReLU)) |
| Parameters | ~30,200 | W_xh + W_hh + W_hy + biases |

### Training hyperparameters (proven recipe)

| Parameter | Value | Description |
|---|---|---|
| Epochs | 5000 | Monitor convergence — may need more or fewer |
| Learning rate | 8e-4 | With cosine annealing to 1e-5 |
| Scheduler | cosine | CosineAnnealingLR |
| Convergence weight | 5.0 | Up-weight first 30 autonomous steps |
| Convergence steps | 30 | How many steps to up-weight |
| Noise std | 0.08 → 0.002 | Linear annealing over training |
| Circular shift | True | Augment by rotating neuron dimension |
| Batch size | 32 | |
| Gradient clipping | 1.0 | Max norm |
| K_min, K_max | 5, 20 | Teacher-forcing window (randomized per trial) |

### Spike processing pipeline

| Parameter | Value | Description |
|---|---|---|
| dt | 0.01 | Integration timestep |
| rate_scale | 100 | Poisson λ multiplier |
| bin_factor | 50 | Integration steps per time bin |
| smoothing_window | 3 | Causal boxcar kernel width |

---

## 8. After Milestone 1: what's next

Once the ring is confirmed with 2+ seeds:

1. **Save the best checkpoint** as `checkpoints/best_v2.pt` (or similar).
2. **Move to Milestone 2: Partial observation sweep.**

```python
for obs_frac in [1.0, 0.5, 0.25, 0.1]:
    result = train("data/ring_attractor_dataset.npz", TrainingConfig(
        model_type="vanilla",
        n_epochs=5000,
        learning_rate=8e-4,
        scheduler="cosine",
        convergence_weight=5.0,
        convergence_steps=30,
        noise_std=0.08,
        noise_std_final=0.002,
        circular_shift_augment=True,   # auto-disabled when obs_frac < 1.0
        observation_fraction=obs_frac,
    ))
    report = full_evaluation(
        result['model'], "data/ring_attractor_dataset.npz",
        result['device'], result['val_idx'], result['observed_idx'],
    )
    print(f"k/N = {obs_frac}: pass={report['milestone_1_pass']}, "
          f"unif={report['ring_score'].uniformity:.3f}, "
          f"circ={report['ring_score'].circularity:.3f}, "
          f"drift={report['generalization'].mean_abs_drift_deg:.1f}°")
```

**Note:** Circular shift augmentation is automatically disabled for partial observation (the fix from Section 1.3). The model will only see 36 discrete angles during training. This may make the drift metric harder to pass — if so, consider training on more angles (increase `N_ANGLES` in `defaults.py` and regenerate).

**What to record for each observation fraction:**
- All three Milestone 1 metrics (uniformity, circularity, drift)
- PCA scatter plot of converged states
- Drift-vs-angle plot
- Eigenvalue spectrum
- At what k/N does the ring break? What breaks first — uniformity, circularity, or drift?
