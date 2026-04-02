# 02 — Training Execution Log: Ring Attractor RNN Recovery

**Date:** 2026-04-02
**Status:** Milestone 1 PASS — ring attractor recovered with vanilla RNN

---

## Result Summary

After 14 training attempts, the vanilla RNN passes all three Milestone 1 criteria:

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Uniformity | 0.920 | > 0.8 | PASS |
| Circularity | 0.889 | > 0.7 | PASS |
| Mean drift | 4.71° | < 5° | PASS |

Additional metrics: MSE = 0.033, angle error = 3.56°, spread = 7.91.

**Winning configuration:** 5000 epochs, circular shift augmentation, noise annealing (0.08 → 0.002), convergence weighting (5×, 30 steps), cosine LR schedule (8e-4 → 1e-5). Saved to `checkpoints/best_augmented.pt`.

---

## Attempt-by-Attempt Log

### Attempt 1: Vanilla, default settings

**Config:** vanilla, 1500 epochs, alpha=0.5, lr=1e-3, plateau scheduler, no augmentation, no noise

**Results:**
- Uniformity: 0.761 (FAIL), Circularity: 0.815, Drift: ~9°
- 2 eigenvalues near unit circle (good ring signature)
- MSE: 0.038, angle error: 4.1°

**Conclusions:** Ring structure emerges out of the box, but uniformity and drift fail. The model has 10k recurrent parameters — enough capacity to memorise 36 discrete attractors rather than a continuous ring.

### Attempt 2: Low-rank (rank=2), convergence weighting

**Config:** lowrank R=2, 2000 epochs, cosine sched, convergence weight=5, 30 steps

**Results:**
- Uniformity: 0.759, Circularity: 0.001 (collapsed to 1D), Drift: 44.65°
- Loss stuck at ~0.85 (barely below random-guess level)
- sigma_1=1.02, sigma_2=0.034 — collapsed to rank 1

**Conclusions:** The low-rank model failed to train. Root cause: the initialisation `randn(H,R)/sqrt(H)` makes `UV^T/H` have singular values ~0.01, versus ~0.14 for Xavier-initialised vanilla. The recurrent pathway is effectively dead at initialisation.

### Attempt 3: Vanilla, convergence weighting + cosine schedule

**Config:** vanilla, 2000 epochs, cosine sched, conv weight=5, 30 steps, lr=5e-4

**Results:**
- **Uniformity: 0.902 (PASS), Circularity: 0.927 (PASS)**, Drift: 9.53° (FAIL)
- 2 eigenvalues near unit circle — perfect ring signature
- Best ring structure of all attempts

**Conclusions:** Convergence weighting is the key ingredient for ring structure. The loss up-weighting forces the RNN to learn active dynamics (convergence) rather than passive state maintenance. But the drift still fails — the model has discrete attractors at the 36 training angles.

### Attempt 4: Vanilla + noise injection (0.03)

**Config:** Same as attempt 3 + noise_std=0.03

**Results:**
- Uniformity: 0.926, Circularity: 0.843, Drift: 8.80°
- Marginal improvement on drift from noise

**Conclusions:** Light noise injection doesn't break discrete attractors. The noise is too small compared to the basin depth.

### Attempt 5: Vanilla + strong noise (0.1) + weight decay

**Config:** vanilla, 3000 epochs, noise=0.1, wd=1e-4, cosine sched

**Results:**
- Uniformity: 0.907, Circularity: 0.892, Drift: 7.09°
- Eigenvalues: max |λ|=0.945, 0 near unit — the ring mode is too damped

**Conclusions:** Noise + weight decay improves drift slightly. But the strong noise makes the model learn over-contracting dynamics (all eigenvalues well below 1). The "ring" becomes a set of weakly separated discrete attractors.

### Attempt 6: Vanilla, smaller hidden dim (50)

**Config:** vanilla, hidden_dim=50, 2000 epochs, noise=0.05

**Results:**
- Uniformity: 0.841, Circularity: 0.807, Drift: 8.70°
- Reducing capacity didn't help

**Conclusions:** The 50-dim model has fewer parameters but still enough to create discrete attractors. The bottleneck isn't capacity — it's the discrete angle coverage in training data.

### Diagnostic: Drift vs T_gen and k_teacher

**Key finding:** Drift scales with T_gen (number of autonomous steps):

| T_gen | Drift (no noise) | Drift (noise=0.05) |
|---|---|---|
| 50 | 3.14° | — |
| 100 | 4.34° | — |
| 200 | 6.04° | — |
| 500 | 8.73° | — |

The drift grows roughly linearly (~0.015°/step), characteristic of slow drift along a slightly non-uniform ring. NOT polygon behaviour (which would show immediate convergence to discrete points). Teacher-forcing length (10–100 steps) doesn't affect drift — the issue is in autonomous dynamics, not initialisation.

### Attempt 7: Low-rank (fixed initialisation)

**Config:** lowrank R=2, U/V scaled to give sigma_k(W)~1.5, 2000 epochs, noise=0.05

**Results:**
- sigma_1=1.80, sigma_2=1.49 (reasonable magnitudes now)
- But: Circularity 0.001, Drift 45.64°, MSE 0.67
- Loss improved from 0.96 to 0.57, but plateaued far above vanilla

**Conclusions:** The fixed initialisation gave reasonable singular values, but the rank-2 constraint is fundamentally too tight. The vanilla model's top 5 SVs are all similar (1.95, 1.89, 1.86, 1.82, 1.78) — no clear rank-2 break. The tanh nonlinearity introduces effective higher-rank structure that the rank-2 model can't capture.

### Attempt 8: Vanilla, alpha=0.3

**Config:** vanilla, alpha=0.3, 2000 epochs, noise=0.05, conv weight=5

**Results:**
- Uniformity: 0.842, Circularity: 0.774, Drift: 8.11°
- 2 eigenvalues near unit circle

**Per-angle drift pattern:** NOT sawtooth. Drifts are ±20° at seemingly random angles. Some angles are stable (drift ~1°), others drift significantly (~18°). This confirms the ring has non-uniform regions, not 36 discrete attractors.

### Attempts 9–10: Strong noise (0.15–0.2)

**Key finding:** Strong noise makes drift CONSTANT across T_gen values. Example: drift=8.50° for ALL T_gen from 50 to 500. This means the model converges immediately to fixed points — the noise made the dynamics too contracting. Eigenvalues are 0.83–0.90, far below the neutral stability (λ=1) needed for a ring.

**Fundamental tension:** Noise encourages smooth dynamics (good for continuity) but also makes the model contract more strongly (bad for neutral ring stability).

### Warm-start: Vanilla → Low-rank distillation

**Approach:** Train vanilla, SVD of W_hh, initialise rank-2 model from top-2 SVD components, fine-tune.

**Results:**
- Excellent ring structure: Uniformity 0.890, Circularity 0.945
- Eigenvalues: 1.8 near unit circle
- But: drift at T_gen=500 was 15.06° (drift grows super-linearly with T_gen)
- MSE: 0.23 (much worse than vanilla's 0.03)

**Conclusions:** The rank-2 approximation loses 73% of W_hh energy (Frobenius norm 2.72 vs 10.07). The "background" modes are important for the output mapping, even if the ring dynamics are rank-2.

### Two-phase training

**Approach:** Phase 1 with noise (build ring), Phase 2 without noise (fine-tune for stability).

**Results:** Drift improved at short T_gen (2.13° at T=50), but Phase 2 WITHOUT augmentation hurt uniformity (dropped from 0.94 to 0.73). The model overfits to discrete angles during fine-tuning.

### Circular shift augmentation (breakthrough)

**Approach:** Randomly circularly shift the neuron dimension per batch during training. Since the ring attractor has rotational symmetry, `torch.roll(x, shift, dim=-1)` produces valid training examples at shifted angles. This provides continuous angle coverage.

Per-neuron standardisation stats are approximately uniform (std of means: 0.37, std of stds: 0.30 — ~3% variation), so circular shifts are a valid augmentation.

**Results (3000 epochs, noise annealing 0.05→0.005):**
- Uniformity: 0.926, Circularity: 0.789
- **T_gen=50: 1.65° (best ever), T_gen=200: 3.74° (PASSES!)**
- T_gen=500: 8.75° (still fails, but best short-term drift)

### Drift decomposition diagnostic

Measured initialization error vs pure RNN drift:
- Initialization error (spike noise): 2.47° mean
- Pure RNN drift (θ_final − θ_init): 8.52° mean
- Total measured drift: 8.69° mean

The spike pipeline contributes ~2.5° of angle error, but the RNN drift dominates.

### Winning run: 5000 epochs + augmentation + noise annealing

**Config:**
```python
# Model
model_type = 'vanilla', hidden_dim = 100, alpha = 0.5

# Training
n_epochs = 5000
lr = 8e-4, cosine schedule → 1e-5
noise_std: 0.08 → 0.002 (linear annealing)
convergence_weight = 5.0, convergence_steps = 30
gradient clipping: max_norm = 1.0
batch_size = 32

# Augmentation
circular_shift = True  # torch.roll(x, randint(0, N), dim=-1) per batch

# Seed
torch.manual_seed(42)
```

**Training curve:**
```
Epoch 1000: train=0.119, val=0.069, noise=0.064, lr=7.2e-4
Epoch 2000: train=0.091, val=0.071, noise=0.049, lr=5.3e-4
Epoch 3000: train=0.075, val=0.047, noise=0.033, lr=2.8e-4
Epoch 4000: train=0.061, val=0.039, noise=0.018, lr=8.5e-5
Epoch 5000: train=0.055, val=0.041, noise=0.002, lr=1.0e-5
Best val = 0.033
```

**Evaluation:**
- MSE: 0.033, Angle error: 3.56°
- Spread: 7.91, Uniformity: 0.920, Circularity: 0.889
- **Mean drift: 4.71° (PASSES)**
- Max drift: 16.77° (at θ_test = 95°)
- Eigenvalues: max |λ| = 0.92–0.99

**Drift vs T_gen:**
```
T_gen= 50: drift = 2.36°
T_gen=100: drift = 3.56°
T_gen=200: drift = 4.30°
T_gen=500: drift = 4.71°
```

The drift PLATEAUS between T=200 and T=500 (from 4.30° to 4.71°), meaning the ring dynamics are nearly neutral at long times. This is qualitatively different from earlier attempts where drift grew linearly.

---

## Key Insights and Lessons

### 1. Convergence weighting is essential

Up-weighting the first 30 autonomous steps by 5× forces the RNN to learn active dynamics (convergence toward the manifold) rather than just passive state maintenance. Without it, the model can achieve low loss with identity-like dynamics.

### 2. The fundamental challenge: continuous ring from discrete data

Training on 36 discrete angles with 10k recurrent parameters creates a model that CAN memorise 36 individual attractors. The ring needs to be made continuous through data augmentation and/or regularisation.

### 3. Circular shift augmentation is the key breakthrough

Because the ring attractor has rotational symmetry, circularly shifting the neuron dimension produces valid training data at arbitrary angles. This gives the model continuous angle coverage during training, preventing discrete attractor formation.

### 4. Noise annealing balances smoothness and stability

Strong noise during early training builds a smooth ring (prevents individual basins). Decaying noise in later training allows the ring mode to approach neutral stability (eigenvalue near 1). Constant strong noise makes eigenvalues too contracting; no noise allows discrete attractors.

### 5. The low-rank model (rank-2) cannot practically train

Despite the teacher's cosine kernel being exactly rank 2, the student RNN needs higher effective rank. The tanh nonlinearity and the output mapping require the full hidden space to represent the dynamics accurately. The low-rank model either fails to train (bad init) or produces poor outputs (good ring structure but MSE 10× worse than vanilla).

### 6. Eigenvalue analysis is informative but imperfect

The ideal ring attractor signature (2 eigenvalues at |λ|=1, all others <1) appeared in early attempts (attempts 3, 8) but not in the winning model. The winning model has max |λ| ≈ 0.95 — the ring mode is slightly damped. Yet the ring passes all structural tests. The Jacobian eigenvalues at a POINT on the ring don't fully capture the global dynamics.

### 7. The drift metric is the hardest to satisfy

Uniformity and circularity pass with most reasonable configurations. Drift requires near-perfect ring uniformity: if any angle is even slightly more attractive, the bump will slowly drift toward it over 500 autonomous steps. This is an extremely stringent criterion.

---

## Code Changes Made

### `src/train/models.py`
- Added `noise_std` parameter to `forward()` in both `VanillaRateRNN` and `LowRankRateRNN`
- When `noise_std > 0` and `self.training`, Gaussian noise is added to `h` at each step
- `run_autonomous()` is unchanged (no noise during evaluation)

### `src/train/training.py`
- Added `noise_std: float = 0.0` to `TrainingConfig`
- Training loop passes `config.noise_std` to `model(x, noise_std=...)`

### Low-rank initialisation fix
- Changed `LowRankRateRNN` U/V initialisation from `randn(H,R)/sqrt(H)` to `randn(H,R)*sqrt(1.5)`, giving initial singular values of W_hh ≈ 1.5 (matching vanilla Xavier scale)

---

## What the winning model looks like

### Singular values of W_hh (top 5)
```
sigma_1 = 2.34   ← ring modes
sigma_2 = 2.27   ← ring modes
sigma_3 = 2.14
sigma_4 = 2.13
sigma_5 = 1.93
```

No sharp rank-2 break — the model uses the full 100D space. But the top 2 SVs do have a slight edge over the rest.

### Per-angle drift pattern
The drift is NOT sawtooth (not a polygon). Most angles have |drift| < 7°, with one outlier at 95° (+16.77°). The drift pattern is roughly random with no systematic bias — consistent with a continuous but slightly non-uniform ring.

### PCA of converged hidden states
300 random initialisations converge to a smooth ring in PCA space (circularity 0.89). The decoded angles span the full circle uniformly (uniformity 0.92).

---

## Reproduction

```python
import torch
from src.train.models import create_model
from src.train.evaluation import full_evaluation

model = create_model('vanilla', 100, 100, 100, alpha=0.5)
model.load_state_dict(torch.load('checkpoints/best_augmented.pt', weights_only=True))
device = torch.device('mps')  # or 'cpu'
model = model.to(device)

report = full_evaluation(model, 'data/ring_attractor_dataset.npz', device)
# Should print: MILESTONE 1: PASS
```

To retrain from scratch (the circular shift augmentation is NOT yet in the standard `train()` function — it was used in a standalone training script; see next steps):
```python
# See the inline training code in the winning run.
# Circular shift augmentation needs to be integrated into training.py
# before this can be run via the standard API.
```

---

## Next Steps

1. **Integrate augmentation into training.py** — Add `circular_shift_augment: bool` to `TrainingConfig` and implement in the training loop. Also add `noise_schedule: tuple[float, float]` for noise annealing.

2. **Move to Milestone 2: Partial observation** — Sweep `observation_fraction` ∈ {1.0, 0.5, 0.25, 0.1} using the winning configuration and characterise how the ring degrades.

3. **Try to improve drift further** — The current 4.71° barely passes. Ideas: more epochs, different noise schedule, explicit angle-stability loss. But per the plan, don't over-optimise — move on.

4. **Investigate the 95° outlier** — Why does one test angle drift 16.77° while most are <7°? Could reveal interesting structure in the ring.

---

## Attempt Summary Table

| # | Config | Unif | Circ | Drift₅₀₀ | Key insight |
|---|---|---|---|---|---|
| 1 | Vanilla defaults | 0.761 | 0.815 | ~9° | Ring emerges but discrete |
| 2 | Low-rank R=2 | 0.759 | 0.001 | 44.6° | Bad init — SVs too small |
| 3 | +conv weight | **0.902** | **0.927** | 9.5° | Conv weighting crucial |
| 4 | +noise 0.03 | 0.926 | 0.843 | 8.8° | Marginal improvement |
| 5 | +noise 0.1, wd | 0.907 | 0.892 | 7.1° | Over-contracting |
| 6 | hidden=50 | 0.841 | 0.807 | 8.7° | Capacity not the issue |
| 7 | LR fixed init | 0.869 | 0.001 | 45.6° | R=2 too tight |
| 8 | α=0.3 | 0.842 | 0.774 | 8.1° | No improvement |
| 9 | noise=0.2 | 0.861 | 0.889 | 8.5° | Drift constant vs T_gen |
| 10 | α=0.3, noise=0.15 | 0.933 | 0.921 | 6.4° | Best no-augmentation |
| 11 | Two-phase | 0.734 | 0.839 | 7.7° | P2 hurts uniformity |
| — | Noise anneal 3000ep | 0.878 | 0.938 | 7.1° | Anneal helps |
| — | Circ shift 3000ep | 0.926 | 0.789 | 8.8° | Best T₅₀ drift (1.65°) |
| — | Shift+wd+2phase | 0.761 | 0.857 | 7.3° | P2 still hurts |
| **✓** | **Shift+anneal 5000ep** | **0.920** | **0.889** | **4.71°** | **MILESTONE 1 PASS** |
