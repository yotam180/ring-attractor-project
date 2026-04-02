# Training Guide: Ring Attractor RNN Recovery

**Date:** 2026-04-02
**For:** Next student picking up this project
**Prerequisites:** Familiarity with PyTorch, RNNs, basic dynamical systems

---

## 1. What this project is about

We have a **teacher** (a ring attractor simulator) and a **student** (an RNN). The teacher produces neural firing patterns where a "bump" of activity sits at some angle on a ring of 100 neurons, and maintains it purely through recurrent dynamics. Our goal is to train the student RNN to reproduce these dynamics — not just predict the next timestep, but actually embed a continuous ring attractor in its recurrent weights.

**Why this matters:** The larger research question is whether you can recover attractor dynamics from partial observations (seeing only k out of N neurons). But first, we need the full-observation case (k = N) to work. That's what this document covers.

**What "working" means (Milestone 1 exit criteria):**

| Metric | Threshold | What it tests |
|---|---|---|
| Uniformity | > 0.8 | Converged states cover the full ring, not a single point |
| Circularity | > 0.7 | The ring is a circle in PCA, not a line or blob |
| Mean generalization drift | < 5 deg | The ring is continuous, not a polygon of 36 discrete points |

If you hit these three numbers, you have a working ring attractor RNN. Move on to partial observation.

---

## 2. What's already built

### Codebase map

```
src/
  ring_attractor/           # Teacher simulator
    network.py              # RingAttractor class: cosine-kernel weights,
                            #   phi(x) = tanh(4 * ReLU(x)), divisive normalisation
    spiking.py              # SpikeProcessor: rates -> Poisson spikes -> bin -> smooth
    plotting.py             # Polar plots, heatmaps

  train/                    # Student RNN training
    models.py               # VanillaRateRNN (30k params), LowRankRateRNN (21k params)
    dataset.py              # PyTorch Dataset with per-trial teacher-forcing cutoff
    training.py             # Training loop, CLI entry point
    evaluation.py           # All 5 evaluation metrics + Milestone 1 pass/fail

  generate_dataset.py       # Generates data/ring_attractor_dataset.npz
  demo_ring_attractor.py    # Interactive demo of the simulator

data/
  ring_attractor_dataset.npz  # 72 trials, 200 time bins, 100 neurons (5 MB)

plans/
  revised_training_strategy.md  # Full theoretical background and strategy
```

### The dataset

Already generated. 72 trials:

- **Group A (36 trials):** Bump maintenance. A converged bump at one of 36 angles (0, 10, 20, ..., 350 deg), recorded for 200 time bins during steady-state autonomous dynamics. The simulator has noise (sigma=0.1), so there's realistic jitter.

- **Group B (36 trials):** Perturbation recovery. Same angles, but Gaussian noise is added to the converged bump, then the simulator recovers to a clean bump. The trajectory shows the transient recovery followed by maintenance.

All data is **standardised per neuron** (zero mean, unit variance). The normalisation parameters (mean, std) are saved in the .npz file for de-standardisation during evaluation.

To regenerate (if needed): `python src/generate_dataset.py`

### The models

Both follow the same dynamics:

```
h[t] = (1 - alpha) * h[t-1] + alpha * tanh(W_rec @ h[t-1] + W_xh @ x[t] + b)
y[t] = W_hy @ h[t]
```

- **VanillaRateRNN:** W_rec is a full 100x100 matrix. 30,200 trainable parameters.
- **LowRankRateRNN:** W_rec = U @ V^T / H where U, V are 100x2 matrices (rank 2). 20,600 parameters. This is motivated by the teacher's cosine connectivity kernel being exactly rank 2.

The leak rate `alpha` controls how fast the RNN integrates. At alpha=0.5, each time bin corresponds to ~0.5 membrane time constants of the simulator.

---

## 3. The training approach (and why it's this way)

### The problem with naive training

The obvious approach — feed the RNN ground-truth rates at every step and predict the next step — **doesn't work**. The ring attractor is a stable system: x[t+1] ~ x[t] most of the time. The RNN can achieve near-zero loss by learning a near-identity mapping through the input pathway (W_xh), completely ignoring its recurrent weights (W_hh). The recurrent weights are exactly where attractor structure needs to live.

The old notebooks (11b, 12b in `old/`) tried this approach and failed for exactly this reason.

### The fix: autonomous training

The key insight is forcing the RNN to predict **without input** for most of each trial:

1. **Teacher-forcing phase (steps 0 to K-1):** Feed the RNN the observed firing rates as input. This initialises the hidden state to "hold" the right bump. No loss is computed here.

2. **Autonomous phase (steps K to T-1):** Input is zeroed. The RNN must maintain (Group A) or recover (Group B) the bump using only its recurrent dynamics. **All loss is computed here.**

3. **K is randomised per trial** (Uniform between 5 and 20 bins) to prevent the RNN from learning a fixed transition artifact.

This forces the network to embed attractor dynamics in W_hh and the bias b, because there's no input to fall back on during the autonomous phase.

---

## 4. How to train

### Quick start (proven recipe)

```python
from src.train import train, TrainingConfig, full_evaluation

# Winning configuration — passes Milestone 1
result = train("data/ring_attractor_dataset.npz", TrainingConfig(
    model_type="vanilla",
    n_epochs=5000,
    scheduler="cosine",
    learning_rate=8e-4,
    convergence_weight=5.0,
    convergence_steps=30,
    noise_std=0.08,
    noise_std_final=0.002,
    circular_shift_augment=True,
))

# Evaluate
report = full_evaluation(
    result['model'], "data/ring_attractor_dataset.npz",
    result['device'], result['val_idx'], result['observed_idx'],
)
```

Or from the command line:
```bash
python -m src.train.training data/ring_attractor_dataset.npz \
    --model vanilla --epochs 5000 --lr 8e-4 \
    --conv-weight 5 --conv-steps 30 \
    --noise 0.08 --noise-final 0.002 --circular-shift
```

A pre-trained model is available at `checkpoints/best_augmented.pt`:
```python
from src.train.models import create_model
import torch
model = create_model('vanilla', 100, 100, 100, alpha=0.5)
model.load_state_dict(torch.load('checkpoints/best_augmented.pt', weights_only=True))
```

### What to expect during training

- **Epochs 1-50:** Loss drops quickly from ~1.0 to ~0.3-0.5. The RNN learns the rough scale and shape of bumps.
- **Epochs 50-300:** Slower improvement. The RNN starts learning to maintain bumps at different angles.
- **Epochs 300-1000:** Fine-tuning. Ring structure emerges (if it's going to). Watch the validation loss — if it plateaus here, the model may be stuck.
- **Epochs 1000-1500:** Diminishing returns. If the ring hasn't appeared by epoch 1000, hyperparameter tuning is needed (see Section 6).

**Gradient norms** should stay roughly in [0.1, 2.0]. If they're consistently hitting the clip threshold (1.0), the model may be struggling with long autonomous rollouts. If they're very small (< 0.01), the model may have converged to a local minimum (likely the identity/origin attractor).

### Training the low-rank model

```python
result = train("data/ring_attractor_dataset.npz", TrainingConfig(
    model_type="lowrank",
    n_epochs=1500,
))
```

The low-rank model has fewer parameters and a structural prior that matches the teacher (rank 2). It may converge faster or struggle more — try both. The low-rank model is more interpretable (you can directly inspect U and V to see if they learned cosine/sine structure).

### Suggested experiment order

1. **Vanilla, default settings.** This is the baseline. If it works, great. If not, it tells you the training pipeline needs adjustment, not the architecture.
2. **Low-rank, default settings.** Compare. The structural prior might help or hurt.
3. **Whichever worked better** becomes the basis for hyperparameter exploration (Section 6).

---

## 5. How to evaluate: what the metrics mean

After training, `full_evaluation()` prints a report. Here's how to read it.

### 5.1 Predictive metrics (necessary but NOT sufficient)

- **MSE:** Prediction error on the autonomous phase of held-out trials. Should be well below 1.0 (the data is standardised, so MSE=1 is random-guess level). A good model might reach 0.01-0.1.
- **Angle error:** Mean absolute circular error between decoded predicted angle and decoded true angle. Below 10 deg is good.

**Why not sufficient:** An RNN that learns a single stable fixed point (one angle) can get decent MSE on Group A trials at nearby angles. You need the structural metrics to confirm a real ring.

### 5.2 Autonomous fixed-point analysis (THE core test)

This is the most important evaluation. It answers: **does the trained RNN contain a ring attractor?**

The procedure: initialise 300 random hidden states, run each autonomously (zero input) for 500 steps, see where they converge.

- **Spread** (mean ||h_final||): Should be >> 0. If near 0, all states collapsed to the origin — the RNN learned the trivial fixed point (no bump at all).

- **Uniformity** (1 - |mean(exp(i*theta))|): Should be > 0.8. If near 0, all converged states encode the same angle — the RNN learned a single fixed point, not a ring. If near 1, the converged states cover the full circle uniformly.

- **Circularity** (min/max std ratio after PCA): Should be > 0.7. If near 0, the converged states form a line or cluster in hidden space, not a circle. If near 1, they form a proper ring.

**Visual diagnostic:** The `ring_score.pca_proj` array (M x 2) and `ring_score.theta_final` can be used for a scatter plot:
```python
import matplotlib.pyplot as plt
rs = report['ring_score']
plt.scatter(rs.pca_proj[:, 0], rs.pca_proj[:, 1], c=rs.theta_final, cmap='hsv', s=5)
plt.colorbar(label='decoded angle')
plt.title('PCA of converged hidden states')
plt.axis('equal')
```
A success looks like a smooth circle with a rainbow colour gradient. A failure looks like a blob, a line, or a single cluster.

### 5.3 Generalization test (ring vs. polygon)

Training uses 36 discrete angles (0, 10, 20, ..., 350 deg). The model might learn 36 discrete fixed points arranged in a circle rather than a true continuous ring. These look identical in PCA.

To distinguish: test on 36 **intermediate** angles (5, 15, 25, ..., 355 deg) that were never in the training set. Teacher-force the RNN to hold each intermediate angle, then let it run autonomously. A true ring maintains the angle. A polygon of 36 points pulls the bump toward the nearest training angle.

- **Mean |drift| < 5 deg:** Pass. The ring is continuous.
- **Mean |drift| > 10 deg:** The model learned discrete attractors, not a ring.
- **Sawtooth pattern in drift vs. angle:** Classic polygon signature.

**Visual diagnostic:**
```python
gen = report['generalization']
plt.plot(np.degrees(gen.test_angles), gen.drift_deg, 'o-')
plt.axhline(0, color='k', ls='--')
plt.xlabel('Test angle (deg)')
plt.ylabel('Drift (deg)')
```

### 5.4 Eigenvalue analysis

At a fixed point on the ring, the Jacobian of the RNN dynamics should have:
- **One eigenvalue pair near |lambda| = 1:** The neutral direction (sliding along the ring). This is THE ring attractor signature.
- **All other eigenvalues |lambda| < 1:** These are the attracting directions (perturbations perpendicular to the ring decay).

If you see:
- All |lambda| < 1: the model learned isolated stable fixed points, not a ring.
- Some |lambda| > 1: unstable dynamics. The model may blow up during long autonomous rollouts.
- The count "# near unit" should be ~2 (a conjugate pair for the cos/sin modes).

### 5.5 Singular values of W_hh

For the low-rank model, SVD of W_hh should show exactly 2 dominant singular values (the model is rank-2 by construction). But look at whether the *ratio* between the top 2 and the rest is large — if the two modes are weak, the recurrent dynamics are too weak to sustain a bump.

For the vanilla model, SVD reveals what effective rank the network learned. If 2 singular values dominate, the network discovered the rank-2 structure on its own.

---

## 6. When it doesn't work: a troubleshooting guide

### Symptom: Loss plateaus above 0.5

**Likely cause:** The RNN is predicting something close to the mean (which is 0 in standardised space) everywhere. It hasn't learned to produce bumps at all.

**Try:**
- Lower alpha (e.g., 0.2). A slower leak rate makes the dynamics more stable and easier to learn initially.
- Increase the teacher-forcing window: `k_min=10, k_max=30`. More context during teacher-forcing gives the RNN a better initialisation.
- Check that the data loaded correctly: load the .npz and plot a few trajectories.

### Symptom: Low MSE but spread ~ 0 (origin collapse)

**Likely cause:** The RNN achieves low loss by learning to "hold" whatever the teacher-forcing gave it, then slowly decaying to zero. This works because the standardised data has mean 0, and predicting 0 everywhere is not terrible for Group A trials.

**Try:**
- Use convergence weighting: `convergence_weight=5.0, convergence_steps=30`. This up-weights the first 30 autonomous steps, penalising the model more for failing to maintain the bump immediately after teacher-forcing ends.
- Ensure Group B trials are in the training set — they explicitly require the RNN to clean up perturbations, which fights the "decay to zero" tendency.
- Reduce alpha. Too-fast dynamics (high alpha) can cause the hidden state to jump around and eventually settle at the origin.

### Symptom: Uniformity ~ 0 (single fixed point)

**Likely cause:** The RNN learned a single bump angle and always converges to it regardless of initialisation. This can happen if the recurrent dynamics have one strongly stable fixed point.

**Try:**
- Train longer. The ring structure may emerge late in training (after epoch 500+).
- Lower the learning rate: `learning_rate=5e-4`. A high learning rate can lock the model into a single basin early on.
- Check the gradient norms. If they're tiny (< 0.01), the model is stuck. Try restarting with a different random seed.

### Symptom: Circularity ~ 0 (line, not circle)

**Likely cause:** The model learned two stable fixed points (or a few) arranged along a line in hidden space, not a circle. The uniformity might look okayish because the angles at these points span some range.

**Try:**
- Switch to the low-rank model. The rank-2 constraint structurally favours circular geometry.
- Increase the number of training epochs.
- Inspect the PCA plot — if you see 2-3 clusters, the model is on its way but hasn't achieved continuity yet. More training may help.

### Symptom: Generalization drift > 10 deg (polygon)

**Likely cause:** The model learned 36 discrete fixed points, one per training angle, not a continuous ring. This is a common failure mode because training only uses 36 discrete angles.

**Try:**
- Train longer. Discrete attractors are an intermediate state that sometimes melts into a continuous ring with more training.
- Reduce the learning rate for the final phase: use cosine annealing (`scheduler="cosine"`) which naturally reduces the LR toward the end.
- Add small noise to the hidden state during training (not yet implemented — this is a Milestone 3 idea, but you could try adding `h = h + 0.01 * torch.randn_like(h)` inside the model's forward loop during the autonomous phase).

### Symptom: Eigenvalues |lambda| > 1 (instability)

**Likely cause:** The recurrent dynamics are unstable. The RNN's outputs will blow up during long autonomous rollouts.

**Try:**
- Tighten gradient clipping: `clip_grad=0.5`.
- Add weight decay: `weight_decay=1e-4`.
- Lower alpha. Instability often comes from too-fast dynamics amplifying perturbations.

### General advice

- **Don't over-tune.** The plan explicitly warns against spending too long on Milestone 1. Once the ring appears (even imperfectly), move to partial observation.
- **Try both architectures.** The vanilla and low-rank models may behave very differently. The low-rank model has a structural prior that helps, but fewer parameters to work with.
- **Watch the loss curve, not just the final loss.** A model whose loss is still decreasing at epoch 1500 will benefit from more training. A model that flatlined at epoch 200 needs a different approach.
- **The evaluation takes ~60s** (the generalization test runs 36 simulator trials). Don't run it every epoch — run it after training completes, or every few hundred epochs during long runs.

---

## 7. Hyperparameters to explore

If the defaults don't work, here's the priority order for tuning:

| Parameter | Default | Range to try | Why |
|---|---|---|---|
| `alpha` | 0.5 | 0.1 - 0.8 | Controls RNN timescale. Too fast = instability. Too slow = can't track dynamics. |
| `learning_rate` | 1e-3 | 5e-4 - 3e-3 | Standard knob. Lower for stability, higher for speed. |
| `convergence_weight` | 1.0 | 1.0 - 10.0 | Up-weight early autonomous steps. Helps if the model decays to zero. |
| `convergence_steps` | 0 | 0, 20, 30, 50 | How many steps to up-weight. Set > 0 only if using convergence_weight > 1. |
| `k_min, k_max` | 5, 20 | (3, 15) to (10, 40) | Teacher-forcing window. Wider = easier for RNN but less autonomous training. |
| `n_epochs` | 1500 | 1000 - 3000 | More is generally better, with diminishing returns. |
| `model_type` | vanilla | vanilla, lowrank | Both worth trying. |

---

## 8. After Milestone 1: partial observation

Once you have a working ring (all three Milestone 1 criteria pass), the real experiment begins.

The question: **how much of the network do you need to observe to recover the ring?**

```python
for obs_frac in [1.0, 0.5, 0.25, 0.1]:
    result = train("data/ring_attractor_dataset.npz", TrainingConfig(
        observation_fraction=obs_frac,
        n_epochs=1500,
    ))
    report = full_evaluation(
        result['model'], "data/ring_attractor_dataset.npz",
        result['device'], result['val_idx'], result['observed_idx'],
    )
    print(f"k/N = {obs_frac}: milestone_1 = {report['milestone_1_pass']}")
```

When `observation_fraction < 1.0`, only a random subset of neurons is given to the RNN as input during teacher-forcing. The RNN still has to predict all N=100 neurons. This tests whether partial information about the bump (its shape, position) is enough for the RNN to infer the full state and learn the underlying dynamics.

**Expected result:** At some threshold (maybe k/N = 0.25 or 0.1), the ring breaks. The uniformity and circularity will drop, and the generalization drift will increase. Characterising this threshold and understanding *why* it breaks is the core scientific contribution of this project.

**What to plot:**
- Uniformity vs. k/N
- Circularity vs. k/N
- Mean drift vs. k/N
- PCA of converged states at each observation level (to see the ring degrade)

---

## 9. Reference: file-by-file guide

| File | What to do with it |
|---|---|
| `src/generate_dataset.py` | Run once. Only re-run if you change simulator parameters. |
| `src/train/models.py` | Read to understand the RNN architecture. Modify if you want to try different architectures. |
| `src/train/dataset.py` | Read to understand how teacher-forcing cutoff works. Rarely needs modification. |
| `src/train/training.py` | The main training loop. Modify `TrainingConfig` defaults or add new features here. |
| `src/train/evaluation.py` | All evaluation metrics. Run via `full_evaluation()` after training. Contains individual functions if you need just one metric. |
| `data/ring_attractor_dataset.npz` | Training data. Load with `np.load()`. Keys: `trajectories`, `trajectories_raw`, `groups`, `target_angles`, `neuron_angles`, `mean`, `std`. |
| `plans/revised_training_strategy.md` | The full theoretical background. Read Sections 1-2 for motivation, Section 4 for evaluation details, Section 6 for known limitations. |

---

## 10. Summary: what success looks like

You will know you succeeded when:

1. **`full_evaluation()` prints MILESTONE 1: PASS.**
2. **The PCA plot shows a smooth circle** with a rainbow colour gradient (angles distributed continuously, not clustered).
3. **The generalization drift is flat** near zero (not a sawtooth).
4. **The eigenvalue spectrum shows ~2 eigenvalues near |lambda| = 1** and all others strictly inside the unit circle.

At that point, the RNN has learned a genuine continuous ring attractor in its recurrent dynamics — the same mathematical structure as the teacher, but discovered purely from data. You can then ask the real question: how does this break down when you can only see part of the network?
