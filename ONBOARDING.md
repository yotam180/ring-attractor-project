# Project Onboarding Guide

**Ring Attractor Recovery Under Partial Observation**

This document brings you up to speed on the entire project — what we're doing, why, what we found, and what you need to read. Start here, then follow the reading order below.

---

## 1. The Research Question (2 minutes)

**Can a recurrent neural network recover the correct dynamical mechanism of a neural circuit from incomplete observations?**

Specifically: a "ring attractor" is a neural circuit where activity forms a bump that can sit at any angle on a circle (like a compass needle). Real experiments only record a fraction of neurons for a limited time. We ask:

- How many neurons do you need to observe? (the **k/N axis**)
- How long do you need to observe them? (the **T axis**)
- Below what thresholds does the RNN learn the wrong mechanism — even though its predictions still look good?

This matters because modern neuroscience increasingly uses RNNs to infer mechanisms from data. If partial observation systematically biases what the RNN learns, conclusions drawn from limited recordings could be wrong.

---

## 2. What We Found (5 minutes)

### The k/N axis: neuron dropout

We trained RNNs observing {100%, 75%, 50%, 25%, 20%, 15%, 10%} of 100 neurons. Results:

| Neurons observed | Pass rate | Verdict |
|---|---|---|
| 100% – 20% | 100% (or near) | **Ring reliably recovered** |
| 15% | 50% | **Threshold zone** |
| 10% | 40% | **Unreliable** |

**The cliff is at ~15-20 neurons.** Above this, the ring is recovered perfectly. Below, the RNN learns the right shape but not the right dynamics — bumps drift toward discrete training angles instead of staying put.

### The T axis: observation time

We trained RNNs on trials of {200, 100, 50, 25} time bins (~{100, 50, 25, 12.5} seconds each). Results:

| Trial length | Pass rate | Verdict |
|---|---|---|
| 200 or 100 bins | 100% | **Ring recovered** |
| 50 bins | 0% | **Fails** |
| 25 bins | 0% | **Fails** |

**The cliff is at ~50-100 bins.** Below 50 bins, the model sees too little autonomous dynamics per trial to learn continuous neutral stability.

### The 2D picture

The two thresholds are **independent** — you need enough neurons AND enough time. Observing more neurons doesn't compensate for short recordings, and longer recordings don't compensate for too few neurons. The threshold surface is L-shaped.

### The failure mode

When the RNN fails, it doesn't collapse to nothing. It still learns a ring-shaped manifold (uniformity and circularity pass). But the ring has **discrete attracting regions** at the training angles — like a 72-sided polygon instead of a smooth circle. The model learns what bumps look like at each angle, but not that bumps should be neutrally stable everywhere.

---

## 3. The Technical Pipeline (10 minutes)

### Teacher → Student setup

```
Teacher (ring attractor simulator)
  → generates bumps at 72 angles
  → Poisson spikes → binning → smoothing
  → 144 trials of standardized firing rates

Student (vanilla RNN)
  → sees first K steps of each trial (teacher-forcing)
  → must predict the remaining steps autonomously (no input)
  → loss only on autonomous phase
```

The key insight: by zeroing the input after K steps, we force the RNN to embed attractor dynamics in its recurrent weights. It can't cheat by routing information through the input pathway.

### The simulator

- 100 neurons on a ring with cosine connectivity: W_ij = (J0 + J1·cos(θ_i - θ_j))/N
- Nonlinearity: φ(x) = tanh(4·ReLU(x)) — threshold + saturation, like a biological f-I curve
- Produces stable bumps with FWHM=112°, confidence=0.84
- Two trial types: Group A (bump maintenance), Group B (perturbation recovery)

### The student RNN

- h[t] = (1-α)h[t-1] + α·tanh(W_hh·h[t-1] + W_xh·x[t] + b)
- 100 hidden units, ~30k parameters
- Trained with: cosine LR schedule, noise annealing, convergence weighting
- No data augmentation (72 training angles provide sufficient coverage)

### Evaluation metrics

1. **Uniformity** (> 0.8): Do converged autonomous states cover all 360°?
2. **Circularity** (> 0.7): Is the manifold circular in PCA?
3. **Drift** (< 5°): Can the ring hold intermediate angles it never trained on?

All three must pass for Milestone 1. Drift is the most sensitive metric — it fails first.

---

## 4. Background Reading

### Required (read in this order)

1. **`old/plans/project_overview.md`** — Original project proposal. Frames the two-phase structure (simulation + biology), explains why partial observation matters, defines the axes of investigation.

2. **`old/plans/project_essence.md`** — Clarifies how the simulation and real data relate. Answers "why do we need both?" Explains the teacher-student relationship and what the real HD cell recordings contribute.

3. **`plans/revised_training_strategy.md`** — The technical blueprint. Defines the RNN architecture, training procedure (teacher-forcing → autonomous), all evaluation metrics, and milestones. **Section 2 (Key Concepts)** is essential — it explains what a ring attractor IS, what makes it different from an identity mapping, and what the eigenvalue signature looks like. **Appendix A** has the simulator parameters (note: some values are outdated — the nonlinearity changed from ReLU to tanh(4·ReLU), see below).

4. **`plans/01_ring_simulation.md`** — Build log for the simulator. Documents why ReLU failed (with analytical proof), how we arrived at tanh(4·ReLU), and the parameter tuning process. Read this to understand the nonlinearity choice.

5. **`plans/04_partial_observation_results.md`** — The main experimental results. Sections 1-5 cover the k/N sweep, Section 8 covers the T-axis sweep. This is what we'll be writing up.

### Recommended (fills in details)

6. **`plans/01_relu_instability_proof.md`** — Mathematical proof that ReLU + cosine kernel cannot support stable bumps. Short and educational if you want to understand the analysis.

7. **`plans/02_training.md`** — Detailed training guide with hyperparameter explanations and troubleshooting. Read if you want to understand why we use convergence weighting, noise annealing, etc.

8. **`plans/03_training_second_attempt.md`** — Documents the three code fixes we made after a critical review: removing divisive normalization, extracting shared defaults, fixing the evaluation bug. Also documents the augmentation confound discovery and switch from 36 to 72 training angles.

9. **`plans/03_training_execution.md`** — Detailed execution log: attempt-by-attempt results, LR scheduler comparison, augmentation investigation, T_auto sensitivity test.

### External references

10. **Qian, Zavatone-Veth, Ruben & Pehlevan (2024). "Partial observation can induce mechanistic mismatches in data-constrained models of neural dynamics." NeurIPS 2024.** — The paper most directly related to ours. They study the same question (partial observation → wrong mechanism) but for **line attractors**, not ring attractors. Our extension to ring attractors (topologically S¹ instead of ℝ¹) and our 2D (k/N, T) threshold surface are the novel contributions. Read their abstract and introduction to understand the framing. See `old/plans/make_this_a_paper.md` for a detailed comparison.

11. **Durstewitz group — shPLRNN tutorial.** Referenced in the training strategy as the starting point for RNN training on dynamical systems. Their approach (teacher-forcing with BPTT on autonomous dynamics) influenced our training procedure. The key difference: they work with chaotic attractors (Lorenz63) where one long trajectory suffices; our ring attractor requires multiple trials at different angles.

12. **DANDI:000939 dataset** — The real HD cell recordings we may use for Phase 5 (biological validation). 31 sessions, 21-117 HD cells each. See `plans/dataset_000939_exploration.md` for exploration notes. Not needed for the simulation paper, but important context for the broader project vision.

---

## 5. History of Technical Decisions

These are the non-obvious choices that shaped the project. Understanding WHY we made them is important for the write-up.

| Decision | Why | Where documented |
|---|---|---|
| tanh(4·ReLU) instead of ReLU | ReLU bumps dissolve (proven analytically). Plain tanh gives wide plateaus. tanh(4·ReLU) gives peaked bumps. | `plans/01_ring_simulation.md`, `plans/01_relu_instability_proof.md` |
| Removed divisive normalization | Student RNN can't replicate nonlocal gain control. Creates unnecessary teacher-student mismatch. Bumps are unchanged without it. | `plans/03_training_second_attempt.md` Section 1.1 |
| 72 training angles (not 36) | Circular shift augmentation is incompatible with partial observation. 72 angles provide sufficient angular coverage without augmentation. | `plans/03_training_execution.md` Phase 2-4 |
| Cosine LR schedule (not plateau) | Plateau scheduler kills LR by epoch 1050, wasting 80% of training. Cosine maintains meaningful learning through epoch 4000+. | `plans/03_training_execution.md` Phase 1 |
| Loss only on autonomous phase | Prevents the RNN from learning an identity mapping. Forces attractor dynamics into recurrent weights. | `plans/revised_training_strategy.md` Section 1.2-1.3 |
| Noise annealing (0.08 → 0.002) | Prevents discrete attractor basins early in training. Reduces to near-zero for fine-tuning. | `plans/02_training.md` |
| Convergence weighting (5×, 30 steps) | Up-weights the transient after teacher-forcing ends. Without it, steady-state maintenance dominates the loss gradient. | `plans/02_training.md` |

---

## 6. Codebase Quick Reference

See `INDEX.md` for the complete file-by-file map. The essential files:

| What | Where |
|---|---|
| Simulator | `src/ring_attractor/network.py` |
| Spike pipeline | `src/ring_attractor/spiking.py` |
| All default parameters | `src/ring_attractor/defaults.py` |
| Dataset generation | `src/generate_dataset.py` |
| RNN models | `src/train/models.py` |
| Training loop | `src/train/training.py` |
| All evaluation metrics | `src/train/evaluation.py` |
| k/N sweep runner | `run_sweep.py` + `run_sweep_extra.py` |
| T sweep runner | `run_sweep_T.py` |
| k/N sweep figures | `visualize_sweep.py` → `figs/04_partial_observation/` |
| T sweep figures | `visualize_sweep_T.py` → `figs/04_observation_time/` |
| All results (k/N) | `data/sweep_results/summary.json` |
| All results (T) | `data/sweep_results_T/summary.json` |
| Model checkpoints | `checkpoints/sweep_*/best.pt` + `config.json` |

---

## 7. Suggested Reading Schedule

**Day 1 (conceptual foundation):**
- This document (done!)
- `old/plans/project_overview.md` — the big picture
- `old/plans/project_essence.md` — how simulation and biology relate
- `plans/revised_training_strategy.md` — Section 2 (Key Concepts) and Section 4 (Evaluation)

**Day 2 (technical details):**
- `plans/01_ring_simulation.md` — simulator design and failures
- `plans/02_training.md` — training approach
- `plans/03_training_second_attempt.md` — code fixes and augmentation confound
- Skim Qian et al. (2024) abstract + intro

**Day 3 (results):**
- `plans/04_partial_observation_results.md` — all experimental findings
- Look at the figures in `figs/04_partial_observation/` and `figs/04_observation_time/`
- `INDEX.md` — verify you know where everything lives

After that, you'll have the same understanding of the project as I do and we can start writing together.
