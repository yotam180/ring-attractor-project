#!/usr/bin/env python3
"""Training script for ring attractor RNN recovery (Milestone 1).

Uses the proven recipe:
- Cosine LR schedule (8e-4 → 1e-5)
- Noise annealing (0.08 → 0.002)
- Convergence weighting (5×, 30 steps)
- No augmentation (72 training angles provide sufficient coverage)
"""
import sys
import torch
from src.train import train, TrainingConfig, full_evaluation

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
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
    circular_shift_augment=False,
))

report = full_evaluation(
    result['model'], "data/ring_attractor_dataset.npz",
    result['device'], result['val_idx'], result['observed_idx'],
)

print(f"\nSeed {seed}: milestone_1 = {report['milestone_1_pass']}")
