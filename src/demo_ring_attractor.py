"""
Interactive Ring Attractor Demo
===============================

This script is designed to be converted to a Jupyter notebook:
    jupytext --to notebook src/demo_ring_attractor.py

Or copy-paste individual cells into a notebook.
Each "# %%" marks a new cell.
"""

# %%
# Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ""))
import numpy as np
import matplotlib.pyplot as plt
from src.simulator import simulate, decode_theta, generate_spikes, bin_spikes, smooth_bins

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 120

# %%
# ## 1. Basic ring attractor: cue a bump, watch it persist
#
# We inject a cosine cue at θ=60° for 2000 steps, then let the network
# run autonomously.  The bump should lock in and persist indefinitely.

target = np.pi / 3  # 60°
res = simulate(T=7500, cue_angles=[target], cue_duration=2000, sigma=0.1, seed=42)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

# Heatmap of firing rates
order = np.argsort(res["angles"])
ax1.imshow(res["rates"][:, order].T, aspect="auto", origin="lower", cmap="inferno")
ax1.axvline(2000, color="cyan", ls="--", lw=1, label="cue off")
ax1.set_ylabel("Neuron"); ax1.legend(fontsize=8)
ax1.set_title("Ring Attractor: Cue at 60°, then autonomous")

# Decoded theta
ax2.plot(np.degrees(res["theta"]), lw=0.6, color="tab:blue")
ax2.axhline(60, color="red", ls="--", lw=1, label="target")
ax2.axvline(2000, color="cyan", ls="--", lw=1)
ax2.set_ylabel("θ (°)"); ax2.legend(fontsize=8)

# Confidence
ax3.plot(res["confidence"], lw=0.8, color="tab:green")
ax3.axvline(2000, color="cyan", ls="--", lw=1)
ax3.set_ylabel("Confidence"); ax3.set_xlabel("Step")
ax3.set_ylim(0, 1)
plt.tight_layout(); plt.show()

# %%
# ## 2. The bump is a TRUE attractor: perturbation recovery
#
# We form a clean bump, then blast it with large Gaussian noise.
# The attractor dynamics clean up the noise and restore the bump.

res1 = simulate(T=5000, cue_angles=[0.0], cue_duration=2000, seed=10)
clean = res1["rates"][-1].copy()

rng = np.random.default_rng(99)
perturbed = clean + 0.5 * rng.standard_normal(clean.shape)

res2 = simulate(T=15000, cue_angles=None, sigma=0.1, seed=20, init_rates=perturbed)

fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw={"projection": "polar"})
fig.suptitle("Perturbation Recovery", fontsize=13, y=1.05)
for ax, t in zip(axes, [0, 500, 2000, 5000, 14000]):
    width = 2 * np.pi / 100
    r = np.maximum(res2["rates"][t], 0)
    colors = plt.cm.inferno(r / (r.max() + 1e-9))
    ax.bar(res2["angles"], r, width=width, color=colors, edgecolor="none")
    ax.set_title(f"t={t}\nconf={res2['confidence'][t]:.2f}", fontsize=9, pad=12)
    ax.set_yticks([]); ax.set_xticks([])
plt.tight_layout(); plt.show()

# %%
# ## 3. Spontaneous bump formation (no cue!)
#
# The ring attractor is UNSTABLE at the origin.  Starting from tiny noise,
# a bump emerges spontaneously.  The angle is random (determined by noise).

res_noise = simulate(T=50000, cue_angles=None, sigma=0.1, seed=77, init_noise_scale=0.1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle("Spontaneous Bump Formation (no cue)", fontsize=13)

ax1.plot(res_noise["rates"].max(axis=1), lw=0.5, color="tab:red")
ax1.set_ylabel("Peak rate")

ax2.plot(res_noise["confidence"], lw=0.5, color="tab:green")
ax2.set_ylabel("Confidence"); ax2.set_xlabel("Step")
ax2.set_ylim(0, 1)
plt.tight_layout(); plt.show()

# %%
# ## 4. All 36 angles: the full ring
#
# We cue at 36 evenly-spaced angles and verify each one produces a
# stable bump at the correct location.

n_angles = 36
targets = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
decoded = np.zeros(n_angles)
confs = np.zeros(n_angles)

for i, tgt in enumerate(targets):
    r = simulate(T=7500, cue_angles=[tgt], cue_duration=2000, seed=i * 7)
    z = np.exp(1j * r["theta"][5000:]).mean()
    decoded[i] = np.angle(z)
    confs[i] = r["confidence"][5000:].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Ring Coverage: 36 Cued Angles", fontsize=13)

ax1.scatter(np.degrees(targets), np.degrees(decoded), s=25, c="tab:blue", zorder=3)
ax1.plot([0, 360], [0, 360], "k--", lw=0.8)
ax1.set_xlabel("Target (°)"); ax1.set_ylabel("Decoded (°)")
ax1.set_title("Angle accuracy")
ax1.set_aspect("equal")

errors = np.abs(np.degrees(np.arctan2(np.sin(decoded - targets), np.cos(decoded - targets))))
ax2.bar(np.degrees(targets), errors, width=8, color="tab:orange")
ax2.set_xlabel("Target (°)"); ax2.set_ylabel("|Error| (°)")
ax2.set_title(f"Error: mean={errors.mean():.2f}°, max={errors.max():.2f}°")
ax2.set_ylim(0, 5)
plt.tight_layout(); plt.show()

print(f"Mean confidence: {confs.mean():.3f}")
print(f"Mean |error|: {errors.mean():.2f}°")

# %%
# ## 5. Spike generation pipeline
#
# For training data, we convert continuous rates to Poisson spikes,
# then bin and smooth them.

trial = simulate(T=7500, cue_angles=[np.pi/4], cue_duration=2000, sigma=0.1, seed=42)
order = np.argsort(trial["angles"])

spikes = generate_spikes(trial["rates"], dt=0.01, rate_scale=100, seed=42)
binned = bin_spikes(spikes, bin_factor=50)
smoothed = smooth_bins(binned, window=3)

fig, axes = plt.subplots(2, 2, figsize=(14, 7))
fig.suptitle("Data Pipeline: Rates → Spikes → Binned → Smoothed", fontsize=13)

for ax, (data, title, cmap) in zip(axes.flat, [
    (trial["rates"][:, order].T, "Raw rates", "inferno"),
    (spikes[:, order].T, "Poisson spikes", "gray_r"),
    (binned[:, order].T, "Binned (50 steps/bin)", "inferno"),
    (smoothed[:, order].T, "Smoothed (window=3)", "inferno"),
]):
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title); ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.show()

# %%
# ## 6. Key parameters
#
# The simulator uses:
# - **N=100** neurons with cosine connectivity W_ij = (J0 + J1 cos(θ_i - θ_j)) / N
# - **J0=-2, J1=4**: global inhibition + local excitation
# - **φ(x) = tanh(4 × ReLU(x))** nonlinearity — threshold + saturation
#   (plain ReLU has no stable equilibrium; plain tanh gives too-wide bumps)
# - **α=0.01** leak rate (= dt/τ)
# - **σ=0.1** noise
#
# Equilibrium bump has:
# - Peak rate ≈ 0.84
# - FWHM ≈ 90° (quarter of the ring — a proper peaked bump)
# - Confidence ≈ 0.84
# - Angle error < 2°
# - Stable indefinitely (tested to 100k steps)
