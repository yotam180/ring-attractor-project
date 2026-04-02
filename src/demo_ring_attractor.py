"""
Interactive Ring Attractor Demo
===============================

Convert to notebook:  jupytext --to notebook src/demo_ring_attractor.py
Each "# %%" marks a new cell.
"""

# %%
# Setup
import numpy as np
import matplotlib.pyplot as plt
from src.ring_attractor import RingAttractor, SpikeProcessor
from src.ring_attractor.plotting import polar_snapshot

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 120

ring = RingAttractor()

# %%
# ## 1. Basic ring attractor: cue a bump, watch it persist
#
# We inject a Gaussian cue at θ=60° for 2000 steps, then let the network
# run autonomously.  The bump should lock in and persist indefinitely.

target = np.pi / 3  # 60°
res = ring.simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=42)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
order = np.argsort(res.angles)
ax1.imshow(res.rates[:, order].T, aspect="auto", origin="lower", cmap="inferno")
ax1.axvline(2000, color="cyan", ls="--", lw=1, label="cue off")
ax1.set_ylabel("Neuron"); ax1.legend(fontsize=8)
ax1.set_title("Ring Attractor: Cue at 60°, then autonomous")
ax2.plot(np.degrees(res.theta), lw=0.6, color="tab:blue")
ax2.axhline(60, color="red", ls="--", lw=1, label="target")
ax2.axvline(2000, color="cyan", ls="--", lw=1)
ax2.set_ylabel("θ (°)"); ax2.legend(fontsize=8)
ax3.plot(res.confidence, lw=0.8, color="tab:green")
ax3.axvline(2000, color="cyan", ls="--", lw=1)
ax3.set_ylabel("Confidence"); ax3.set_xlabel("Step"); ax3.set_ylim(0, 1)
plt.tight_layout(); plt.show()

# %%
# ## 2. The bump is a TRUE attractor: perturbation recovery

res1 = ring.simulate(T=5000, cue_angles=[0.0], cue_duration=2000, seed=10)
rng = np.random.default_rng(99)
perturbed = res1.rates[-1] + 0.5 * rng.standard_normal(ring.N)
res2 = ring.simulate(T=15000, cue_angles=None, seed=20, init_rates=perturbed)

fig, axes = plt.subplots(1, 5, figsize=(15, 3), subplot_kw={"projection": "polar"})
fig.suptitle("Perturbation Recovery", fontsize=13, y=1.05)
for ax, t in zip(axes, [0, 500, 2000, 5000, 14000]):
    polar_snapshot(ax, res2.rates[t], res2.angles, title=f"t={t}\nconf={res2.confidence[t]:.2f}")
plt.tight_layout(); plt.show()

# %%
# ## 3. Spontaneous bump formation (no cue!)

res_noise = ring.simulate(T=50000, cue_angles=None, seed=77, init_noise_scale=0.1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle("Spontaneous Bump Formation (no cue)", fontsize=13)
ax1.plot(res_noise.rates.max(axis=1), lw=0.5, color="tab:red"); ax1.set_ylabel("Peak rate")
ax2.plot(res_noise.confidence, lw=0.5, color="tab:green")
ax2.set_ylabel("Confidence"); ax2.set_xlabel("Step"); ax2.set_ylim(0, 1)
plt.tight_layout(); plt.show()

# %%
# ## 4. All 36 angles: the full ring

n_angles = 36
targets = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
decoded, confs = np.zeros(n_angles), np.zeros(n_angles)
for i, tgt in enumerate(targets):
    r = ring.simulate(T=7500, cue_angles=[tgt], cue_duration=2000, seed=i * 7)
    z = np.exp(1j * r.theta[5000:]).mean()
    decoded[i], confs[i] = np.angle(z), r.confidence[5000:].mean()

errors = np.abs(np.degrees(np.arctan2(np.sin(decoded - targets), np.cos(decoded - targets))))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Ring Coverage: 36 Cued Angles", fontsize=13)
ax1.scatter(np.degrees(targets), np.degrees(decoded), s=25, c="tab:blue", zorder=3)
ax1.plot([0, 360], [0, 360], "k--", lw=0.8); ax1.set_xlabel("Target (°)")
ax1.set_ylabel("Decoded (°)"); ax1.set_title("Angle accuracy"); ax1.set_aspect("equal")
ax2.bar(np.degrees(targets), errors, width=8, color="tab:orange")
ax2.set_xlabel("Target (°)"); ax2.set_ylabel("|Error| (°)")
ax2.set_title(f"Error: mean={errors.mean():.2f}°, max={errors.max():.2f}°"); ax2.set_ylim(0, 5)
plt.tight_layout(); plt.show()
print(f"Mean confidence: {confs.mean():.3f}  |  Mean |error|: {errors.mean():.2f}°")

# %%
# ## 5. Spike generation pipeline

res_sp = ring.simulate(T=7500, cue_angles=[np.pi / 4], cue_duration=2000, seed=42)
order = np.argsort(res_sp.angles)
data = SpikeProcessor().process(res_sp.rates, seed=42)

fig, axes = plt.subplots(2, 2, figsize=(14, 7))
fig.suptitle("Data Pipeline: Rates → Spikes → Binned → Smoothed", fontsize=13)
for ax, (d, title, cmap) in zip(axes.flat, [
    (res_sp.rates[:, order].T, "Raw rates", "inferno"),
    (data.spikes[:, order].T, "Poisson spikes", "gray_r"),
    (data.binned[:, order].T, "Binned (50 steps/bin)", "inferno"),
    (data.smoothed[:, order].T, "Smoothed (window=3)", "inferno"),
]):
    im = ax.imshow(d, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title); ax.set_ylabel("Neuron"); plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.show()

# %%
# ## 6. Key parameters
#
# - **N=100** neurons, cosine kernel W_ij = (J0 + J1 cos(θ_i − θ_j)) / N
# - **J0=−2, J1=4**: global inhibition + local excitation
# - **φ(x) = tanh(4 × ReLU(x))**: threshold + saturation (f-I curve)
# - **α=0.01**, **σ=0.1**, **Gaussian cue σ=20°**
# - Equilibrium: peak≈0.84, FWHM≈90°, confidence≈0.84, stable indefinitely
