"""
Validation script for the ring attractor simulator.

Runs five scenarios and prints diagnostic results:
  1. Bump formation from cue
  2. Long-term bump maintenance  (100k steps)
  3. Spontaneous bump from noise
  4. Perturbation recovery
  5. Multiple angles  (36 cues)

Run:  python -m src.validate_simulator   (from project root)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.ring_attractor import RingAttractor

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# Default network used by every test.
RING = RingAttractor()


def _circ_stats(theta, confidence, label, start=0, end=None):
    c = confidence[start:end]
    t = theta[start:end]
    R = np.abs(np.exp(1j * t).mean())
    circ_std = np.degrees(np.sqrt(max(-2 * np.log(R + 1e-12), 0)))
    print(
        f"  [{label}]  confidence: mean={c.mean():.3f}  min={c.min():.3f}"
        f"  |  θ circ-std={circ_std:.1f}°"
    )
    return c.mean(), R


# -------------------------------------------------------------------


def test_bump_formation():
    print("\n" + "=" * 65)
    print("TEST 1: Bump formation from cue")
    print("=" * 65)
    target = np.pi / 3
    res = RING.simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=42)

    _circ_stats(res.theta, res.confidence, "during cue (0-2000)", 0, 2000)
    mean_conf, _ = _circ_stats(res.theta, res.confidence, "free-run  (3000-7500)", 3000)

    z_mean = np.exp(1j * res.theta[3000:]).mean()
    angle_err = np.abs(np.angle(np.exp(1j * (np.angle(z_mean) - target))))
    fwhm = np.sum(res.rates[-1] > 0.5 * res.rates[-1].max()) * 360 / RING.N
    print(
        f"  Target θ = {np.degrees(target):.1f}°  |  Decoded θ = {np.degrees(np.angle(z_mean)):.1f}°"
        f"  |  Error = {np.degrees(angle_err):.1f}°"
    )
    print(f"  Peak rate: {res.rates[-1].max():.4f}  |  FWHM: {fwhm:.0f}°")

    ok = mean_conf > 0.80 and angle_err < np.radians(10) and fwhm < 120
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_bump_maintenance():
    print("\n" + "=" * 65)
    print("TEST 2: Long-term bump maintenance (100k steps free-run)")
    print("=" * 65)
    res = RING.simulate(T=102000, cue_angles=[np.pi], cue_duration=2000, seed=99)

    all_ok = True
    for s, e, label in [(2000, 10000, "0-8k"), (10000, 50000, "8-48k"), (50000, 102000, "48-100k")]:
        mc, _ = _circ_stats(res.theta, res.confidence, label, s, e)
        if mc < 0.80:
            all_ok = False

    early = res.rates[5000:10000].max(axis=1).mean()
    late = res.rates[90000:].max(axis=1).mean()
    print(f"  Peak rate: early={early:.4f}  late={late:.4f}  ratio={late / early:.2f}")
    if late < 0.1 * early:
        all_ok = False

    print(f"  → {PASS if all_ok else FAIL}")
    return all_ok


def test_noise_bump_formation():
    print("\n" + "=" * 65)
    print("TEST 3: Spontaneous bump formation from noise (no cue)")
    print("=" * 65)
    res = RING.simulate(T=50000, cue_angles=None, seed=77, init_noise_scale=0.1)

    _circ_stats(res.theta, res.confidence, "early (0-5k)", 0, 5000)
    mc, _ = _circ_stats(res.theta, res.confidence, "late  (30k-50k)", 30000)
    print(f"  Peak rate at end: {res.rates[-1].max():.4f}")

    ok = mc > 0.75
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_perturbation_recovery():
    print("\n" + "=" * 65)
    print("TEST 4: Perturbation recovery (noisy bump → clean bump)")
    print("=" * 65)
    res1 = RING.simulate(T=5000, cue_angles=[0.0], cue_duration=2000, seed=123)
    clean = res1.rates[-1].copy()
    print(f"  Clean bump: peak={clean.max():.4f}  conf={res1.confidence[-1]:.3f}")

    rng = np.random.default_rng(456)
    perturbed = clean + 0.5 * rng.standard_normal(clean.shape)

    res2 = RING.simulate(T=20000, cue_angles=None, seed=789, init_rates=perturbed)
    print(f"  After perturbation:  conf={res2.confidence[:200].mean():.3f}")
    print(f"  After recovery:      conf={res2.confidence[10000:].mean():.3f}")
    print(f"  Peak rate at end:    {res2.rates[-1].max():.4f}")

    ok = res2.confidence[10000:].mean() > 0.80
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_multiple_angles():
    print("\n" + "=" * 65)
    print("TEST 5: Multiple angles (36 cues, all should persist)")
    print("=" * 65)
    targets = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    errors, confs = [], []
    for i, tgt in enumerate(targets):
        res = RING.simulate(T=7500, cue_angles=[tgt], cue_duration=2000, seed=i * 10)
        z = np.exp(1j * res.theta[4000:]).mean()
        errors.append(np.degrees(np.abs(np.angle(np.exp(1j * (np.angle(z) - tgt))))))
        confs.append(res.confidence[4000:].mean())

    errors, confs = np.array(errors), np.array(confs)
    print(f"  Angle errors:  mean={errors.mean():.1f}°  max={errors.max():.1f}°")
    print(f"  Confidences:   mean={confs.mean():.3f}  min={confs.min():.3f}")

    ok = errors.mean() < 5 and confs.mean() > 0.80
    print(f"  → {PASS if ok else FAIL}")
    return ok


# -------------------------------------------------------------------

if __name__ == "__main__":
    tests = {
        "Bump formation": test_bump_formation,
        "Bump maintenance (100k)": test_bump_maintenance,
        "Noise bump formation": test_noise_bump_formation,
        "Perturbation recovery": test_perturbation_recovery,
        "Multiple angles (36)": test_multiple_angles,
    }
    results = {name: fn() for name, fn in tests.items()}

    print("\n" + "=" * 65 + "\nSUMMARY\n" + "=" * 65)
    all_pass = True
    for name, ok in results.items():
        if not ok:
            all_pass = False
        print(f"  {PASS if ok else FAIL}  {name}")
    print("=" * 65)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 65)
