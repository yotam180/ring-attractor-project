"""
Validation script for the ring attractor simulator.

Runs five scenarios and prints diagnostic results:
  1. Bump formation from cue      (θ should lock, confidence high)
  2. Long-term bump maintenance    (bump persists 100k steps after cue)
  3. Spontaneous bump from noise   (bump forms without any cue)
  4. Perturbation recovery         (noisy bump → clean bump restored)
  5. Multiple angles               (36 cues at different θ all persist)

Run:  python src/validate_simulator.py       (from project root)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.simulator import simulate, decode_theta

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _circ_stats(theta, confidence, label, start=0, end=None):
    """Print summary stats for a window."""
    c = confidence[start:end]
    t = theta[start:end]
    z = np.exp(1j * t)
    R = np.abs(z.mean())
    circ_std_deg = np.degrees(np.sqrt(max(-2 * np.log(R + 1e-12), 0)))
    print(f"  [{label}]  confidence: mean={c.mean():.3f}  min={c.min():.3f}  |  θ circ-std={circ_std_deg:.1f}°")
    return c.mean(), R


def test_bump_formation():
    """Scenario 1: Cue at θ=π/3 for 2000 steps, then free-run."""
    print("\n" + "=" * 65)
    print("TEST 1: Bump formation from cue")
    print("=" * 65)
    target = np.pi / 3
    res = simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=42)

    _circ_stats(res["theta"], res["confidence"], "during cue (0-2000)", 0, 2000)
    mean_conf, R = _circ_stats(res["theta"], res["confidence"], "free-run  (3000-7500)", 3000, 7500)

    # Decoded angle should be near target
    late_theta = res["theta"][3000:]
    z_mean = np.exp(1j * late_theta).mean()
    mean_angle = np.angle(z_mean)
    angle_err = np.abs(np.angle(np.exp(1j * (mean_angle - target))))
    print(f"  Target θ = {np.degrees(target):.1f}°  |  Decoded θ = {np.degrees(mean_angle):.1f}°  |  Error = {np.degrees(angle_err):.1f}°")
    print(f"  Peak rate at end: {res['rates'][-1].max():.4f}")

    # Bump width
    final_rates = res["rates"][-1]
    n_above_half = np.sum(final_rates > 0.5 * final_rates.max())
    fwhm = n_above_half * 360 / len(final_rates)
    print(f"  FWHM: {fwhm:.0f}°  ({n_above_half} neurons above half-max)")

    ok = mean_conf > 0.80 and angle_err < np.radians(10) and fwhm < 120
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_bump_maintenance():
    """Scenario 2: Form bump, then check it persists for 100k free-run steps."""
    print("\n" + "=" * 65)
    print("TEST 2: Long-term bump maintenance (100k steps free-run)")
    print("=" * 65)
    target = np.pi
    res = simulate(T=102000, cue_angles=[target], cue_duration=2000, seed=99)

    windows = [
        (2000, 10000, "0-8k free"),
        (10000, 50000, "8-48k free"),
        (50000, 102000, "48-100k free"),
    ]
    all_ok = True
    for s, e, label in windows:
        mc, R = _circ_stats(res["theta"], res["confidence"], label, s, e)
        if mc < 0.80:
            all_ok = False

    # Check peak rate is stable (not decaying to zero)
    peak_early = res["rates"][5000:10000].max(axis=1).mean()
    peak_late = res["rates"][90000:].max(axis=1).mean()
    print(f"  Peak rate: early={peak_early:.4f}  late={peak_late:.4f}  ratio={peak_late/peak_early:.2f}")
    if peak_late < 0.1 * peak_early:
        all_ok = False

    print(f"  → {PASS if all_ok else FAIL}")
    return all_ok


def test_noise_bump_formation():
    """Scenario 3: No cue — bump should form spontaneously from noise."""
    print("\n" + "=" * 65)
    print("TEST 3: Spontaneous bump formation from noise (no cue)")
    print("=" * 65)
    res = simulate(T=50000, cue_angles=None, sigma=0.1, seed=77, init_noise_scale=0.1)

    _circ_stats(res["theta"], res["confidence"], "early (0-5k)", 0, 5000)
    mean_conf_late, R = _circ_stats(res["theta"], res["confidence"], "late  (30k-50k)", 30000, 50000)
    print(f"  Peak rate at end: {res['rates'][-1].max():.4f}")

    ok = mean_conf_late > 0.75
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_perturbation_recovery():
    """Scenario 4: Form bump at θ=0, add large noise, check it recovers."""
    print("\n" + "=" * 65)
    print("TEST 4: Perturbation recovery (noisy bump → clean bump)")
    print("=" * 65)
    target = 0.0

    # Phase 1: form the bump
    res1 = simulate(T=5000, cue_angles=[target], cue_duration=2000, seed=123)
    clean_rates = res1["rates"][-1].copy()
    print(f"  Clean bump: peak_rate={clean_rates.max():.4f}  conf={res1['confidence'][-1]:.3f}")

    # Phase 2: perturb and let it recover
    rng = np.random.default_rng(456)
    noise_amp = 0.5  # Large perturbation
    perturbed = clean_rates + noise_amp * rng.standard_normal(clean_rates.shape)

    res2 = simulate(T=20000, cue_angles=None, sigma=0.1, seed=789, init_rates=perturbed)
    conf_start = res2["confidence"][:200].mean()
    conf_late = res2["confidence"][10000:].mean()
    print(f"  After perturbation:  conf={conf_start:.3f}")
    print(f"  After recovery:      conf={conf_late:.3f}")
    print(f"  Peak rate at end:    {res2['rates'][-1].max():.4f}")

    ok = conf_late > 0.80
    print(f"  → {PASS if ok else FAIL}")
    return ok


def test_multiple_angles():
    """Scenario 5: Cue at 36 different angles — all should persist with correct θ."""
    print("\n" + "=" * 65)
    print("TEST 5: Multiple angles (36 cues, all should persist)")
    print("=" * 65)
    angles_target = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    errors = []
    confs = []
    for i, target in enumerate(angles_target):
        res = simulate(T=7500, cue_angles=[target], cue_duration=2000, seed=i * 10)
        late_theta = res["theta"][4000:]
        z_mean = np.exp(1j * late_theta).mean()
        mean_angle = np.angle(z_mean)
        err = np.abs(np.angle(np.exp(1j * (mean_angle - target))))
        errors.append(np.degrees(err))
        confs.append(res["confidence"][4000:].mean())

    errors = np.array(errors)
    confs = np.array(confs)
    print(f"  Angle errors:  mean={errors.mean():.1f}°  max={errors.max():.1f}°  std={errors.std():.1f}°")
    print(f"  Confidences:   mean={confs.mean():.3f}  min={confs.min():.3f}")

    ok = errors.mean() < 5 and confs.mean() > 0.80
    print(f"  → {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    results = {}
    results["Bump formation"] = test_bump_formation()
    results["Bump maintenance (100k)"] = test_bump_maintenance()
    results["Noise bump formation"] = test_noise_bump_formation()
    results["Perturbation recovery"] = test_perturbation_recovery()
    results["Multiple angles (36)"] = test_multiple_angles()

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        if not ok:
            all_pass = False
        print(f"  {status}  {name}")
    print("=" * 65)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 65)
