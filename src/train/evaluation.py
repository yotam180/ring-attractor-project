"""
Evaluation metrics for ring attractor RNN recovery.

Sections correspond to revised_training_strategy.md, Section 4:
  4.1  Predictive:        MSE, circular angle error
  4.2  Fixed-point:       spread, uniformity, circularity  (the core test)
  4.3  Generalisation:    intermediate-angle drift  (ring vs. polygon)
  4.4  Eigenvalues:       Jacobian spectrum at fixed points
  4.5  Singular values:   SVD of W_hh  (low-rank only)

Milestone 1 exit criteria:
  uniformity > 0.8,  circularity > 0.7,  mean |drift| < 5 deg
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ── Result containers ────────────────────────────────────────────────────


@dataclass
class PredictionMetrics:
    mse: float
    angle_error_deg: float


@dataclass
class RingScore:
    spread: float          # mean ||h_final||
    uniformity: float      # 1 - |mean(exp(i*theta))|
    circularity: float     # min(std)/max(std) of PC1,PC2
    h_final: np.ndarray    # (M, H)
    theta_final: np.ndarray  # (M,)  decoded angles
    pca_proj: np.ndarray   # (M, 2)  for visualisation


@dataclass
class GeneralizationResult:
    test_angles: np.ndarray     # (S,) rad
    final_angles: np.ndarray    # (S,) rad
    drift_deg: np.ndarray       # (S,) signed drift in degrees
    mean_abs_drift_deg: float


@dataclass
class EigenvalueResult:
    eigenvalues: np.ndarray     # complex, shape (H,)
    magnitudes: np.ndarray      # |lambda|, shape (H,)
    n_near_unit: int            # count with 0.95 < |lambda| < 1.05
    max_magnitude: float
    fp_residual: float          # ||f(h*) - h*||  (smaller = better fixed point)


# ── Angle decoding helpers ───────────────────────────────────────────────


def _decode_angles(y: np.ndarray, neuron_angles: np.ndarray) -> np.ndarray:
    """Population-vector decode from (*, N) rate array.  Returns angles."""
    z = y @ np.exp(1j * neuron_angles)
    return np.angle(z)


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise absolute circular distance (radians)."""
    return np.abs(np.angle(np.exp(1j * (a - b))))


def _destandardise(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """y * std + mean,  broadcasts over leading dims."""
    return y * std + mean


# ── 4.1  Predictive metrics ─────────────────────────────────────────────


def evaluate_predictions(
    model: torch.nn.Module,
    trajectories: np.ndarray,
    neuron_angles: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    k_eval: int = 12,
) -> PredictionMetrics:
    """
    Compute MSE and circular angle error on a set of trials.

    Uses a fixed teacher-forcing length *k_eval* for reproducibility.
    Loss is on the autonomous phase only (steps k_eval … T).
    """
    n_trials, T, N = trajectories.shape
    input_dim = model.input_dim

    # Build input: first k_eval steps from trajectory, zeros after
    x = np.zeros((n_trials, T, input_dim), dtype=np.float32)
    x[:, :k_eval, :] = trajectories[:, :k_eval, :input_dim]

    x_t = torch.from_numpy(x).to(device)
    y_true = torch.from_numpy(trajectories.astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        y_pred, _ = model(x_t)

    # MSE on autonomous phase only
    auto = slice(k_eval, T)
    mse = ((y_pred[:, auto] - y_true[:, auto]) ** 2).mean().item()

    # Circular angle error (de-standardise first)
    yp = _destandardise(y_pred[:, auto].cpu().numpy(), norm_mean, norm_std)
    yt = _destandardise(y_true[:, auto].cpu().numpy(), norm_mean, norm_std)

    theta_pred = _decode_angles(yp.reshape(-1, N), neuron_angles)
    theta_true = _decode_angles(yt.reshape(-1, N), neuron_angles)
    err = float(np.degrees(_circular_error(theta_pred, theta_true).mean()))

    return PredictionMetrics(mse=mse, angle_error_deg=err)


# ── 4.2  Autonomous fixed-point analysis ─────────────────────────────────


def autonomous_fixed_points(
    model: torch.nn.Module,
    neuron_angles: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    M: int = 300,
    T_auto: int = 500,
    sigma_init: float = 1.0,
) -> RingScore:
    """
    The primary evaluation.

    1. Initialise M random hidden states  h0 ~ N(0, sigma_init^2 I)
    2. Run autonomously for T_auto steps
    3. Decode converged angles, compute spread / uniformity / circularity

    A successful ring shows: spread >> 0, uniformity ~ 1, circularity ~ 1.
    """
    model.eval()
    H = model.hidden_dim
    h0 = torch.randn(M, H, device=device) * sigma_init

    with torch.no_grad():
        _, h_all = model.run_autonomous(h0, T_auto)

    h_final = h_all[:, -1, :].cpu().numpy()                   # (M, H)

    # Decode angles through readout
    with torch.no_grad():
        y_final = model.W_hy(h_all[:, -1, :]).cpu().numpy()   # (M, N)
    y_raw = _destandardise(y_final, norm_mean, norm_std)
    theta = _decode_angles(y_raw, neuron_angles)               # (M,)

    # ── Spread: mean L2 norm of converged hidden states
    spread = float(np.linalg.norm(h_final, axis=1).mean())

    # ── Uniformity: 1 - |mean(exp(i*theta))|
    #    1 = uniform around ring,  0 = all same angle
    uniformity = float(np.clip(1.0 - np.abs(np.exp(1j * theta).mean()), 0, 1))

    # ── Circularity: PCA (via SVD), ratio of std along PC1 vs PC2
    h_centered = h_final - h_final.mean(axis=0)
    _, _, Vt = np.linalg.svd(h_centered, full_matrices=False)
    proj = h_centered @ Vt[:2].T                               # (M, 2)
    s1, s2 = proj[:, 0].std(), proj[:, 1].std()
    circularity = float(min(s1, s2) / max(s1, s2)) if max(s1, s2) > 1e-8 else 0.0

    return RingScore(
        spread=spread,
        uniformity=uniformity,
        circularity=circularity,
        h_final=h_final,
        theta_final=theta,
        pca_proj=proj,
    )


# ── 4.3  Generalisation test (ring vs. discrete points) ─────────────────


def generalization_test(
    model: torch.nn.Module,
    neuron_angles: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    n_test: int = 36,
    k_teacher: int = 15,
    T_gen: int = 500,
) -> GeneralizationResult:
    """
    Test intermediate angles that the RNN never saw during training.

    For each test angle theta_test (midway between training angles):
      1. Generate a converged bump via the teacher simulator
      2. Teacher-force the RNN for k_teacher bins
      3. Run autonomously for T_gen bins
      4. Decode the final angle

    A true ring: drift ~ 0.   Discrete fixed points: sawtooth drift.
    """
    from src.ring_attractor import RingAttractor, SpikeProcessor

    # Test angles: 5, 15, 25, ... (midway between training angles 0, 10, 20, ...)
    train_spacing = 2 * np.pi / n_test
    test_angles = np.linspace(0, 2 * np.pi, n_test, endpoint=False) + train_spacing / 2

    ring = RingAttractor()  # defaults match dataset generation
    sp = SpikeProcessor()

    T_cue = 2000
    T_settle = 500
    T_record = (k_teacher + 5) * sp.bin_factor  # a few extra bins for safety

    model.eval()
    input_dim = model.input_dim
    T_total_seq = k_teacher + T_gen

    final_angles = np.zeros(n_test)

    for i, theta_test in enumerate(test_angles):
        # Generate converged bump at this angle
        T_sim = T_cue + T_settle + T_record
        res = ring.simulate(
            T=T_sim, cue_angles=[theta_test],
            cue_duration=T_cue, seed=10000 + i,
        )
        rec_rates = res.rates[T_cue + T_settle:]
        data = sp.process(rec_rates, seed=20000 + i)
        smoothed = data.smoothed  # (T_bin, N)

        # Standardise with the same params used for training data
        standardised = (smoothed - norm_mean) / norm_std

        # Build input: teacher-force for k_teacher steps, zeros after
        x = np.zeros((1, T_total_seq, input_dim), dtype=np.float32)
        x[0, :k_teacher, :] = standardised[:k_teacher, :input_dim]
        x_t = torch.from_numpy(x).to(device)

        # Build target placeholder (we only need the output)
        with torch.no_grad():
            y_pred, _ = model(x_t)

        # Decode angle from the last output step
        y_last = y_pred[0, -1].cpu().numpy()  # (N,)
        y_raw = _destandardise(y_last, norm_mean, norm_std)
        final_angles[i] = _decode_angles(y_raw[np.newaxis, :], neuron_angles)[0]

    drift = np.degrees(np.angle(np.exp(1j * (final_angles - test_angles))))

    return GeneralizationResult(
        test_angles=test_angles,
        final_angles=final_angles,
        drift_deg=drift,
        mean_abs_drift_deg=float(np.abs(drift).mean()),
    )


# ── 4.4  Eigenvalue analysis ────────────────────────────────────────────


def eigenvalue_analysis(
    model: torch.nn.Module,
    h_star: np.ndarray,
) -> EigenvalueResult:
    """
    Compute eigenvalues of the Jacobian at a fixed point h*.

    J = (1-alpha)I + alpha * diag(1 - tanh^2(W @ h* + b)) * W

    where W is the recurrent weight matrix.

    Args:
        model: trained RNN (VanillaRateRNN or LowRankRateRNN)
        h_star: (H,) approximate fixed point (e.g. from autonomous_fixed_points)
    """
    alpha = model.alpha
    H = model.hidden_dim
    b = model.bias.detach().cpu().numpy()
    W = model.get_recurrent_weights()  # (H, H) numpy

    # Pre-activation at fixed point
    pre = W @ h_star + b
    sech2 = 1.0 - np.tanh(pre) ** 2  # element-wise

    # Jacobian
    J = (1 - alpha) * np.eye(H) + alpha * np.diag(sech2) @ W

    eigs = np.linalg.eigvals(J)
    mags = np.abs(eigs)

    # Fixed-point residual: ||f(h*) - h*||
    h_next = (1 - alpha) * h_star + alpha * np.tanh(pre)
    residual = float(np.linalg.norm(h_next - h_star))

    return EigenvalueResult(
        eigenvalues=eigs,
        magnitudes=mags,
        n_near_unit=int(((mags > 0.95) & (mags < 1.05)).sum()),
        max_magnitude=float(mags.max()),
        fp_residual=residual,
    )


# ── 4.5  Singular value analysis (low-rank only) ────────────────────────


def singular_value_analysis(model: torch.nn.Module) -> np.ndarray:
    """SVD of the recurrent weight matrix.  Returns singular values."""
    W = model.get_recurrent_weights()
    return np.linalg.svd(W, compute_uv=False)


# ── Full evaluation ─────────────────────────────────────────────────────


def full_evaluation(
    model: torch.nn.Module,
    data_path: str | Path,
    device: torch.device,
    val_idx: np.ndarray | None = None,
    observed_idx: np.ndarray | None = None,
) -> dict:
    """
    Run all evaluations and print a summary.

    Designed to work with the output of train():
        result = train(data_path, config)
        report = full_evaluation(
            result['model'], data_path, result['device'],
            result['val_idx'], result['observed_idx'],
        )

    Returns a dict with all metric objects + a milestone_1_pass bool.
    """
    data_path = Path(data_path)
    data = np.load(data_path)
    trajectories = data["trajectories"]
    neuron_angles = data["neuron_angles"]
    norm_mean = data["mean"]
    norm_std = data["std"]

    n_trials, T, N = trajectories.shape

    # Default: evaluate on all trials
    if val_idx is None:
        val_idx = np.arange(n_trials)
    val_traj = trajectories[val_idx]

    print("=" * 60)
    print("  Ring Attractor RNN Evaluation")
    print("=" * 60)

    # ── 4.1  Prediction metrics ──────────────────────────────────
    print("\n4.1  Predictive metrics")
    pred = evaluate_predictions(
        model, val_traj, neuron_angles, norm_mean, norm_std, device,
    )
    print(f"  MSE (autonomous):     {pred.mse:.6f}")
    print(f"  Angle error:          {pred.angle_error_deg:.2f} deg")

    # ── 4.2  Autonomous fixed-point analysis ─────────────────────
    print("\n4.2  Autonomous fixed-point analysis  (M=300, T=500)")
    ring = autonomous_fixed_points(
        model, neuron_angles, norm_mean, norm_std, device,
    )
    print(f"  Spread:               {ring.spread:.4f}   (>> 0 = non-trivial)")
    print(f"  Uniformity:           {ring.uniformity:.4f}   (> 0.8 = pass)")
    print(f"  Circularity:          {ring.circularity:.4f}   (> 0.7 = pass)")

    # ── 4.3  Generalisation test ─────────────────────────────────
    print("\n4.3  Generalisation test  (36 intermediate angles)")
    gen = generalization_test(
        model, neuron_angles, norm_mean, norm_std, device,
    )
    print(f"  Mean |drift|:         {gen.mean_abs_drift_deg:.2f} deg  (< 5 = pass)")
    print(f"  Max  |drift|:         {np.abs(gen.drift_deg).max():.2f} deg")

    # ── 4.4  Eigenvalue analysis ─────────────────────────────────
    print("\n4.4  Eigenvalue analysis")
    # Use a few representative converged hidden states as fixed points
    n_fp = min(5, len(ring.h_final))
    fp_indices = np.linspace(0, len(ring.h_final) - 1, n_fp, dtype=int)
    eig_results = []
    for idx in fp_indices:
        er = eigenvalue_analysis(model, ring.h_final[idx])
        eig_results.append(er)
    # Summarise across fixed points
    avg_max_mag = np.mean([e.max_magnitude for e in eig_results])
    avg_near_unit = np.mean([e.n_near_unit for e in eig_results])
    avg_residual = np.mean([e.fp_residual for e in eig_results])
    print(f"  Max |lambda| (avg):   {avg_max_mag:.4f}   (> 1 = unstable)")
    print(f"  # near unit (avg):    {avg_near_unit:.1f}     (expect ~2 for ring)")
    print(f"  FP residual (avg):    {avg_residual:.6f}  (~ 0 = good FP)")

    # ── 4.5  Singular value analysis ─────────────────────────────
    svs = singular_value_analysis(model)
    print(f"\n4.5  Singular values of W_hh  (top 5)")
    for j, sv in enumerate(svs[:5]):
        marker = " <-- ring modes" if j < 2 else ""
        print(f"  sigma_{j+1} = {sv:.4f}{marker}")
    if len(svs) > 5:
        print(f"  sigma_6..{len(svs)} <= {svs[5]:.6f}")

    # ── Milestone 1 summary ──────────────────────────────────────
    m1_pass = (
        ring.uniformity > 0.8
        and ring.circularity > 0.7
        and gen.mean_abs_drift_deg < 5.0
    )
    print("\n" + "=" * 60)
    print(f"  MILESTONE 1:  {'PASS' if m1_pass else 'FAIL'}")
    print(f"    uniformity  {ring.uniformity:.3f}  {'ok' if ring.uniformity > 0.8 else 'FAIL'}  (need > 0.8)")
    print(f"    circularity {ring.circularity:.3f}  {'ok' if ring.circularity > 0.7 else 'FAIL'}  (need > 0.7)")
    print(f"    drift       {gen.mean_abs_drift_deg:.2f} deg  "
          f"{'ok' if gen.mean_abs_drift_deg < 5 else 'FAIL'}  (need < 5 deg)")
    print("=" * 60)

    return {
        "predictions": pred,
        "ring_score": ring,
        "generalization": gen,
        "eigenvalues": eig_results,
        "singular_values": svs,
        "milestone_1_pass": m1_pass,
    }
