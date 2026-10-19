"""Fixed-denominator objective and initial-gradient sampling mathematics."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from idea_screens_r7.model_math import MeanVarianceMLP

UNIFORM_MIXTURE = 0.5
UNIFORM_SEED = 19042
DRAW_ROUNDS = 12
BATCH_SIZE = 512


def clipped_weight_table(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Clip fit targets, form canonical magnitude weights, and compute fixed ``mu_w``."""
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("fit targets must have shape (N,2) with N positive")
    if not np.isfinite(values).all():
        raise ValueError("fit targets must be finite before clipping")
    clipped = np.clip(values, -2.0, 2.0).astype(np.float32, copy=False)
    weights = np.abs(clipped).astype(np.float32, copy=False)
    mu_w = float(weights.sum(dtype=np.float64) / len(weights))
    if not math.isfinite(mu_w) or mu_w <= 0.0:
        raise ValueError("fixed mean row-sum weight mu_w must be positive and finite")
    return np.ascontiguousarray(clipped), np.ascontiguousarray(weights), mu_w


def analytic_joint_gradient_norms(
    features: np.ndarray,
    clipped_targets: np.ndarray,
    weights: np.ndarray,
    mu_w: float,
    model: MeanVarianceMLP,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Compute exact-GELU per-row joint trunk/mean-head gradient norms in FP64."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(clipped_targets, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x), 2) or w.shape != y.shape or len(x) == 0:
        raise ValueError("features and two-target tables must have matching nonempty rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("gradient-norm inputs must be finite")
    if not math.isfinite(mu_w) or mu_w <= 0.0 or chunk_size < 1:
        raise ValueError("mu_w and chunk_size must be positive and finite")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("initial-gradient prepass is CPU-only")

    trunk_weight = model.trunk.weight.detach().to(torch.float64)
    trunk_bias = model.trunk.bias.detach().to(torch.float64)
    head_weight = model.mean_head.weight.detach().to(torch.float64)
    head_bias = model.mean_head.bias.detach().to(torch.float64)
    norms = np.empty(len(x), dtype=np.float64)
    root_two = math.sqrt(2.0)
    inverse_root_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            stop = min(start + chunk_size, len(x))
            xb = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(torch.float64)
            yb = torch.from_numpy(np.ascontiguousarray(y[start:stop])).to(torch.float64)
            wb = torch.from_numpy(np.ascontiguousarray(w[start:stop])).to(torch.float64)
            preactivation = F.linear(xb, trunk_weight, trunk_bias)
            hidden = F.gelu(preactivation, approximate="none")
            mean = F.linear(hidden, head_weight, head_bias)
            output_gradient = 2.0 * wb * (mean - yb) / mu_w
            cdf = 0.5 * (1.0 + torch.erf(preactivation / root_two))
            density = torch.exp(-0.5 * preactivation.square()) * inverse_root_two_pi
            gelu_derivative = cdf + preactivation * density
            hidden_gradient = (output_gradient @ head_weight) * gelu_derivative
            head_norm_squared = output_gradient.square().sum(dim=1) * (
                hidden.square().sum(dim=1) + 1.0
            )
            trunk_norm_squared = hidden_gradient.square().sum(dim=1) * (
                xb.square().sum(dim=1) + 1.0
            )
            row_norms = torch.sqrt(head_norm_squared + trunk_norm_squared)
            norms[start:stop] = row_norms.cpu().numpy()
    if not np.isfinite(norms).all() or (norms < 0.0).any():
        raise FloatingPointError("initial joint gradient norms are nonfinite or negative")
    if float(norms.sum(dtype=np.float64)) <= 0.0:
        raise ValueError("initial joint gradient norms have zero total mass")
    return norms


def importance_probabilities(norms: np.ndarray) -> np.ndarray:
    """Return the fixed 50%-uniform mixture distribution over fitting rows."""
    values = np.asarray(norms, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("gradient norms must be a nonempty finite vector")
    if (values < 0.0).any():
        raise ValueError("gradient norms cannot be negative")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("importance distribution requires positive total gradient norm")
    probabilities = UNIFORM_MIXTURE / len(values) + (1.0 - UNIFORM_MIXTURE) * values / total
    residual = 1.0 - float(probabilities.sum(dtype=np.float64))
    probabilities[int(np.argmax(probabilities))] += residual
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < UNIFORM_MIXTURE / len(values)).any()
    ):
        raise FloatingPointError("importance probabilities violate the uniform mixture floor")
    if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise FloatingPointError("importance probabilities do not sum to one")
    return probabilities


def sample_indices(uniforms: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Map shared uniforms through one categorical inverse CDF with replacement."""
    values = np.asarray(uniforms, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values >= 1.0).any()
    ):
        raise ValueError("uniform variates must be a finite vector in [0,1)")
    if probs.ndim != 1 or len(probs) == 0 or not np.isfinite(probs).all() or (probs <= 0.0).any():
        raise ValueError("sampling probabilities must be a positive finite vector")
    if not math.isclose(float(probs.sum()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise ValueError("sampling probabilities must sum to one")
    cdf = np.cumsum(probs, dtype=np.float64)
    cdf[-1] = 1.0
    return np.minimum(np.searchsorted(cdf, values, side="right"), len(probs) - 1).astype(
        np.int64, copy=False
    )


def correction_factors(probabilities: np.ndarray) -> np.ndarray:
    """Return the exact fixed-design importance correction ``1/(N*p_i)``."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or len(probs) == 0 or not np.isfinite(probs).all() or (probs <= 0).any():
        raise ValueError("sampling probabilities must be positive and finite")
    if not math.isclose(float(probs.sum()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise ValueError("sampling probabilities must sum to one")
    corrections = 1.0 / (len(probs) * probs)
    if not np.isfinite(corrections).all() or float(corrections.max()) > 2.0 + 1e-12:
        raise FloatingPointError("importance correction exceeds the reviewed mixture bound")
    return corrections


def uncentered_gradient_second_moment(norms: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the fixed-theta norm-only uncentered sampling second moment."""
    values = np.asarray(norms, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or probs.shape != values.shape or len(values) == 0:
        raise ValueError("norm and probability vectors must be nonempty and shape-matched")
    if not np.isfinite(values).all() or not np.isfinite(probs).all() or (probs <= 0).any():
        raise ValueError("norms/probabilities must be finite with positive probabilities")
    return float(np.sum(np.square(values) / probs, dtype=np.float64) / len(values) ** 2)
