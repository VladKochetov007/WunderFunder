"""Fixed R4 random projections over canonical raw and signed-history bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from idea_screens_r1.feature_views import build_view

BASE_SEEDS = {"raw112": 19_031, "raw_signed224": 19_032}
RANDOM_WIDTH = 64


@dataclass(frozen=True)
class ProjectionDraw:
    """One fixed, seed-bound Gaussian projection and bias draw."""

    weights: np.ndarray
    bias: np.ndarray
    seed: int


@dataclass(frozen=True)
class BaseNormalizer:
    """Training-only standardization used before the fixed projection."""

    mean: np.ndarray
    scale: np.ndarray


def build_bases(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Return raw112 and raw-plus-signed-EMA224 full-sequence feature bases."""
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 112 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("R4 bases require finite nonempty full-sequence (T,112) input")
    ema, _ = build_view(x, "raw_net_absinnovation_ema")
    return {"raw112": x, "raw_signed224": np.concatenate((x, ema[:, 112:224]), axis=1)}


def draw_projection(dimension: int, seed: int) -> ProjectionDraw:
    """Draw the contract-fixed 64-column projection and bias from one RNG."""
    if dimension < 1:
        raise ValueError("projection dimension must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / np.sqrt(dimension), (dimension, RANDOM_WIDTH))
    bias = rng.uniform(-1.0, 1.0, RANDOM_WIDTH)
    return ProjectionDraw(weights=weights, bias=bias, seed=seed)


def fit_base_normalizer(training_rows: np.ndarray) -> BaseNormalizer:
    """Fit mean/std on training rows, replacing near-constant scales by one."""
    values = np.asarray(training_rows, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("base normalizer requires finite nonempty matrix rows")
    mean = values.mean(axis=0)
    scale = np.sqrt(values.var(axis=0))
    scale[scale < 1e-12] = 1.0
    return BaseNormalizer(mean=mean, scale=scale)


def augment_pair(
    rows: np.ndarray, normalizer: BaseNormalizer, draw: ProjectionDraw
) -> tuple[np.ndarray, np.ndarray]:
    """Append paired identity and tanh features using the same W/b and norm."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(normalizer.mean):
        raise ValueError("R4 projection input and normalizer dimensions differ")
    if draw.weights.shape[0] != values.shape[1] or draw.weights.shape[1] != RANDOM_WIDTH:
        raise ValueError("R4 projection draw has an invalid shape")
    if not np.isfinite(values).all():
        raise ValueError("R4 projection input must be finite")
    hidden = ((values - normalizer.mean) / normalizer.scale) @ draw.weights + draw.bias
    identity = np.concatenate((values, hidden), axis=1)
    nonlinear = np.concatenate((values, np.tanh(hidden)), axis=1)
    return identity, nonlinear


def projection_diagnostics(
    training_rows: np.ndarray, normalizer: BaseNormalizer, draw: ProjectionDraw
) -> dict[str, float | int]:
    """Summarize the fixed projection on training rows without selecting features."""
    values = np.asarray(training_rows, dtype=np.float64)
    preactivation = ((values - normalizer.mean) / normalizer.scale) @ draw.weights + draw.bias
    return {
        "input_dimension": int(values.shape[1]),
        "width": RANDOM_WIDTH,
        "seed": draw.seed,
        "preactivation_mean": float(preactivation.mean()),
        "preactivation_std": float(preactivation.std()),
        "preactivation_max_abs": float(np.abs(preactivation).max()),
        "fraction_abs_preactivation_gt3": float(np.mean(np.abs(preactivation) > 3.0)),
        "weight_frobenius_norm": float(np.linalg.norm(draw.weights)),
        "bias_l2_norm": float(np.linalg.norm(draw.bias)),
    }
