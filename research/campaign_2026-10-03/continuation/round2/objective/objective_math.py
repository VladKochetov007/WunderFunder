from __future__ import annotations

import numpy as np


def weighted_wp(y: np.ndarray, prediction: np.ndarray) -> float:
    """Compute official single-target clipped, label-magnitude-weighted WP."""
    y = np.clip(y.astype(float), -2, 2)
    prediction = np.clip(prediction.astype(float), -2, 2)
    weight = np.abs(y)
    if weight.sum() <= 1e-8:
        return 0.0
    y_centered = y - np.average(y, weights=weight)
    p_centered = prediction - np.average(prediction, weights=weight)
    denominator = np.sqrt(np.sum(weight * y_centered**2) * np.sum(weight * p_centered**2))
    return (
        float(np.sum(weight * y_centered * p_centered) / denominator) if denominator > 1e-8 else 0.0
    )


def cell_fit(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """Fit saturated weighted least squares with per-cluster denominators."""
    effective = weight / denominator
    return np.bincount(x, weights=effective * y, minlength=3) / np.bincount(
        x, weights=effective, minlength=3
    )


def next_pair_change(y: np.ndarray) -> np.ndarray:
    """First strictly future raw-label-pair change, equivalent to trainer semantics."""
    n = len(y)
    changed = np.r_[False, np.any(y[1:] != y[:-1], axis=1)]
    marker = np.where(changed, np.arange(n), n)
    at_or_after = np.minimum.accumulate(marker[::-1])[::-1]
    return np.r_[at_or_after[1:], n]


def bootstrap_target(y: np.ndarray, teacher: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Reconstruct exact stopped-bootstrap blend after teacher averaging and clipping."""
    clipped = np.clip(y, -2, 2)
    teacher = np.clip(teacher, -2, 2)
    ahead = np.arange(len(y)) + horizon
    alive = (next_pair_change(y) > ahead) & (ahead < len(y)) if horizon else np.zeros(len(y), bool)
    label = np.where(alive[:, None], teacher[np.minimum(ahead, len(y) - 1)], clipped)
    return (1 - alpha) * label + alpha * teacher


def summaries(values: np.ndarray) -> dict[str, float | list[float]]:
    """Describe finite scalar diagnostics without treating rows as independent."""
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "cv": float(values.std() / values.mean()),
        "q01_q10_q50_q90_q99": np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
        "min": float(values.min()),
        "max": float(values.max()),
    }
