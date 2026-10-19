"""Fit fixed-alpha conditional label moments on complete causal histories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel, predict
from idea_screens_r2.views import build_view_for

ALPHA = 100.0
FIT_ROW_STRIDE = 40
BASE_VIEW = "raw_signed_delta_ema_hl32"
MOMENT_NAMES = ("A_plus", "A_minus", "B_abs")


@dataclass(frozen=True)
class MomentFit:
    """Shared standardized six-column ridge moment model and its fixed floors."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    mu_b: np.ndarray
    epsilon: np.ndarray


def moment_targets(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped targets and six moments ordered by target then A+, A-, B."""
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise ValueError("moment fitting needs finite (rows,2) targets")
    clipped = np.clip(y, -2.0, 2.0)
    positive = np.where(clipped > 0.0, clipped * clipped, 0.0)
    negative = np.where(clipped < 0.0, clipped * clipped, 0.0)
    moments = np.stack((positive, negative, np.abs(clipped)), axis=2)
    return clipped, moments.reshape(len(y), 6)


def fit_moment_model(features: np.ndarray, targets: np.ndarray) -> MomentFit:
    """Fit all six unweighted moment columns with one centered ridge solve."""
    x = np.asarray(features, dtype=np.float64)
    clipped, target_columns = moment_targets(targets)
    if x.ndim != 2 or len(x) != len(clipped) or not len(x) or not np.isfinite(x).all():
        raise ValueError("moment fitting needs finite nonempty features aligned to targets")
    means = x.mean(axis=0)
    scales = np.sqrt(x.var(axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = (x - means) / scales
    feature_offset = normalized.mean(axis=0)
    target_offset = target_columns.mean(axis=0)
    centered_x = normalized - feature_offset
    centered_y = target_columns - target_offset
    gram = centered_x.T @ centered_x
    gram.flat[:: len(gram) + 1] += ALPHA
    coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)
    intercepts = target_offset - feature_offset @ coefficients
    moments = target_columns.reshape(len(clipped), 2, 3)
    mu_b = moments[:, :, 2].mean(axis=0)
    if not np.isfinite(mu_b).all() or np.any(mu_b <= 0.0):
        raise ValueError("each clipped target must have positive fit mean absolute value")
    epsilon = 0.05 * mu_b
    return MomentFit(means, scales, coefficients, intercepts, mu_b, epsilon)


def _head_model(fit: MomentFit, moment_index: int) -> RidgeModel:
    """Adapt one pair of moment columns to the unchanged R2 prediction helper."""
    columns = np.array([moment_index, 3 + moment_index])
    coefficients = fit.coefficients[:, columns]
    intercepts = fit.intercepts[columns]
    norms = tuple(float(value) for value in np.linalg.norm(coefficients, axis=0))
    return RidgeModel(fit.mean, fit.scale, coefficients, intercepts, norms)


def predict_moments(features: np.ndarray, fit: MomentFit) -> np.ndarray:
    """Predict A+, A-, and B for both targets through the canonical R2 helper."""
    if features.ndim != 2 or features.shape[1] != len(fit.mean):
        raise ValueError("prediction features do not match the fitted base view")
    heads = [predict(features, _head_model(fit, index)) for index in range(3)]
    prediction = np.stack(heads, axis=2)
    if not np.isfinite(prediction).all():
        raise ValueError("moment heads produced nonfinite predictions")
    return prediction


def decode_moments(moment_prediction: np.ndarray, mu_b: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the two prespecified decoders to identical fitted moment predictions."""
    moments = np.asarray(moment_prediction, dtype=np.float64)
    means = np.asarray(mu_b, dtype=np.float64)
    if moments.ndim != 3 or moments.shape[1:] != (2, 3) or means.shape != (2,):
        raise ValueError("expected moment predictions (rows,2,3) and two fit means")
    if not np.isfinite(moments).all() or not np.isfinite(means).all() or np.any(means <= 0):
        raise ValueError("decoder inputs must be finite with positive fit mean denominators")
    a_plus, a_minus, b_hat = (moments[:, :, index] for index in range(3))
    numerator = a_plus - a_minus
    denominator = np.maximum(b_hat, 0.05 * means[None, :])
    return {
        "A_plus": a_plus,
        "A_minus": a_minus,
        "B_hat": b_hat,
        "D_hat": numerator,
        "conditional_ratio": numerator / denominator,
        "constant_denominator": numerator / means[None, :],
        "ratio_denominator": denominator,
    }


def build_sequence_features(raw: np.ndarray) -> np.ndarray:
    """Build one complete reset-local R2 history view before any row stride."""
    return build_view_for(np.asarray(raw, dtype=np.float32), BASE_VIEW)


def scored_stride_indices(scored: np.ndarray) -> np.ndarray:
    """Select original scored rows on the fixed stride after history creation."""
    mask = np.asarray(scored, dtype=bool)
    if mask.ndim != 1 or not len(mask) or mask[:99].any():
        raise ValueError("stride selection needs a nonempty scored mask after the 99-row warmup")
    rows = np.arange(0, len(mask), FIT_ROW_STRIDE)
    return rows[mask[rows]]


def collect_fit_rows(
    sequences: list[dict],
    component_order: list[int],
    progress: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build full fit-sequence histories, then retain original-mask stride rows."""
    items = [item for item in sequences if item.get("role") == "fit"]
    by_component = {int(item["component"]): item for item in items}
    if (
        len(items) != 64
        or len(by_component) != 64
        or len(component_order) != 64
        or len(set(component_order)) != 64
        or set(by_component) != set(component_order)
    ):
        raise ValueError("R5 requires one fit sequence for each ordered 64 components")
    feature_parts, target_parts = [], []
    scored_counts, sampled_counts = {}, {}
    for index, component in enumerate(component_order):
        item = by_component[component]
        x = np.asarray(item["x"], dtype=np.float32)
        y = np.asarray(item["y"], dtype=np.float32)
        mask = np.asarray(item["scored"], dtype=bool)
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or mask.shape != (20_000,):
            raise ValueError("R5 requires complete source-bound 20,000-row fit sequences")
        if mask[:99].any():
            raise ValueError("fit mask must preserve the original 99-row warmup")
        if item.get("source") != "train" or int(item["component"]) != component:
            raise ValueError("R5 fit role/source binding differs")
        features = build_sequence_features(x)
        rows = scored_stride_indices(mask)
        if not len(rows) or not np.isfinite(y[rows]).all():
            raise ValueError("fit sequence has no finite scored stride rows")
        feature_parts.append(features[rows])
        target_parts.append(y[rows])
        scored_counts[str(component)] = int(mask.sum())
        sampled_counts[str(component)] = int(len(rows))
        if progress is not None:
            progress(
                {"stage": "fit_sequence_collected", "index": index + 1, "component": component}
            )
    features = np.concatenate(feature_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    fit_scored_rows = sum(scored_counts.values())
    if len(features) != 31_808 or fit_scored_rows != 1_273_664:
        raise ValueError("R5 fit mask/stride counts differ from the frozen R3 panel")
    return (
        features,
        targets,
        {
            "fit_components": 64,
            "fit_sequences": 64,
            "fit_scored_rows": fit_scored_rows,
            "fit_sampled_rows": len(features),
            "scored_rows_by_component": scored_counts,
            "sampled_rows_by_component": sampled_counts,
            "training_stride": FIT_ROW_STRIDE,
            "history_before_stride": True,
        },
    )
