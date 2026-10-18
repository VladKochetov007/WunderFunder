"""Bounded deterministic ridge screens over fixed causal feature views."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wnn.gmetric import METRIC_ID, GlobalAccumulator

if __package__:
    from .feature_views import VIEW_NAMES, build_view
else:
    from feature_views import VIEW_NAMES, build_view

RIDGE_ALPHA = 1.0
FIT_ROW_STRIDE = 40
_FIT_BLOCK_ROWS = 4096
_PREDICT_BLOCK_ROWS = 1024


@dataclass(frozen=True)
class ProbeSequence:
    """Validated single-sequence inputs and source-qualified grouping keys."""

    x: np.ndarray
    y: np.ndarray
    scored: np.ndarray
    role: str
    source: str
    group: str
    component: str


@dataclass(frozen=True)
class RidgeModel:
    """Train-normalized two-target ridge coefficients and intercepts."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    coefficient_norms: tuple[float, float]


def _sequence(item: dict[str, object]) -> ProbeSequence:
    """Validate one caller-supplied sequence without interpreting its source pin."""
    x = np.asarray(item["x"], dtype=np.float32)
    y = np.asarray(item["y"], dtype=np.float32)
    scored = np.asarray(item["scored"], dtype=bool)
    role = str(item["role"])
    if x.ndim != 2 or x.shape[1] != 112 or y.shape != (len(x), 2):
        raise ValueError("each sequence needs x=(T,112) and y=(T,2)")
    if scored.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("sequence mask shape is invalid or features are nonfinite")
    if role not in {"fit", "eval"}:
        raise ValueError("sequence role must be 'fit' or 'eval'")
    if not all(item.get(key) is not None for key in ("source", "group", "component")):
        raise ValueError("source, group and component pins are required")
    return ProbeSequence(
        x, y, scored, role, str(item["source"]), str(item["group"]), str(item["component"])
    )


def _weighted_ridge(
    features: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    """Solve an intercept-unpenalized weighted ridge model from normal equations."""
    n, width = features.shape
    if target.shape != (n,) or weights.shape != (n,) or alpha < 0:
        raise ValueError("ridge target/weight shapes or alpha are invalid")
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return np.zeros(width, dtype=np.float64), 0.0
    x_mean = (weights @ features) / total_weight
    y_mean = float(weights @ target) / total_weight
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros(width, dtype=np.float64)
    for start in range(0, n, _FIT_BLOCK_ROWS):
        end = min(start + _FIT_BLOCK_ROWS, n)
        centered = features[start:end] - x_mean
        row_weights = weights[start:end]
        centered_target = target[start:end] - y_mean
        gram += centered.T @ (centered * row_weights[:, None])
        rhs += centered.T @ (row_weights * centered_target)
    gram.flat[:: width + 1] += alpha
    coefficients = np.linalg.solve(gram, rhs)
    intercept = y_mean - float(x_mean @ coefficients)
    return coefficients, intercept


def _fit_model(features: np.ndarray, targets: np.ndarray) -> RidgeModel:
    """Fit two clipped-target weighted ridge heads with training-only scaling."""
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_scale = np.sqrt(features.var(axis=0, dtype=np.float64))
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = features.astype(np.float64)
    standardized -= feature_mean
    standardized /= feature_scale
    clipped = np.clip(targets.astype(np.float64), -2.0, 2.0)
    coefficients = np.zeros((features.shape[1], 2), dtype=np.float64)
    intercepts = np.zeros(2, dtype=np.float64)
    norms: list[float] = []
    for target_index in range(2):
        weights = np.abs(clipped[:, target_index])
        coefficients[:, target_index], intercepts[target_index] = _weighted_ridge(
            standardized, clipped[:, target_index], weights, RIDGE_ALPHA
        )
        norms.append(float(np.linalg.norm(coefficients[:, target_index])))
    return RidgeModel(feature_mean, feature_scale, coefficients, intercepts, (norms[0], norms[1]))


def _predict(features: np.ndarray, model: RidgeModel) -> np.ndarray:
    """Predict every row in bounded matrix blocks using fit-only normalization."""
    predictions = np.empty((len(features), 2), dtype=np.float32)
    for start in range(0, len(features), _PREDICT_BLOCK_ROWS):
        end = min(start + _PREDICT_BLOCK_ROWS, len(features))
        block = features[start:end].astype(np.float64)
        block -= model.feature_mean
        block /= model.feature_scale
        predictions[start:end] = (block @ model.coefficients + model.intercepts).astype(np.float32)
    return predictions


def _fit_rows(
    sequences: list[ProbeSequence], view: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Materialize only deterministic stride rows after each complete causal transform."""
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    metadata: dict[str, int] = {}
    for sequence in sequences:
        if sequence.role != "fit":
            continue
        transformed, details = build_view(sequence.x, view)
        for key, value in details.items():
            metadata[key] = metadata.get(key, 0) + value
        row_indices = np.arange(0, len(sequence.x), FIT_ROW_STRIDE)
        row_indices = row_indices[sequence.scored[row_indices]]
        if row_indices.size == 0:
            continue
        target = sequence.y[row_indices]
        if not np.isfinite(target).all():
            raise ValueError("selected fit targets must be finite")
        feature_parts.append(transformed[row_indices])
        target_parts.append(target)
    if not feature_parts:
        raise ValueError("fit partition has no selected deterministic-stride rows")
    return np.concatenate(feature_parts), np.concatenate(target_parts), metadata


def _score_view(
    sequences: list[ProbeSequence], view: str, model: RidgeModel
) -> tuple[dict[str, object] | None, list[dict[str, object]], int]:
    """Score all marked rows and pool official moments globally and by component."""
    total = GlobalAccumulator()
    components: dict[str, GlobalAccumulator] = {}
    selected_rows = 0
    for sequence in sequences:
        if sequence.role != "eval":
            continue
        transformed, _ = build_view(sequence.x, view)
        prediction = _predict(transformed, model)
        mask = sequence.scored[None, :]
        selected_rows += int(sequence.scored.sum())
        key = sequence.component
        total.add(sequence.y[None, :, :], prediction[None, :, :], mask, group=key)
        if not sequence.scored.any():
            continue
        accumulator = components.setdefault(key, GlobalAccumulator())
        accumulator.add(sequence.y[None, :, :], prediction[None, :, :], mask)
    if selected_rows == 0:
        return None, [], 0
    rows: list[dict[str, object]] = []
    for key in sorted(components):
        accumulator = components[key]
        metric = accumulator.result()
        rows.append(
            {
                "component": key,
                "selected_rows": int(metric["selected_rows"]),
                "official_metric": metric,
                "pooled_raw_moments": total.groups[key].tolist(),
            }
        )
    return total.result(), rows, selected_rows


def _variant_result(sequences: list[ProbeSequence], view: str) -> dict[str, object]:
    """Fit and report one fixed view without consulting evaluation labels in fitting."""
    fit_features, fit_targets, details = _fit_rows(sequences, view)
    if view == "raw_nonnegative_volume_proportions_log_total":
        sampled_valid = {
            "i0": int(np.count_nonzero(fit_features[:, 138] > 0.0)),
            "i1": int(np.count_nonzero(fit_features[:, 165] > 0.0)),
        }
        volume_stats = {
            "sampled_fit_valid_rows": sampled_valid,
            "sampled_fit_valid_fraction": {
                "i0": sampled_valid["i0"] / len(fit_features),
                "i1": sampled_valid["i1"] / len(fit_features),
            },
            "row_handling": (
                "invalid rows are zero-filled in added features; raw channels remain present"
            ),
        }
    else:
        volume_stats = None
    if view == "raw_nonnegative_volume_proportions_log_total" and not any(sampled_valid.values()):
        return {
            "variant": view,
            "status": "inapplicable",
            "reason": "no positive valid volume rows in sampled fit panel",
            "fit_rows": int(len(fit_features)),
            "feature_dim": int(fit_features.shape[1]),
            "volume_applicability": volume_stats,
        }
    model = _fit_model(fit_features, fit_targets)
    metric, components, eval_rows = _score_view(sequences, view, model)
    return {
        "variant": view,
        "status": "scored" if metric is not None else "no_evaluation_scored_rows",
        "feature_dim": int(fit_features.shape[1]),
        "fit_rows": int(len(fit_features)),
        "eval_scored_rows": eval_rows,
        "ridge_alpha": RIDGE_ALPHA,
        "fit_weight_formula": "abs(clip(y, -2, 2))",
        "coefficient_l2_norm_by_target": list(model.coefficient_norms),
        "official_metric": metric,
        "component_pooled_moments": components,
        "feature_view_metadata": details,
        "volume_applicability": volume_stats,
    }


def run(sequences: list[dict[str, object]], output: Path) -> dict[str, object]:
    """Fit fixed screens, score full eval panels and write compact JSON.

    Args:
        sequences: Whole fit/eval sequences with source, group and component pins.
        output: New destination JSON path in an existing output directory.

    Returns:
        The dictionary written to ``output``.

    Raises:
        ValueError: Inputs are invalid or a partition is absent.
    """
    checked = [_sequence(item) for item in sequences]
    if not any(item.role == "fit" for item in checked) or not any(
        item.role == "eval" for item in checked
    ):
        raise ValueError("both fit and eval sequences are required")
    manifest = sorted({(item.role, item.source, item.group, item.component) for item in checked})
    progress_path = output.parent / "incremental_progress.jsonl"
    if output.exists() or progress_path.exists():
        raise FileExistsError("ridge probe output or progress file already exists")
    variants: list[dict[str, object]] = []
    with progress_path.open("x", encoding="utf-8") as progress:
        for view in VIEW_NAMES:
            variant = _variant_result(checked, view)
            variants.append(variant)
            summary = {
                "variant": view,
                "status": variant["status"],
                "metric": variant.get("official_metric"),
            }
            progress.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
    result: dict[str, object] = {
        "schema": "bounded_simple_ridge_probe_v1",
        "metric_id": METRIC_ID,
        "metric_source": inspect.getsourcefile(GlobalAccumulator),
        "numeric_note": "float64 normal equations; not a strict-FP32 campaign training claim",
        "fit_sampling": {
            "row_stride_after_full_causal_features": FIT_ROW_STRIDE,
            "fit_only_partition": True,
            "normalization": "unweighted feature mean/std from sampled fit rows only",
        },
        "source_pins": [
            {"role": role, "source": source, "group": group, "component": component}
            for role, source, group, component in manifest
        ],
        "variants": variants,
        "progress_file": progress_path.name,
    }
    with output.open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True, allow_nan=False)
        result_file.write("\n")
    return result
