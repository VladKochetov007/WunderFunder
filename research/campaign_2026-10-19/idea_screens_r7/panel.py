"""R7 complete-history feature collection and fixed-panel scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from idea_screens_r2.views import build_view_for
from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group

from wnn.gmetric import GlobalAccumulator

from .model_math import MeanVarianceMLP, apply_normalizer, predict_rows

FIT_ROW_STRIDE = 40
SEQUENCE_ROWS = 20_000
FIT_COMPONENTS = 64
EVAL_COMPONENTS = 32
FIT_ROWS = 31_808
EVAL_SCORED_ROWS = 59_436
VIEW_NAME = "raw_signed_delta_ema_hl32"


def _component_key(sequence: Mapping[str, object]) -> int:
    value = sequence.get("component")
    if not isinstance(value, (int, np.integer)):
        raise ValueError("each R3 sample record needs an integer component")
    return int(value)


def collect_fit_rows(sequences: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Build full reset-local histories, then keep only frozen fit stride rows."""
    fit = [item for item in sequences if item.get("role") == "fit"]
    evaluation = [item for item in sequences if item.get("role") == "eval"]
    if len(fit) != FIT_COMPONENTS or len(evaluation) != EVAL_COMPONENTS:
        raise ValueError("R7 requires the accepted 64-fit/32-evaluation R3 panel")
    fit_components = [_component_key(item) for item in fit]
    eval_components = [_component_key(item) for item in evaluation]
    if len(set(fit_components)) != FIT_COMPONENTS or len(set(eval_components)) != EVAL_COMPONENTS:
        raise ValueError("R3 panel must contain one sequence per distinct component")
    if set(fit_components) & set(eval_components):
        raise ValueError("fit and evaluation components overlap")
    group_keys = [(str(item.get("source")), str(item.get("group"))) for item in sequences]
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("duplicate source-qualified sample group")

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    fit_rows_by_component: dict[str, int] = {}
    fit_scored_rows = 0
    for sequence in fit:
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (SEQUENCE_ROWS, 112) or y.shape != (SEQUENCE_ROWS, 2):
            raise ValueError("R3 fit records must contain complete 20,000-row arrays")
        if mask.shape != (SEQUENCE_ROWS,) or not np.isfinite(x).all():
            raise ValueError("invalid fit mask or nonfinite fit features")
        rows = np.arange(0, SEQUENCE_ROWS, FIT_ROW_STRIDE)
        rows = rows[mask[rows]]
        if not np.isfinite(y[rows]).all():
            raise ValueError("selected fit targets must be finite")
        full_view = build_view_for(x, VIEW_NAME)
        if full_view.shape != (SEQUENCE_ROWS, 224) or not np.isfinite(full_view).all():
            raise ValueError("R2 signed-EMA view must be finite and shape (20,000,224)")
        feature_parts.append(np.array(full_view[rows], dtype=np.float32, copy=True))
        target_parts.append(np.array(y[rows], dtype=np.float32, copy=True))
        fit_rows_by_component[str(_component_key(sequence))] = int(rows.size)
        fit_scored_rows += int(mask.sum())
        del full_view

    fit_features = np.concatenate(feature_parts, axis=0)
    fit_targets = np.concatenate(target_parts, axis=0)
    eval_scored_rows = 0
    for sequence in evaluation:
        x = np.asarray(sequence["x"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (SEQUENCE_ROWS, 112) or mask.shape != (SEQUENCE_ROWS,):
            raise ValueError("R3 evaluation records must contain complete features and masks")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite evaluation features")
        eval_scored_rows += int(mask.sum())
    if fit_features.shape != (FIT_ROWS, 224) or fit_targets.shape != (FIT_ROWS, 2):
        raise ValueError("R3 fit stride selection must yield exactly 31,808 rows")
    if eval_scored_rows != EVAL_SCORED_ROWS:
        raise ValueError("R3 original evaluation masks must yield exactly 59,436 rows")
    return (
        fit_features,
        fit_targets,
        evaluation,
        {
            "fit_components": FIT_COMPONENTS,
            "eval_components": EVAL_COMPONENTS,
            "fit_stride_rows": int(len(fit_targets)),
            "fit_original_scored_rows": fit_scored_rows,
            "eval_original_scored_rows": eval_scored_rows,
            "fit_rows_by_component": fit_rows_by_component,
            "feature_width": 224,
            "feature_view": VIEW_NAME,
            "histories_built_before_row_selection": True,
        },
    )


def _raw_prediction_sums(targets: np.ndarray, predictions: np.ndarray) -> list[list[float]]:
    """Compute canonical-weighted six sums while leaving predictions unclipped."""
    y = np.clip(np.asarray(targets, dtype=np.float64), -2.0, 2.0)
    p = np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("raw sufficient statistics require matching (N,2) arrays")
    weights = np.abs(y)
    return np.stack(
        (
            weights.sum(axis=0),
            (weights * y).sum(axis=0),
            (weights * p).sum(axis=0),
            (weights * y * y).sum(axis=0),
            (weights * p * p).sum(axis=0),
            (weights * y * p).sum(axis=0),
        ),
        axis=0,
    ).T.tolist()


def _new_diagnostic() -> dict[str, np.ndarray]:
    return {
        key: np.zeros(2, dtype=np.float64)
        for key in ("count", "below", "above", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")
    }


def _add_diagnostic(
    state: dict[str, np.ndarray], raw_logvar: np.ndarray, residual: np.ndarray
) -> None:
    clamped = np.clip(raw_logvar.astype(np.float64), -4.0, 4.0)
    absolute_error = np.abs(residual.astype(np.float64))
    state["count"] += len(clamped)
    state["below"] += (raw_logvar < -4.0).sum(axis=0)
    state["above"] += (raw_logvar > 4.0).sum(axis=0)
    state["sum_x"] += clamped.sum(axis=0)
    state["sum_y"] += absolute_error.sum(axis=0)
    state["sum_xx"] += (clamped * clamped).sum(axis=0)
    state["sum_yy"] += (absolute_error * absolute_error).sum(axis=0)
    state["sum_xy"] += (clamped * absolute_error).sum(axis=0)


def _finish_diagnostic(state: dict[str, np.ndarray]) -> dict[str, object]:
    count = state["count"]
    safe = np.maximum(count, 1.0)
    cov = state["sum_xy"] / safe - state["sum_x"] * state["sum_y"] / safe**2
    var_x = state["sum_xx"] / safe - (state["sum_x"] / safe) ** 2
    var_y = state["sum_yy"] / safe - (state["sum_y"] / safe) ** 2
    denominator = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    correlations: list[float | None] = []
    for index in range(2):
        correlations.append(
            float(cov[index] / denominator[index])
            if count[index] > 1 and denominator[index] > 1e-12
            else None
        )
    return {
        "scored_rows_per_target": count.astype(int).tolist(),
        "raw_logvariance_below_minus4_rows": state["below"].astype(int).tolist(),
        "raw_logvariance_above_plus4_rows": state["above"].astype(int).tolist(),
        "fraction_clamped_low": (state["below"] / safe).tolist(),
        "fraction_clamped_high": (state["above"] / safe).tolist(),
        "correlation_clamped_logvariance_abs_residual": correlations,
        "residual_definition": "abs(clip(target,-2,2)-raw_mean)",
    }


def score_pair(
    control: MeanVarianceMLP,
    candidate: MeanVarianceMLP,
    evaluation: Sequence[dict],
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Score both final mean endpoints on original masks and retain scalar evidence."""
    models = {"control": control, "heteroskedastic": candidate}
    pooled = {name: GlobalAccumulator() for name in models}
    by_component: dict[str, dict[int, GlobalAccumulator]] = {name: {} for name in models}
    raw_sums: dict[str, dict[str, list[list[float]]]] = {name: {} for name in models}
    diagnostics = {name: _new_diagnostic() for name in models}
    component_rows: dict[str, dict[str, int]] = {name: {} for name in models}
    for sequence in evaluation:
        component = _component_key(sequence)
        key = str(component)
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (SEQUENCE_ROWS, 112) or y.shape != (SEQUENCE_ROWS, 2):
            raise ValueError("R3 evaluation records must contain complete target/input arrays")
        full_view = build_view_for(x, VIEW_NAME)
        selected_y = np.asarray(y[mask], dtype=np.float32)
        if not np.isfinite(selected_y).all():
            raise ValueError("selected evaluation targets must be finite")
        selected_x = apply_normalizer(full_view[mask], mean, scale)
        if len(selected_y):
            for name, model in models.items():
                prediction, raw_logvar = predict_rows(model, selected_x)
                if not np.isfinite(prediction).all() or not np.isfinite(raw_logvar).all():
                    raise FloatingPointError(f"nonfinite {name} evaluation output")
                clipped_y = np.clip(selected_y, -2.0, 2.0)
                _add_diagnostic(diagnostics[name], raw_logvar, clipped_y - prediction)
                component_accumulator = GlobalAccumulator()
                valid = np.ones(len(selected_y), dtype=bool)
                component_accumulator.add(selected_y, prediction, valid)
                pooled[name].add(selected_y, prediction, valid, group=component)
                by_component[name][component] = component_accumulator
                raw_sums[name][key] = _raw_prediction_sums(selected_y, prediction)
                component_rows[name][key] = int(len(selected_y))
        else:
            for name in models:
                by_component[name][component] = GlobalAccumulator()
                raw_sums[name][key] = np.zeros((2, 6), dtype=np.float64).tolist()
                component_rows[name][key] = 0
        del full_view

    arms: dict[str, object] = {}
    for name in models:
        accumulator = pooled[name]
        result = accumulator.result()
        scores = {
            str(component): (part.result() if part.selected_rows else None)
            for component, part in sorted(by_component[name].items())
        }
        drop_one = {
            str(component): metric_without(by_component[name], component)
            for component in sorted(by_component[name])
        }
        official_sums = raw_sums_by_group(accumulator)
        official_sums = {
            key: official_sums.get(key, np.zeros((2, 6), dtype=np.float64).tolist())
            for key in sorted(raw_sums[name])
        }
        arms[name] = {
            "official_metric": result,
            "component_rows": component_rows[name],
            "component_scores": scores,
            "component_drop_one_scores": drop_one,
            "raw_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": raw_sums[name],
            "official_clipped_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": official_sums,
            "diagnostics": _finish_diagnostic(diagnostics[name]),
        }
    return {
        "metric_id": "global_wp_clip2_v1",
        "components": EVAL_COMPONENTS,
        "rows": EVAL_SCORED_ROWS,
        "component_sensitivity_is_not_independent_replication": True,
        "arms": arms,
        "primary_delta_heteroskedastic_minus_control_wp": (
            arms["heteroskedastic"]["official_metric"]["wp"]
            - arms["control"]["official_metric"]["wp"]
        ),
    }
