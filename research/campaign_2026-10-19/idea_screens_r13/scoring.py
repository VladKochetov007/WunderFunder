"""Canonical pooled scoring for the three fixed R13 mean-prediction arms."""

from __future__ import annotations

import numpy as np
import torch
from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group
from idea_screens_r7.model_math import MeanVarianceMLP
from idea_screens_r7.panel import EVAL_SCORED_ROWS

from wnn.gmetric import GlobalAccumulator

from .input_panel import ARM_NAMES, prepare_evaluation_rows
from .model_arms import R13MeanVarianceMLP


def _add_component_predictions(
    pooled: GlobalAccumulator,
    by_component: dict[int, GlobalAccumulator],
    targets: np.ndarray,
    predictions: np.ndarray,
    component: int,
) -> int:
    """Accumulate one original evaluation component with canonical clipping."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("R13 component predictions require aligned (N,2) arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise FloatingPointError("R13 component contains nonfinite selected values")
    valid = np.ones(len(y), dtype=bool)
    pooled.add(y, p, valid, group=component)
    part = GlobalAccumulator()
    part.add(y, p, valid, group=component)
    by_component[component] = part
    return int(len(y))


def _predict_rows(model: MeanVarianceMLP, features: np.ndarray) -> np.ndarray:
    """Predict raw means in bounded CPU batches without materializing a sequence."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 336:
        raise ValueError("R13 evaluation features must have shape (N,336)")
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            batch = torch.from_numpy(np.ascontiguousarray(values[start : start + 2048]))
            parts.append(model.forward_mean(batch).cpu().numpy())
    if not parts:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def score_models(
    models: dict[str, R13MeanVarianceMLP],
    evaluation: list[dict],
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Score all arms once on the original evaluation masks and component groups."""
    if set(models) != set(ARM_NAMES):
        raise ValueError("R13 scorer requires exactly the three frozen arms")
    components = [int(sequence["component"]) for sequence in evaluation]
    if len(components) != 32 or len(set(components)) != 32:
        raise ValueError("R13 evaluation must contain the original 32 distinct components")
    pooled = {arm: GlobalAccumulator() for arm in ARM_NAMES}
    by_component: dict[str, dict[int, GlobalAccumulator]] = {arm: {} for arm in ARM_NAMES}
    component_rows: dict[str, dict[str, int]] = {arm: {} for arm in ARM_NAMES}
    for sequence in evaluation:
        component, base, sides, targets, row_count = prepare_evaluation_rows(sequence, mean, scale)
        if component in by_component[ARM_NAMES[0]]:
            raise ValueError("duplicate R13 evaluation component")
        features_by_arm = {arm: np.concatenate((base, sides[arm]), axis=1) for arm in ARM_NAMES}
        for arm in ARM_NAMES:
            prediction = _predict_rows(models[arm], features_by_arm[arm])
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite R13 {arm} evaluation prediction")
            component_rows[arm][str(component)] = _add_component_predictions(
                pooled[arm], by_component[arm], targets, prediction, component
            )

    eval_scored_rows = sum(component_rows[ARM_NAMES[0]].values())
    if eval_scored_rows != EVAL_SCORED_ROWS or any(
        sum(component_rows[arm].values()) != EVAL_SCORED_ROWS for arm in ARM_NAMES
    ):
        raise ValueError("R13 scorer did not cover the fixed 59,436 evaluation rows per arm")

    arms: dict[str, object] = {}
    for arm in ARM_NAMES:
        score = pooled[arm].result()
        component_scores = {
            str(component): by_component[arm][component].result()
            if by_component[arm][component].selected_rows
            else None
            for component in components
        }
        drop_one = {}
        for omitted in components:
            remaining = sum(
                part.selected_rows
                for component, part in by_component[arm].items()
                if component != omitted
            )
            drop_one[str(omitted)] = (
                metric_without(by_component[arm], omitted) if remaining else None
            )
        arms[arm] = {
            "official_metric": score,
            "component_rows": component_rows[arm],
            "component_scores": component_scores,
            "component_drop_one_scores": drop_one,
            "official_clipped_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": raw_sums_by_group(
                pooled[arm]
            ),
        }
    wp = {arm: arms[arm]["official_metric"]["wp"] for arm in ARM_NAMES}
    deltas = {
        "warmup99_minus_current_bypass_wp": wp["warmup99"] - wp["current_bypass"],
        "warmup99_minus_zero_wp": wp["warmup99"] - wp["zero"],
        "current_bypass_minus_zero_wp": wp["current_bypass"] - wp["zero"],
    }
    component_deltas = {}
    component_drop_one_deltas = {}
    for component in components:
        key = str(component)
        component_scores = {
            arm: arms[arm]["component_scores"][key]["wp"]
            if arms[arm]["component_scores"][key] is not None
            else None
            for arm in ARM_NAMES
        }
        component_deltas[key] = {
            "warmup99_minus_current_bypass_wp": (
                component_scores["warmup99"] - component_scores["current_bypass"]
                if component_scores["warmup99"] is not None
                and component_scores["current_bypass"] is not None
                else None
            ),
            "warmup99_minus_zero_wp": (
                component_scores["warmup99"] - component_scores["zero"]
                if component_scores["warmup99"] is not None and component_scores["zero"] is not None
                else None
            ),
            "current_bypass_minus_zero_wp": (
                component_scores["current_bypass"] - component_scores["zero"]
                if component_scores["current_bypass"] is not None
                and component_scores["zero"] is not None
                else None
            ),
        }
        drop_scores = {
            arm: arms[arm]["component_drop_one_scores"][key]["wp"]
            if arms[arm]["component_drop_one_scores"][key] is not None
            else None
            for arm in ARM_NAMES
        }
        component_drop_one_deltas[key] = {
            "warmup99_minus_current_bypass_wp": (
                drop_scores["warmup99"] - drop_scores["current_bypass"]
                if drop_scores["warmup99"] is not None and drop_scores["current_bypass"] is not None
                else None
            ),
            "warmup99_minus_zero_wp": (
                drop_scores["warmup99"] - drop_scores["zero"]
                if drop_scores["warmup99"] is not None and drop_scores["zero"] is not None
                else None
            ),
            "current_bypass_minus_zero_wp": (
                drop_scores["current_bypass"] - drop_scores["zero"]
                if drop_scores["current_bypass"] is not None and drop_scores["zero"] is not None
                else None
            ),
        }
    return {
        "metric_id": "global_wp_clip2_v1",
        "eval_components": components,
        "eval_scored_rows": eval_scored_rows,
        "component_sensitivity_is_not_independent_replication": True,
        "arms": arms,
        "primary_delta_warmup99_minus_current_bypass_wp": deltas[
            "warmup99_minus_current_bypass_wp"
        ],
        "deltas_wp": deltas,
        "component_deltas_wp": component_deltas,
        "component_drop_one_deltas_wp": component_drop_one_deltas,
        "prospective_exploratory_gate_margin_wp": 0.005,
        "prospective_exploratory_gate_passed": (
            deltas["warmup99_minus_current_bypass_wp"] > 0.005
            and deltas["warmup99_minus_zero_wp"] > 0.005
        ),
    }
