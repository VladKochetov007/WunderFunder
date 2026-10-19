"""Single-endpoint canonical scoring and additive component evidence for R8."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from idea_screens_r2.views import build_view_for
from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group
from idea_screens_r7.model_math import apply_normalizer, predict_rows
from idea_screens_r7.panel import (
    EVAL_COMPONENTS,
    EVAL_SCORED_ROWS,
    SEQUENCE_ROWS,
    VIEW_NAME,
    _component_key,
    _raw_prediction_sums,
)

from wnn.gmetric import GlobalAccumulator

from .model import ConstantVarianceMLP


def score_endpoint(
    model: ConstantVarianceMLP,
    evaluation: Sequence[dict[str, object]],
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Score the R8 endpoint once on original masks and preserve six-sum audits."""
    if len(evaluation) != EVAL_COMPONENTS:
        raise ValueError("R8 requires the accepted 32-component evaluation panel")
    pooled = GlobalAccumulator()
    by_component: dict[int, GlobalAccumulator] = {}
    raw_sums: dict[str, list[list[float]]] = {}
    component_rows: dict[str, int] = {}

    for sequence in evaluation:
        component = _component_key(sequence)
        key = str(component)
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (SEQUENCE_ROWS, 112) or y.shape != (SEQUENCE_ROWS, 2):
            raise ValueError("R8 evaluation records must be complete 20,000-row sequences")
        if mask.shape != (SEQUENCE_ROWS,) or not np.isfinite(x).all():
            raise ValueError("R8 evaluation input or original mask is invalid")

        full_view = build_view_for(x, VIEW_NAME)
        selected_y = np.asarray(y[mask], dtype=np.float32)
        if not np.isfinite(selected_y).all():
            raise ValueError("selected R8 evaluation targets must be finite")
        selected_x = apply_normalizer(full_view[mask], mean, scale)
        part = GlobalAccumulator()
        if len(selected_y):
            prediction, raw_logvar = predict_rows(model, selected_x)
            if not np.isfinite(prediction).all() or not np.isfinite(raw_logvar).all():
                raise FloatingPointError("R8 final endpoint produced nonfinite predictions")
            if not np.all(raw_logvar == raw_logvar[:1]):
                raise RuntimeError("R8 logvariance head is not constant across rows")
            valid = np.ones(len(selected_y), dtype=bool)
            part.add(selected_y, prediction, valid)
            pooled.add(selected_y, prediction, valid, group=component)
            raw_sums[key] = _raw_prediction_sums(selected_y, prediction)
            component_rows[key] = int(len(selected_y))
        else:
            raw_sums[key] = np.zeros((2, 6), dtype=np.float64).tolist()
            component_rows[key] = 0
        by_component[component] = part
        del full_view

    if pooled.selected_rows != EVAL_SCORED_ROWS:
        raise ValueError("R8 original evaluation masks did not yield 59,436 rows")
    ordered_keys = [str(component) for component in sorted(by_component)]
    component_scores = {
        str(component): (part.result() if part.selected_rows else None)
        for component, part in sorted(by_component.items())
    }
    component_drop_one = {
        str(component): metric_without(by_component, component)
        for component in sorted(by_component)
    }
    official_sums = raw_sums_by_group(pooled)
    official_sums = {
        key: official_sums.get(key, np.zeros((2, 6), dtype=np.float64).tolist())
        for key in ordered_keys
    }
    return {
        "metric_id": "global_wp_clip2_v1",
        "components": EVAL_COMPONENTS,
        "rows": int(pooled.selected_rows),
        "component_sensitivity_is_not_independent_replication": True,
        "official_metric": pooled.result(),
        "component_rows": component_rows,
        "component_scores": component_scores,
        "component_drop_one_scores": component_drop_one,
        "raw_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": raw_sums,
        "official_clipped_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": official_sums,
    }
