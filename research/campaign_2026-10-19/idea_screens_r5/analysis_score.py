"""Stream fixed decoder scores and descriptive moment diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from idea_screens_r2.views import build_view_for
from idea_screens_r3.probe_helpers import (
    metric_without,
    raw_sums_by_group,
    safe_result,
)

from wnn.gmetric import GlobalAccumulator

from .analysis_diagnostics import (
    DECODERS,
    TARGETS,
    empty_diagnostic,
    finish_diagnostic,
    update_diagnostic,
)
from .analysis_fit import BASE_VIEW, MomentFit, decode_moments, predict_moments


def _score_role(
    role_sequences: list[dict], fit: MomentFit, progress: Callable[[dict], None] | None
) -> tuple[dict, dict, dict]:
    """Score one panel role and retain component moments and fit-mask diagnostics."""
    totals = {decoder: GlobalAccumulator() for decoder in DECODERS}
    by_component: dict[int, dict[str, GlobalAccumulator]] = {}
    pooled_diagnostics = {target: empty_diagnostic() for target in TARGETS}
    component_diagnostics: dict[str, dict[str, dict]] = {}
    seen: set[int] = set()
    scored_rows = 0
    for index, sequence in enumerate(role_sequences):
        component = int(sequence["component"])
        if component in seen:
            raise ValueError("multiple R5 sequences in one component/role")
        seen.add(component)
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.ndim != 2 or x.shape[1] != 112 or y.shape != (len(x), 2) or mask.shape != (len(x),):
            raise ValueError("R5 scoring needs complete aligned x/y/scored sequences")
        if mask[:99].any() or not np.isfinite(y[mask]).all():
            raise ValueError("R5 masks or scored labels differ from the warmup/finite contract")
        base = build_view_for(x, BASE_VIEW)
        moment_prediction = predict_moments(base, fit)
        decoded = decode_moments(moment_prediction, fit.mu_b)
        if any(not np.isfinite(decoded[name]).all() for name in DECODERS):
            raise ValueError("R5 raw decoder predictions must be finite on all sequence rows")
        y_batch, mask_batch = y[None, :, :], mask[None, :]
        scored_rows += int(mask.sum())
        by_component[component] = {decoder: GlobalAccumulator() for decoder in DECODERS}
        component_diagnostics[str(component)] = {target: empty_diagnostic() for target in TARGETS}
        for target_index, target in enumerate(TARGETS):
            update_diagnostic(pooled_diagnostics[target], decoded, mask, fit, target_index)
            update_diagnostic(
                component_diagnostics[str(component)][target], decoded, mask, fit, target_index
            )
        for decoder in DECODERS:
            prediction = decoded[decoder]
            totals[decoder].add(y_batch, prediction[None, :, :], mask_batch, group=component)
            by_component[component][decoder].add(y_batch, prediction[None, :, :], mask_batch)
        if progress is not None:
            progress(
                {
                    "stage": "sequence_scored",
                    "role": sequence["role"],
                    "index": index + 1,
                    "component": component,
                }
            )
    components = sorted(seen)
    result = {
        "sequence_count": len(role_sequences),
        "component_count": len(components),
        "scored_rows": scored_rows,
        "by_decoder": {},
    }
    for decoder in DECODERS:
        total = totals[decoder]
        full = safe_result(total)
        component_rows, drop_rows = [], []
        for component in components:
            part = by_component[component][decoder]
            component_rows.append(
                {
                    "component": component,
                    "scored_rows": part.selected_rows,
                    "official_metric": safe_result(part),
                    "additive_raw_sums_W_WY_WP_WYY_WPP_WYP": total.groups[component].tolist(),
                }
            )
            remaining = total.selected_rows - part.selected_rows
            without = (
                metric_without(
                    {key: value[decoder] for key, value in by_component.items()}, component
                )
                if remaining
                else None
            )
            drop_rows.append(
                {
                    "omitted_component": component,
                    "scored_rows": remaining,
                    "official_metric": without,
                    "delta_wp_from_full_fixed_predictions": None
                    if without is None or full is None
                    else without["wp"] - full["wp"],
                }
            )
        result["by_decoder"][decoder] = {
            "official_metric": full,
            "component_moments": component_rows,
            "drop_one_component": drop_rows,
            "component_additive_raw_sums_W_WY_WP_WYY_WPP_WYP": raw_sums_by_group(total),
        }
    contrasts = {}
    left = result["by_decoder"]["conditional_ratio"]
    right = result["by_decoder"]["constant_denominator"]
    full_left, full_right = left["official_metric"], right["official_metric"]
    contrasts["full_conditional_ratio_minus_constant_denominator_t0_t1_wp"] = (
        None
        if full_left is None or full_right is None
        else {key: float(full_left[key] - full_right[key]) for key in ("t0", "t1", "wp")}
    )
    right_drop = {
        row["omitted_component"]: row["official_metric"] for row in right["drop_one_component"]
    }
    contrasts["drop_one_fixed_prediction_conditional_ratio_minus_constant_denominator"] = [
        {
            "omitted_component": row["omitted_component"],
            "conditional_ratio_wp": row["official_metric"]["wp"]
            if row["official_metric"]
            else None,
            "constant_denominator_wp": right_drop[row["omitted_component"]]["wp"]
            if right_drop[row["omitted_component"]]
            else None,
            "delta_wp": (
                None
                if row["official_metric"] is None or right_drop[row["omitted_component"]] is None
                else row["official_metric"]["wp"] - right_drop[row["omitted_component"]]["wp"]
            ),
        }
        for row in left["drop_one_component"]
    ]
    result["paired_decoder_contrasts"] = contrasts
    diagnostics = {
        "pooled": {
            target: finish_diagnostic(value) for target, value in pooled_diagnostics.items()
        },
        "by_component": {
            component: {target: finish_diagnostic(value) for target, value in target_stats.items()}
            for component, target_stats in component_diagnostics.items()
        },
        "ratio_histogram_definition": "B_hat / fit_muB, fixed bins; descriptive only",
    }
    return (
        result,
        diagnostics,
        {
            str(component): int(by_component[component][DECODERS[0]].selected_rows)
            for component in components
        },
    )


def score_panel(
    sequences: list[dict], fit: MomentFit, progress: Callable[[dict], None] | None = None
) -> tuple[dict, dict]:
    """Score both decoders on full fit/eval masks, retaining zero-row groups."""
    roles = {
        role: [item for item in sequences if item.get("role") == role] for role in ("fit", "eval")
    }
    if len(roles["fit"]) != 64 or len(roles["eval"]) != 32:
        raise ValueError("R5 scoring requires the fixed 64-fit/32-evaluation panel")
    fit_components = {int(item["component"]) for item in roles["fit"]}
    eval_components = {int(item["component"]) for item in roles["eval"]}
    if len(fit_components) != 64 or len(eval_components) != 32 or fit_components & eval_components:
        raise ValueError("R5 fit/evaluation components must be unique and disjoint")
    scored, diagnostics, mask_counts = {}, {}, {}
    for role in ("fit", "eval"):
        scored[role], diagnostics[role], mask_counts[role] = _score_role(roles[role], fit, progress)
    if scored["eval"]["scored_rows"] != 59_436:
        raise ValueError("R5 evaluation mask differs from the frozen 59,436-row panel")
    return {
        "fit": scored["fit"],
        "eval": scored["eval"],
        "mask_rows_by_component": mask_counts,
    }, diagnostics
