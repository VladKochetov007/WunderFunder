"""Full-mask scoring and component moments for the fixed R4 evaluation panel."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel, predict
from idea_screens_r3.probe_helpers import metric_without, safe_result

from wnn.gmetric import GlobalAccumulator

from .analysis_fit import CASE_NAMES
from .random_features import BaseNormalizer, ProjectionDraw, augment_pair, build_bases


def accumulate_component(
    total: GlobalAccumulator,
    by_component: dict[int, GlobalAccumulator],
    component: int,
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Add one sequence to the official pooled and component-level moments."""
    total.add(targets, predictions, mask, group=component)
    by_component[component].add(targets, predictions, mask)


def score_outer(
    sequences: list[dict],
    models: dict[str, dict[str, RidgeModel]],
    draws: dict[str, ProjectionDraw],
    normalizers: dict[str, BaseNormalizer],
    progress: Callable[[dict], None],
) -> tuple[dict[str, dict], dict[str, float]]:
    """Score the original complete evaluation masks and retain component moments."""
    evaluation = [item for item in sequences if item["role"] == "eval"]
    components = sorted({int(item["component"]) for item in evaluation})
    if len(evaluation) != 32 or len(components) != 32:
        raise ValueError("R4 requires one evaluation sequence from each of32 components")
    totals = {case: {label: GlobalAccumulator() for label in models[case]} for case in CASE_NAMES}
    by_component = {
        case: {
            label: {component: GlobalAccumulator() for component in components}
            for label in models[case]
        }
        for case in CASE_NAMES
    }
    score_seconds = {case: 0.0 for case in CASE_NAMES}
    shared_transform_seconds = 0.0
    total_scored_rows = 0
    for index, sequence in enumerate(evaluation):
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or mask.shape != (20_000,):
            raise ValueError("R4 evaluation requires complete 20,000-row sequences")
        if not np.isfinite(y[mask]).all():
            raise ValueError("R4 evaluation labels must be finite on the original scored mask")
        started = time.process_time()
        bases = build_bases(x)
        shared_transform_seconds += time.process_time() - started
        component = int(sequence["component"])
        y_batch, mask_batch = y[None, :, :], mask[None, :]
        total_scored_rows += int(mask.sum())
        for base_name in draws:
            started = time.process_time()
            pair = augment_pair(bases[base_name], normalizers[base_name], draws[base_name])
            shared_transform_seconds += time.process_time() - started
            for activation_index, activation in enumerate(("identity", "tanh")):
                case = f"{base_name}_{activation}64"
                for label, model in models[case].items():
                    started = time.process_time()
                    prediction = predict(pair[activation_index], model)[None, :, :]
                    accumulate_component(
                        totals[case][label],
                        by_component[case][label],
                        component,
                        y_batch,
                        prediction,
                        mask_batch,
                    )
                    score_seconds[case] += time.process_time() - started
        progress({"stage": "outer_sequence_scored", "completed": index + 1, "component": component})
    if total_scored_rows != 59_436:
        raise ValueError(
            f"R4 expected59,436 original evaluation mask rows, found{total_scored_rows}"
        )

    results = {}
    for case in CASE_NAMES:
        results[case] = {}
        for label, total in totals[case].items():
            full_metric = safe_result(total)
            if full_metric is None:
                raise ValueError(f"R4 has no scored outer rows for {case}/{label}")
            component_rows, drop_rows = [], []
            for component in components:
                part = by_component[case][label][component]
                component_rows.append(
                    {
                        "component": component,
                        "scored_rows": part.selected_rows,
                        "official_metric": safe_result(part),
                        "additive_raw_sums_W_WY_WP_WYY_WPP_WYP": total.groups[component].tolist(),
                    }
                )
                remaining = total.selected_rows - part.selected_rows
                metric = metric_without(by_component[case][label], component) if remaining else None
                drop_rows.append(
                    {
                        "omitted_component": component,
                        "scored_rows": remaining,
                        "official_metric": metric,
                        "delta_wp_from_full_fixed_predictions": (
                            None if metric is None else metric["wp"] - full_metric["wp"]
                        ),
                    }
                )
            results[case][label] = {
                "official_metric": full_metric,
                "component_moments": component_rows,
                "drop_one_component": drop_rows,
            }
    return results, {
        "outer_score_cpu_s_by_case": score_seconds,
        "outer_transform_cpu_s": shared_transform_seconds,
        "outer_scored_rows": total_scored_rows,
        "outer_sequences": len(evaluation),
        "outer_components": len(components),
    }
