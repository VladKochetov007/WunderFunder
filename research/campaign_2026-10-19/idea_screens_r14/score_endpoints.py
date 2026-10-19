"""Score the three fixed R14 linear endpoints with canonical centered moments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .history_features import build_a_features

ENDPOINTS = ("arm_a_128", "arm_b_128", "common_optimum")


def score_endpoints(
    sequences: list[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: dict[str, np.ndarray],
    memory_guard: Callable[[str], dict[str, int | str]],
) -> dict[str, object]:
    """Evaluate only original eval masks after the fixed optimization completes.

    ``memory_guard`` is the checked runtime guard; its object annotation keeps
    this pure scoring helper independent of runtime-control implementation.
    """
    from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group

    from wnn.gmetric import GlobalAccumulator

    accumulators = {name: GlobalAccumulator() for name in ENDPOINTS}
    by_component = {name: {} for name in ENDPOINTS}
    guards: list[dict[str, int | str]] = []
    selected_total = 0
    evaluation = [item for item in sequences if item.get("role") == "eval"]
    if len(evaluation) != 32:
        raise ValueError("R14 scoring requires the fixed 32 eval components")
    seen_components: set[int] = set()
    for index, sequence in enumerate(evaluation):
        component = int(sequence["component"])
        if component in seen_components:
            raise ValueError("R14 evaluation contains a duplicate component")
        seen_components.add(component)
        raw = np.asarray(sequence["x"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        if mask.shape != (len(raw),):
            raise ValueError("R14 eval mask differs from complete sequence length")
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError("R14 eval sequence has no original scored rows")
        if not np.isfinite(np.asarray(sequence["y"])[rows]).all():
            raise ValueError("R14 selected evaluation targets are nonfinite")
        guards.append(memory_guard(f"before_eval_history_{index}"))
        full_features = build_a_features(raw, mean, scale)
        selected_features = full_features[rows]
        targets = np.asarray(sequence["y"])[rows]
        for endpoint in ENDPOINTS:
            prediction64 = selected_features @ coefficients[endpoint]
            if not np.isfinite(prediction64).all():
                raise ValueError(f"R14 {endpoint} produced nonfinite eval predictions")
            prediction32 = prediction64.astype(np.float32)
            accumulators[endpoint].add(
                targets, prediction32, np.ones(len(rows), dtype=bool), group=component
            )
            accumulator = GlobalAccumulator()
            accumulator.add(targets, prediction32, np.ones(len(rows), dtype=bool))
            by_component[endpoint][component] = accumulator
        selected_total += len(rows)
        del full_features, selected_features, targets, rows, raw, mask
        guards.append(memory_guard(f"after_eval_history_{index}"))
    if selected_total != 59_436 or len(seen_components) != 32:
        raise ValueError("R14 eval scored-row/component coverage differs from its fixed panel")

    result: dict[str, object] = {
        "selected_rows": selected_total,
        "endpoints": {},
        "resource_guards": guards,
    }
    endpoint_results: dict[str, object] = {}
    for endpoint in ENDPOINTS:
        pooled = accumulators[endpoint]
        component_accumulators = by_component[endpoint]
        endpoint_results[endpoint] = {
            "pooled": pooled.result(),
            "component_scores": {
                str(component): accumulator.result()
                for component, accumulator in sorted(component_accumulators.items())
            },
            "component_official_clipped_six_sums_w_wy_wp_wyy_wpp_wyp": raw_sums_by_group(pooled),
            "drop_one_component_scores": {
                str(component): metric_without(component_accumulators, component)
                for component in sorted(component_accumulators)
            },
        }
    result["endpoints"] = endpoint_results
    return result
