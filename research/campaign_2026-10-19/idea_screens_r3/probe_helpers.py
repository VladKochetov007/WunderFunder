"""Small deterministic folds and canonical centered-moment utilities for R3."""

from __future__ import annotations

import random
from collections.abc import Mapping

from wnn.gmetric import GlobalAccumulator, _merge_moments

ALPHAS = (1, 100, 10_000)
FIT_ROW_STRIDE = 40
FOLD_SEED = 19_025


def component_folds(components: list[int]) -> list[set[int]]:
    """Return three deterministic folds for the fixed 32/64 component budgets."""
    ordered = sorted(set(int(value) for value in components))
    if len(ordered) not in (32, 64) or len(ordered) != len(components):
        raise ValueError("R3 folds require exactly 32 or 64 fitting components")
    random.Random(FOLD_SEED).shuffle(ordered)
    folds = [set(ordered[index::3]) for index in range(3)]
    if set.union(*folds) != set(components) or any(
        folds[left] & folds[right] for left in range(3) for right in range(left + 1, 3)
    ):
        raise RuntimeError("component folds are not a partition")
    return folds


def safe_result(accumulator: GlobalAccumulator) -> dict | None:
    """Return the canonical score, or null for a preserved zero-mask group."""
    return accumulator.result() if accumulator.selected_rows else None


def select_alpha(pooled: Mapping[int, GlobalAccumulator]) -> int:
    """Select by one official pooled score, resolving exact ties toward stronger shrinkage."""
    if any(not pooled[alpha].selected_rows for alpha in ALPHAS):
        raise ValueError("every fixed alpha needs pooled held-fit rows")
    return max(ALPHAS, key=lambda alpha: (pooled[alpha].result()["wp"], alpha))


def raw_sums_by_group(accumulator: GlobalAccumulator) -> dict[str, list[list[float]]]:
    """Expose canonical additive sums in W,WY,WP,WYY,WPP,WYP order."""
    return {str(key): value.tolist() for key, value in sorted(accumulator.groups.items())}


def metric_without(component_accumulators: Mapping[int, GlobalAccumulator], omitted: int) -> dict:
    """Pool all non-omitted groups using the canonical centered-moment merge."""
    combined = GlobalAccumulator()
    for component in sorted(component_accumulators):
        if component == omitted:
            continue
        part = component_accumulators[component]
        for target_index in range(2):
            combined._moments[target_index] = _merge_moments(
                combined._moments[target_index], part._moments[target_index]
            )
        combined.selected_rows += part.selected_rows
    return combined.result()
