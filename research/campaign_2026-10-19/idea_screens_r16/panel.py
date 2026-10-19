"""Stream fit-only R3 histories into the fixed R16 component halves."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from idea_screens_r10.panel_io import (
    ValidatedPanel,
    input_file_receipts,
    load_and_validate_metadata,
    map_fit_arrays,
    verify_fit_inputs,
)

from .estimator import HalfRows

FIT_COMPONENTS = 64
ROWS_PER_COMPONENT = 497
FIT_ROWS = FIT_COMPONENTS * ROWS_PER_COMPONENT
HALF_ROWS = FIT_ROWS // 2
SEQUENCE_LENGTH = 20_000
STRIDE = 40
FEATURE_WIDTH = 112


class FitPanel(NamedTuple):
    """Assembled input-only halves and exact hashed file receipts."""

    left: HalfRows
    right: HalfRows
    input_receipts: list[dict[str, object]]
    sample: ValidatedPanel


def selected_innovations(
    x: np.ndarray, scored: np.ndarray, stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed-stride x[t-1] predictors and one-step x[t]-x[t-1] responses."""
    if x.ndim != 2 or x.shape[1] != FEATURE_WIDTH or scored.shape != (x.shape[0],):
        raise ValueError("feature and score-mask shapes are inconsistent")
    if stride <= 0 or x.shape[0] <= 1:
        raise ValueError("sequence and stride must be positive")
    candidates = np.arange(0, x.shape[0], stride, dtype=np.int64)
    rows = candidates[np.asarray(scored, dtype=bool)[candidates]]
    if rows.size == 0 or np.any(rows < 1):
        raise ValueError("selected innovation rows must be nonempty and have a past row")
    past = np.asarray(x[rows - 1], dtype=np.float64)
    current = np.asarray(x[rows], dtype=np.float64)
    if not np.isfinite(past).all() or not np.isfinite(current).all():
        raise FloatingPointError("selected feature rows are nonfinite")
    delta = current - past
    if not np.isfinite(delta).all():
        raise FloatingPointError("selected one-step input innovations are nonfinite")
    return past, delta, rows


def load_fit_panel(
    sample_dir: Path,
    expected_sample_sha256: str,
    memory_guard: Callable[[str], dict[str, int | str]],
) -> FitPanel:
    """Validate all sample metadata, hash 128 fit inputs, and map fit x/masks only."""
    sample = load_and_validate_metadata(sample_dir, expected_sample_sha256)
    bindings = verify_fit_inputs(sample_dir, sample)
    if len(bindings) != FIT_COMPONENTS:
        raise ValueError("R16 must bind exactly one x and scored mask for each fit component")
    if (
        tuple(int(binding.record["component"]) for binding in bindings)
        != sample.fit_component_order
    ):
        raise ValueError("R16 fit inputs do not follow the fixed component order")

    memory_guard("before_fit_matrix_allocation")
    past = np.empty((FIT_ROWS, FEATURE_WIDTH), dtype=np.float64)
    delta = np.empty_like(past)
    memory_guard("after_fit_matrix_allocation")
    component_slices: dict[int, tuple[int, int]] = {}
    offset = 0
    for index, binding in enumerate(bindings):
        memory_guard(f"before_fit_input_map_{index}")
        arrays = map_fit_arrays(binding)
        past_rows, delta_rows, selected_rows = selected_innovations(arrays.x, arrays.scored)
        if len(selected_rows) != ROWS_PER_COMPONENT or int(selected_rows[0]) < 99:
            raise ValueError(
                "R16 expected 497 scored stride rows at or after warmup per fit sequence"
            )
        stop = offset + len(selected_rows)
        past[offset:stop] = past_rows
        delta[offset:stop] = delta_rows
        component = int(binding.record["component"])
        component_slices[component] = (offset, stop)
        offset = stop
        del arrays, past_rows, delta_rows, selected_rows
        memory_guard(f"after_fit_input_map_{index}")
    if offset != FIT_ROWS or not np.isfinite(past).all() or not np.isfinite(delta).all():
        raise ValueError("R16 assembled input rows are incomplete or nonfinite")

    components = sample.fit_component_order
    midpoint = HALF_ROWS
    left_components, right_components = components[:32], components[32:]
    left_slices = {key: component_slices[key] for key in left_components}
    right_slices = {
        key: (start - midpoint, stop - midpoint)
        for key in right_components
        for start, stop in (component_slices[key],)
    }
    left = HalfRows(left_components, past[:midpoint], delta[:midpoint], left_slices)
    right = HalfRows(right_components, past[midpoint:], delta[midpoint:], right_slices)
    # The half views retain the single preallocated matrices without copying them.
    return FitPanel(left, right, input_file_receipts(bindings), sample)
