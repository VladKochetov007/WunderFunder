from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from wnn.data import N_FEATURES
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES

I1_HISTORY_MODE = "i1_history"
I1_RAW_START = 52
I1_RAW_STOP = 104
I1_CHANNELS = I1_RAW_STOP - I1_RAW_START
I1_HISTORY_WIDTH = 3 * I1_CHANNELS
I1_HISTORY_BLOCK_ROWS = 256
I1_HISTORY_SCALE_FLOOR = np.float32(1e-4)
I1_HISTORY_FAST_HALF_LIFE = 32.0
I1_HISTORY_SLOW_HALF_LIFE = 512.0
I1_HISTORY_SCHEMA = "i1_causal_history_v1"
I1_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES + I1_HISTORY_WIDTH
I1_EVENT_NAMES = tuple(f"d_i1px{index}" for index in range(22)) + (
    "ev_i1px",
    "ev_i1vol",
    "ev_i1trade",
    "since_i1px",
)


def i1_history_feature_names() -> tuple[str, ...]:
    """Return the stable 156-channel order for the raw i1 history block."""
    return (
        tuple(f"i1_residual_h32_{index:02d}" for index in range(I1_CHANNELS))
        + tuple(f"i1_residual_h512_{index:02d}" for index in range(I1_CHANNELS))
        + tuple(f"i1_log_energy_h512_{index:02d}" for index in range(I1_CHANNELS))
    )


def validate_i1_history_prior(prior: Mapping[str, object]) -> dict[str, object]:
    """Validate the fitting-only scale and frozen causal-history definition."""
    if prior.get("schema") != I1_HISTORY_SCHEMA:
        raise ValueError("i1 history prior has an unsupported schema")
    scale = np.asarray(prior.get("scale"), dtype=np.float32)
    if scale.shape != (I1_CHANNELS,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("i1 history prior must contain 52 finite positive channel scales")
    if prior.get("scale_estimator") != "within_sequence_first_difference_rms_v1":
        raise ValueError("i1 history prior has an unsupported scale estimator")
    if prior.get("scale_denominator") != "all_adjacent_transitions_within_selected_sequences":
        raise ValueError("i1 history prior has an unsupported scale denominator")
    if float(prior.get("scale_floor", float("nan"))) != float(I1_HISTORY_SCALE_FLOOR):
        raise ValueError("i1 history prior has an unsupported scale floor")
    if prior.get("fast_half_life_rows") != I1_HISTORY_FAST_HALF_LIFE:
        raise ValueError("i1 history prior has an unsupported fast half-life")
    if prior.get("slow_half_life_rows") != I1_HISTORY_SLOW_HALF_LIFE:
        raise ValueError("i1 history prior has an unsupported slow half-life")
    if prior.get("block_rows") != I1_HISTORY_BLOCK_ROWS:
        raise ValueError("i1 history prior has an unsupported scan block size")
    result = dict(prior)
    result["scale"] = scale.tolist()
    return result


def fit_i1_history_prior(raw_sequences: Iterable[np.ndarray]) -> dict[str, object]:
    """Fit channel scales from within-sequence i1 differences only.

    Each supplied sequence contributes all of its row-to-row transitions. The
    caller binds source-qualified fitting membership and its deterministic sample.
    """
    squared_sum = np.zeros(I1_CHANNELS, dtype=np.float64)
    transition_count = 0
    sequence_count = 0
    for raw in raw_sequences:
        sequence = np.asarray(raw, dtype=np.float32)
        if (
            sequence.ndim != 2
            or sequence.shape[1] != N_FEATURES
            or sequence.shape[0] < 1
            or not np.isfinite(sequence).all()
        ):
            raise ValueError("Expected a nonempty finite fitting sequence with 112 raw columns")
        sequence_count += 1
        if sequence.shape[0] < 2:
            continue
        delta = np.diff(sequence[:, I1_RAW_START:I1_RAW_STOP], axis=0).astype(
            np.float64, copy=False
        )
        squared_sum += np.square(delta).sum(axis=0, dtype=np.float64)
        transition_count += delta.shape[0]
    if sequence_count == 0 or transition_count == 0:
        raise ValueError("cannot fit i1 history scales without fitting transitions")
    scale = np.sqrt(squared_sum / transition_count).astype(np.float32)
    scale = np.maximum(scale, I1_HISTORY_SCALE_FLOOR)
    return validate_i1_history_prior(
        {
            "schema": I1_HISTORY_SCHEMA,
            "scale": scale.tolist(),
            "scale_estimator": "within_sequence_first_difference_rms_v1",
            "transition_count": transition_count,
            "sequence_count": sequence_count,
            "scale_denominator": "all_adjacent_transitions_within_selected_sequences",
            "scale_floor": float(I1_HISTORY_SCALE_FLOOR),
            "fast_half_life_rows": I1_HISTORY_FAST_HALF_LIFE,
            "slow_half_life_rows": I1_HISTORY_SLOW_HALF_LIFE,
            "block_rows": I1_HISTORY_BLOCK_ROWS,
            "alpha_definition": "1 - 2**(-1/half_life_rows)",
            "mean_initialization": "first_observation",
            "energy_initialization": "zero",
            "energy_delta": "current_i1_row_minus_previous_i1_row",
            "first_row_history": "zero",
        }
    )


def i1_history_input_mask_indices(control: str, drop_i1: bool) -> tuple[int, ...]:
    """Return semantic input columns zeroed by the matched i1 comparison arms."""
    if control not in ("none", "all"):
        raise ValueError("i1 history control must be none or all")
    indices: set[int] = set()
    history_start = N_FEATURES + N_EVENT_FEATURES
    if control == "all":
        indices.update(range(history_start, I1_INPUT_WIDTH))
    if drop_i1:
        missing = set(I1_EVENT_NAMES).difference(FEATURE_NAMES)
        if missing:
            raise RuntimeError(f"event feature registry lacks i1 fields: {sorted(missing)}")
        indices.update(range(I1_RAW_START, I1_RAW_STOP))
        indices.update(N_FEATURES + FEATURE_NAMES.index(name) for name in I1_EVENT_NAMES)
        indices.update(range(history_start, I1_INPUT_WIDTH))
    return tuple(sorted(indices))


def _ema_weights(half_life: float) -> tuple[np.float32, np.float32]:
    """Return float32 decay and update weights for the declared half-life."""
    decay = np.float32(math.exp2(-1.0 / half_life))
    update = np.float32(1.0 - decay)
    return decay, update
