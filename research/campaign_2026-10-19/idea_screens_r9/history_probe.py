"""Prepare causal inputs and pool the fixed recurrent carry windows."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from idea_screens_r6.model_contract import state_fingerprint
from idea_screens_r6.sample_panel import FitSequence
from torch import nn

from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs

from .history_math import PredictionDifference
from .state_replay import advance, zero_state
from .window_probe import ALTERNATIVES, SNAPSHOT_K, compare_window

INPUT_ROWS = 16_512
INPUT_WIDTH = 211
ANCHORS = (8_192, 16_384)
WINDOW_ROWS = 128
RECENT_ROWS = 256


def _prepare_inputs(environment: TrainingEnvironment, sequence: FitSequence) -> torch.Tensor:
    """Build one sequence's full legal causal feature history without reading labels."""
    if sequence.x.dtype != np.dtype(np.float32) or sequence.x.shape != (20_000, 112):
        raise ValueError("selected raw input array must have shape [20000,112] and float32 dtype")
    raw_array = np.array(sequence.x[:INPUT_ROWS], dtype=np.float32, copy=True)
    if raw_array.shape != (INPUT_ROWS, 112) or not np.isfinite(raw_array).all():
        raise ValueError("selected causal raw input prefix is malformed or nonfinite")
    raw = torch.from_numpy(raw_array).unsqueeze(0)
    inputs = build_inputs(raw, environment=environment)
    del raw, raw_array
    if (
        inputs.shape != (1, INPUT_ROWS, INPUT_WIDTH)
        or inputs.dtype != torch.float32
        or inputs.device.type != "cpu"
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError("canonical causal inputs must be finite CPU FP32 [1,16512,211]")
    return inputs.contiguous()


def _new_pooled() -> dict[str, dict[str, PredictionDifference]]:
    """Create fixed branch/transform accumulator pairs for all windows."""
    return {
        branch: {kind: PredictionDifference() for kind in ("raw", "clipped")}
        for branch in ALTERNATIVES
    }


def _model_checks(model: nn.Module) -> str:
    """Require CPU FP32 inference state and return its exact initial fingerprint."""
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("R9 model must remain CPU FP32")
    model.eval()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("R9 requires empty model .grad fields at start")
    return state_fingerprint(model)


def _expected_traversal(counters: dict[str, int]) -> int:
    """Validate fixed row totals and return total model-forward rows."""
    expected = {"full_prefix_rows": 130_048, "recent_reset_rows": 4_096, "rollout_rows": 6_144}
    if any(counters[key] != value for key, value in expected.items()):
        raise RuntimeError("R9 traversal rows differ from the frozen 8-sequence design")
    return sum(counters[key] for key in expected)


def run_history_probe(
    model: nn.Module,
    environment: TrainingEnvironment,
    sequences: Sequence[FitSequence],
    memory_guard: Callable[[str], dict[str, int | str | None]],
    progress: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    """Measure fixed-checkpoint recurrent carry dependence without labels or updates."""
    if len(sequences) != 8:
        raise ValueError("R9 requires the first eight fixed R3 fit sequences")
    fingerprint_before = _model_checks(model)
    counters = {
        "full_prefix_rows": 0,
        "recent_reset_rows": 0,
        "rollout_rows": 0,
        "model_forward_calls": 0,
        "prepared_input_rows": 0,
    }
    pooled = _new_pooled()
    windows: list[dict[str, object]] = []
    phase_seconds = {"input_preparation": 0.0, "model_forward": 0.0}
    for sequence_index, sequence in enumerate(sequences):
        memory_guard(f"before_input_preparation_{sequence_index}")
        prepared_started = time.perf_counter()
        inputs = _prepare_inputs(environment, sequence)
        phase_seconds["input_preparation"] += time.perf_counter() - prepared_started
        counters["prepared_input_rows"] += INPUT_ROWS
        full_state: torch.Tensor | None = None
        cursor = 0
        for anchor in ANCHORS:
            forward_started = time.perf_counter()
            full_state = advance(model, inputs, full_state, cursor, anchor, "full_prefix", counters)
            recent_state = advance(
                model,
                inputs,
                zero_state(model),
                anchor - RECENT_ROWS,
                anchor,
                "recent_reset",
                counters,
            )
            output = compare_window(
                model,
                inputs,
                anchor,
                full_state,
                recent_state,
                zero_state(model),
                counters,
            )
            phase_seconds["model_forward"] += time.perf_counter() - forward_started
            full_state = output.full_state_after
            cursor = anchor + WINDOW_ROWS
            component = int(sequence.record["component"])
            window_record = {
                "sequence_index": sequence_index,
                "component": component,
                "source": str(sequence.record["source"]),
                "group": int(sequence.record["group"]),
                "seq_ix": int(sequence.record["seq_ix"]),
                **output.record,
            }
            windows.append(window_record)
            for branch in ALTERNATIVES:
                for kind in ("raw", "clipped"):
                    pooled[branch][kind].merge(output.prediction[branch][kind])
            progress(
                {
                    "event": "window_completed",
                    "sequence_index": sequence_index,
                    "component": component,
                    "anchor": anchor,
                    "model_forward_rows": sum(
                        counters[key]
                        for key in ("full_prefix_rows", "recent_reset_rows", "rollout_rows")
                    ),
                }
            )
        del inputs
    if len(windows) != 16:
        raise RuntimeError("R9 did not complete all 16 fixed windows")
    total_rows = _expected_traversal(counters)
    fingerprint_after = state_fingerprint(model)
    if fingerprint_before != fingerprint_after or any(
        parameter.grad is not None for parameter in model.parameters()
    ):
        raise RuntimeError("R9 model state changed or model .grad fields were populated")
    pooled_output = {
        branch: {kind: value.summary() for kind, value in summaries.items()}
        for branch, summaries in pooled.items()
    }
    return {
        "status": "completed_no_update_carried_state_diagnostic",
        "selected_sequences": len(sequences),
        "window_count": len(windows),
        "anchors": list(ANCHORS),
        "snapshot_k": [0, *SNAPSHOT_K],
        "windows": windows,
        "pooled_prediction_differences": pooled_output,
        "traversal_counts": {
            **counters,
            "model_forward_rows_total": total_rows,
            "expected_model_forward_rows_total": 140_288,
            "model_forward_rows_match_design": total_rows == 140_288,
        },
        "model_state_fingerprint_before": fingerprint_before,
        "model_state_fingerprint_after": fingerprint_after,
        "model_state_unchanged": fingerprint_before == fingerprint_after,
        "parameter_grad_fields_empty": True,
        "target_values_accessed": False,
        "scorer_called": False,
        "backward_or_optimizer_used": False,
        "phase_seconds": phase_seconds,
    }
