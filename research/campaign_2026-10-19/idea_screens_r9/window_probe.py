"""Independent fixed-window rollout and scalar branch comparisons."""

from __future__ import annotations

import numpy as np
import torch
from attrs import frozen
from torch import nn

from wnn.training.forward import detach_state, forward_chunk

from .history_math import PredictionDifference, state_distance
from .state_replay import check_state, clone_state

WINDOW_ROWS = 128
SNAPSHOT_K = (1, 16, 64, 128)
BRANCHES = ("full", "recent256", "zero")
ALTERNATIVES = ("recent256", "zero")


@frozen
class WindowOutput:
    """One window's scalar evidence and its FULL state for stream continuation."""

    record: dict[str, object]
    full_state_after: torch.Tensor
    prediction: dict[str, dict[str, PredictionDifference]]


def _numpy(value: torch.Tensor) -> np.ndarray:
    """Expose a temporary CPU view for scalar accumulation only."""
    return value.detach().cpu().numpy()


def _state_snapshot(
    model: nn.Module, full_state: torch.Tensor, branch_states: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Record state norms and each alternative's distance from FULL."""
    check_state(model, full_state)
    reference = _numpy(full_state)
    return {
        "full": state_distance(reference, reference),
        "recent256": state_distance(reference, _numpy(branch_states["recent256"])),
        "zero": state_distance(reference, _numpy(branch_states["zero"])),
    }


def _prediction_accumulators() -> dict[str, dict[str, PredictionDifference]]:
    """Create raw and clipped accumulators for the two fixed alternatives."""
    return {
        branch: {"raw": PredictionDifference(), "clipped": PredictionDifference()}
        for branch in ALTERNATIVES
    }


def compare_window(
    model: nn.Module,
    inputs: torch.Tensor,
    anchor: int,
    full_start: torch.Tensor,
    recent_start: torch.Tensor,
    zero_start: torch.Tensor,
    counters: dict[str, int],
) -> WindowOutput:
    """Roll independent states across one 128-row interval with identical inputs."""
    for state in (full_start, recent_start, zero_start):
        check_state(model, state)
    states = {
        "full": clone_state(full_start),
        "recent256": clone_state(recent_start),
        "zero": clone_state(zero_start),
    }
    record: dict[str, object] = {
        "anchor": anchor,
        "window_rows": WINDOW_ROWS,
        "state_snapshots": {"0": _state_snapshot(model, states["full"], states)},
        "endpoint_predictions": [],
    }
    prediction_stats = _prediction_accumulators()
    previous_k = 0
    with torch.no_grad():
        for endpoint in SNAPSHOT_K:
            predictions: dict[str, torch.Tensor] = {}
            for branch in BRANCHES:
                prediction, next_state = forward_chunk(
                    model,
                    inputs,
                    slice(anchor + previous_k, anchor + endpoint),
                    states[branch],
                )
                if (
                    prediction.shape != (1, endpoint - previous_k, 2)
                    or prediction.dtype != torch.float32
                    or prediction.device.type != "cpu"
                    or not torch.isfinite(prediction).all()
                ):
                    raise FloatingPointError("fixed rollout prediction is malformed or nonfinite")
                states[branch] = detach_state(next_state)
                check_state(model, states[branch])
                predictions[branch] = prediction
                counters["rollout_rows"] += endpoint - previous_k
                counters["model_forward_calls"] += 1
            snapshots = record["state_snapshots"]
            assert isinstance(snapshots, dict)
            snapshots[str(endpoint)] = _state_snapshot(model, states["full"], states)
            full = predictions["full"]
            endpoint_record: dict[str, object] = {"k": endpoint}
            for branch in ALTERNATIVES:
                alternative = predictions[branch]
                accumulator = prediction_stats[branch]
                raw_full, raw_alternative = _numpy(full), _numpy(alternative)
                accumulator["raw"].add(raw_full, raw_alternative)
                accumulator["clipped"].add(
                    _numpy(full.clamp(-2.0, 2.0)), _numpy(alternative.clamp(-2.0, 2.0))
                )
                full_row = raw_full[0, -1]
                alternative_row = raw_alternative[0, -1]
                delta = alternative_row - full_row
                endpoint_record[branch] = {
                    "target_differences": {
                        f"target_{target}": {
                            "absolute_difference": float(abs(delta[target])),
                            "relative_absolute_difference": float(
                                abs(delta[target]) / max(abs(full_row[target]), 1e-12)
                            ),
                        }
                        for target in range(2)
                    },
                }
            endpoint_record["full_prediction_l2"] = float(
                np.linalg.norm(_numpy(full)[0, -1].astype(np.float64))
            )
            record["endpoint_predictions"].append(endpoint_record)
            previous_k = endpoint
    record["prediction_differences"] = {
        branch: {kind: value.summary() for kind, value in summaries.items()}
        for branch, summaries in prediction_stats.items()
    }
    return WindowOutput(record, states["full"], prediction_stats)
