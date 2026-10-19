from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from attrs import frozen
from gradient_math import add_weighted_pair, pair_summary, pooled_summary, target_loss_parts
from model_contract import parameter_inventory, state_fingerprint
from sample_panel import FitSequence
from torch import nn

from wnn.nn import weighted_mse_loss
from wnn.training.environment import TrainingEnvironment
from wnn.training.forward import detach_state, forward_chunk
from wnn.training.inputs import build_inputs

WINDOW_STARTS = (1024, 4096)
WINDOW_LENGTH = 128
MAX_FORWARD_CHUNK = 512
PREFIX_ROWS = 4224


@frozen
class PreparedSequence:
    """Causal model inputs and the two original-masked target windows."""

    record: dict[str, object]
    inputs: torch.Tensor
    windows: tuple[tuple[int, torch.Tensor, torch.Tensor], ...]


def prepare_sequences(
    model: nn.Module, environment: TrainingEnvironment, sequences: Sequence[FitSequence]
) -> list[PreparedSequence]:
    """Build raw-plus-causal-event inputs once for each selected complete prefix."""
    if len(sequences) != 8:
        raise ValueError("R6 requires exactly the fixed first eight fit sequences")
    prepared: list[PreparedSequence] = []
    for sequence in sequences:
        if sequence.scored[:99].any():
            raise ValueError("source scored mask marks warmup rows")
        raw_array = np.array(sequence.x[:PREFIX_ROWS], dtype=np.float32, copy=True)
        if raw_array.shape != (PREFIX_ROWS, 112) or not np.isfinite(raw_array).all():
            raise ValueError("selected causal input prefix is malformed or nonfinite")
        raw = torch.from_numpy(raw_array).unsqueeze(0)
        inputs = build_inputs(raw, environment=environment)
        if (
            inputs.shape != (1, PREFIX_ROWS, 211)
            or inputs.dtype != torch.float32
            or not torch.isfinite(inputs).all()
        ):
            raise ValueError("canonical R01 inputs must be finite FP32 [1,4224,211]")
        windows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for start in WINDOW_STARTS:
            end = start + WINDOW_LENGTH
            target_array = np.array(sequence.y[start:end], dtype=np.float32, copy=True)
            scored_array = np.array(sequence.scored[start:end], dtype=np.bool_, copy=True)
            if target_array.shape != (WINDOW_LENGTH, 2) or not np.isfinite(target_array).all():
                raise ValueError("diagnostic target window is malformed or nonfinite")
            windows.append(
                (
                    start,
                    torch.from_numpy(target_array).unsqueeze(0),
                    torch.from_numpy(scored_array).unsqueeze(0),
                )
            )
        prepared.append(PreparedSequence(sequence.record, inputs.contiguous(), tuple(windows)))
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("R6 model must remain CPU FP32")
    return prepared


def _advance_without_grad(
    model: nn.Module, inputs: torch.Tensor, state: torch.Tensor | None, cursor: int, stop: int
) -> torch.Tensor:
    """Replay a causal prefix in bounded chunks and return its detached GRU state."""
    if cursor < 0 or stop < cursor or stop > inputs.shape[1]:
        raise ValueError("invalid causal prefix interval")
    with torch.no_grad():
        while cursor < stop:
            end = min(cursor + MAX_FORWARD_CHUNK, stop)
            _, state = forward_chunk(model, inputs, slice(cursor, end), state)
            cursor = end
    return detach_state(state)


def run_diagnostic(
    model: nn.Module,
    prepared: Sequence[PreparedSequence],
    memory_guard: Callable[[str], dict[str, int | str | None]],
    progress: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    """Measure separate per-target gradients at the fixed endpoint without updating it."""
    parameters, blocks, inventory = parameter_inventory(model)
    shared_indices = [index for name in blocks for index in blocks[name]]
    shared_position = {
        parameter_index: position for position, parameter_index in enumerate(shared_indices)
    }
    shared_parameters = [parameters[index][1] for index in shared_indices]
    fingerprint_before = state_fingerprint(model)
    pools: dict[str, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}
    for name, indices in blocks.items():
        zero = tuple(
            torch.zeros_like(shared_parameters[index], dtype=torch.float64) for index in indices
        )
        pools[name] = zero, zero
    windows: list[dict[str, object]] = []
    total_weight = 0.0
    phase_seconds = {"prefix": 0.0, "gradient": 0.0}
    for item in prepared:
        state: torch.Tensor | None = None
        cursor = 0
        for start, target, scored in item.windows:
            prefix_started = time.perf_counter()
            state = _advance_without_grad(model, item.inputs, state, cursor, start)
            phase_seconds["prefix"] += time.perf_counter() - prefix_started
            cursor = start
            memory = memory_guard("before_gradient")
            gradient_started = time.perf_counter()
            prediction, state_after = forward_chunk(
                model, item.inputs, slice(start, start + WINDOW_LENGTH), state
            )
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise FloatingPointError("diagnostic prediction is malformed or nonfinite")
            loss0, loss1, weight_tensor, clipped = target_loss_parts(prediction, target, scored)
            canonical = weighted_mse_loss(prediction, target, scored)
            if not torch.allclose(loss0 + loss1, canonical, rtol=2e-6, atol=1e-7):
                raise RuntimeError("per-target WMSE decomposition differs from canonical loss")
            weight = float(weight_tensor.detach())
            weight_by_target = (clipped.abs() * scored[..., None]).sum(dim=(0, 1))
            gradients0 = torch.autograd.grad(loss0, shared_parameters, retain_graph=True)
            gradients1 = torch.autograd.grad(loss1, shared_parameters)
            if not all(torch.isfinite(gradient).all() for gradient in (*gradients0, *gradients1)):
                raise FloatingPointError("nonfinite per-target parameter gradient")
            block_results: dict[str, object] = {}
            for name, indices in blocks.items():
                positions = [shared_position[index] for index in indices]
                gradient0 = tuple(gradients0[index] for index in positions)
                gradient1 = tuple(gradients1[index] for index in positions)
                local = pair_summary(gradient0, gradient1)
                running0, running1 = pools[name]
                next0, next1, pool_audit = add_weighted_pair(
                    running0, running1, gradient0, gradient1, weight
                )
                pools[name] = next0, next1
                block_results[name] = {**local, "pool_audit": pool_audit}
            elapsed_gradient = time.perf_counter() - gradient_started
            phase_seconds["gradient"] += elapsed_gradient
            record = {
                "component": int(item.record["component"]),
                "source": str(item.record["source"]),
                "group": int(item.record["group"]),
                "seq_ix": int(item.record["seq_ix"]),
                "window_start": start,
                "window_rows": WINDOW_LENGTH,
                "scored_rows": int(scored.sum()),
                "joint_weight_mass": weight,
                "target_weight_mass": [float(value) for value in weight_by_target.detach()],
                "target_loss_contribution": [float(loss0.detach()), float(loss1.detach())],
                "blocks": block_results,
                "timing_s": {"prefix": phase_seconds["prefix"], "gradient": elapsed_gradient},
                "memory": memory,
            }
            windows.append(record)
            progress(
                {
                    "component": record["component"],
                    "window_start": start,
                    "elapsed_gradient_s": elapsed_gradient,
                    "memory": memory,
                }
            )
            total_weight += weight
            cursor = start + WINDOW_LENGTH
            state = detach_state(state_after)
            del prediction, state_after, loss0, loss1, canonical, gradients0, gradients1
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("autograd.grad unexpectedly populated parameter .grad")
    if len(windows) != 16 or total_weight <= 0.0:
        raise RuntimeError("fixed R6 diagnostic did not complete all 16 positive-mass windows")
    pooled = {name: pooled_summary(*pair, total_weight) for name, pair in pools.items()}
    cosine_counts: dict[str, dict[str, int]] = {}
    heterogeneity: dict[str, dict[str, float | int | None]] = {}
    for name in blocks:
        cosines = [record["blocks"][name]["gradient_cosine"] for record in windows]
        observed = [float(value) for value in cosines if value is not None]
        cosine_counts[name] = {
            "window_count_denominator": len(windows),
            "defined_cosine_windows": len(observed),
            "negative_cosine_windows": sum(value < 0.0 for value in observed),
            "cosine_below_minus_0_1_windows": sum(value < -0.1 for value in observed),
        }
        heterogeneity[name] = {
            "min_cosine": min(observed) if observed else None,
            "median_cosine": float(np.median(observed)) if observed else None,
            "max_cosine": max(observed) if observed else None,
        }
    fingerprint_after = state_fingerprint(model)
    if fingerprint_before != fingerprint_after or any(
        parameter.grad is not None for parameter in model.parameters()
    ):
        raise RuntimeError("model state changed or .grad fields were populated")
    return {
        "status": "completed_fixed_no_update_diagnostic",
        "windows": windows,
        "window_count": len(windows),
        "weight_mass_total": total_weight,
        "pooled_by_parameter_block": pooled,
        "cosine_counts": cosine_counts,
        "cosine_heterogeneity": heterogeneity,
        "parameter_inventory": inventory,
        "model_state_fingerprint_before": fingerprint_before,
        "model_state_fingerprint_after": fingerprint_after,
        "model_state_unchanged": fingerprint_before == fingerprint_after,
        "parameter_grad_fields_empty": True,
        "phase_seconds": phase_seconds,
    }
