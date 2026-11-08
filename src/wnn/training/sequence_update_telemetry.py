"""Opt-in sampled optimizer-step measurements for strict-FP32 sequence fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _ParameterSnapshot:
    parameter: torch.nn.Parameter
    value: torch.Tensor


@dataclass(frozen=True)
class _GroupSnapshot:
    name: str
    role: str
    partition: str | None
    parameters: tuple[_ParameterSnapshot, ...]


@dataclass(frozen=True)
class OptimizerUpdateSnapshot:
    """CPU copies of sampled pre-update weights; other updates allocate no copies."""

    update_step: int
    sampling_interval: int
    groups: tuple[_GroupSnapshot, ...]


def _group_name(group: dict[str, Any], index: int) -> str:
    return str(group.get("group_name", f"group_{index}"))


def capture_optimizer_update(
    optimizer: torch.optim.Optimizer,
    update_step: int,
    sampling_interval: int,
) -> OptimizerUpdateSnapshot | None:
    """Snapshot only update 1 and each Nth update, using one-based global steps."""
    if type(update_step) is not int or update_step < 1:
        raise ValueError("optimizer telemetry update_step must be a positive integer")
    if type(sampling_interval) is not int or sampling_interval < 1:
        raise ValueError("optimizer_update_diagnostics_every must be a positive integer")
    if update_step != 1 and update_step % sampling_interval:
        return None

    groups: list[_GroupSnapshot] = []
    names: set[str] = set()
    for index, group in enumerate(optimizer.param_groups):
        name = _group_name(group, index)
        if name in names:
            raise ValueError(f"optimizer telemetry group name is duplicated: {name}")
        names.add(name)
        parameters = tuple(
            _ParameterSnapshot(
                parameter,
                parameter.detach().to(device="cpu", dtype=torch.float32).clone(),
            )
            for parameter in group["params"]
        )
        groups.append(
            _GroupSnapshot(
                name,
                str(group.get("role", group.get("optimizer_role", "unknown"))),
                group.get("parameter_partition"),
                parameters,
            )
        )
    return OptimizerUpdateSnapshot(update_step, sampling_interval, tuple(groups))


def optimizer_gradient_statistics(
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict[str, float | int]]:
    """Measure per-group gradient L2 norm and both tensor/scalar counts without mutation."""
    output: dict[str, dict[str, float | int]] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = _group_name(group, index)
        squared_by_device: dict[torch.device, torch.Tensor] = {}
        tensor_count = 0
        scalar_count = 0
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            tensor_count += 1
            scalar_count += gradient.numel()
            values = gradient.detach().to(dtype=torch.float64)
            contribution = values.square().sum()
            previous = squared_by_device.get(gradient.device)
            squared_by_device[gradient.device] = (
                contribution if previous is None else previous + contribution
            )
        norm = math.sqrt(sum(float(value.item()) for value in squared_by_device.values()))
        output[name] = {
            "norm": norm,
            "parameter_tensor_count": tensor_count,
            "scalar_count": scalar_count,
        }
    return output


def _state_statistics(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    state_name: str,
) -> tuple[float, int, int]:
    squared_by_device: dict[torch.device, torch.Tensor] = {}
    tensor_count = 0
    scalar_count = 0
    for parameter in parameters:
        value = optimizer.state.get(parameter, {}).get(state_name)
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        scalar_count += value.numel()
        values = value.detach().to(dtype=torch.float64)
        contribution = values.square().sum()
        previous = squared_by_device.get(value.device)
        squared_by_device[value.device] = (
            contribution if previous is None else previous + contribution
        )
    norm = math.sqrt(sum(float(value.item()) for value in squared_by_device.values()))
    return norm, tensor_count, scalar_count


def finalize_optimizer_update(
    snapshot: OptimizerUpdateSnapshot,
    optimizer: torch.optim.Optimizer,
    applied_learning_rates: dict[str, float],
    preclip_gradient_stats: dict[str, dict[str, float | int]],
    postclip_gradient_stats: dict[str, dict[str, float | int]],
    preclip_gradient_norm: float,
) -> dict[str, object]:
    """Build one JSON-ready sampled event after the actual optimizer update."""
    group_map = {
        _group_name(group, index): group for index, group in enumerate(optimizer.param_groups)
    }
    if set(group_map) != {group.name for group in snapshot.groups}:
        raise ValueError("optimizer groups changed between telemetry snapshot and optimizer step")
    if set(applied_learning_rates) != set(group_map):
        raise ValueError("applied learning rates do not match optimizer telemetry groups")
    if set(preclip_gradient_stats) != set(group_map) or set(postclip_gradient_stats) != set(
        group_map
    ):
        raise ValueError("gradient metrics do not match optimizer telemetry groups")

    floor = 1e-12
    groups: dict[str, dict[str, object]] = {}
    preclip_squared = 0.0
    postclip_squared = 0.0
    preclip_tensors = postclip_tensors = preclip_scalars = postclip_scalars = 0
    for before in snapshot.groups:
        parameters = tuple(item.parameter for item in before.parameters)
        before_squared = torch.zeros((), dtype=torch.float64)
        delta_squared = torch.zeros((), dtype=torch.float64)
        scalar_count = 0
        for item in before.parameters:
            current = item.parameter.detach().to(device="cpu", dtype=torch.float32)
            delta = current - item.value
            before_squared += item.value.double().square().sum()
            delta_squared += delta.double().square().sum()
            scalar_count += item.value.numel()
        before_norm = math.sqrt(float(before_squared.item()))
        delta_norm = math.sqrt(float(delta_squared.item()))
        state: dict[str, object] = {}
        for key, state_name in (("momentum", "momentum_buffer"), ("exp_avg", "exp_avg")):
            norm, tensor_count, values_count = _state_statistics(optimizer, parameters, state_name)
            state[f"{key}_norm"] = norm
            state[f"{key}_parameter_tensor_count"] = tensor_count
            state[f"{key}_scalar_count"] = values_count

        preclip = preclip_gradient_stats[before.name]
        postclip = postclip_gradient_stats[before.name]
        preclip_norm = float(preclip["norm"])
        postclip_norm = float(postclip["norm"])
        preclip_squared += preclip_norm * preclip_norm
        postclip_squared += postclip_norm * postclip_norm
        preclip_tensors += int(preclip["parameter_tensor_count"])
        postclip_tensors += int(postclip["parameter_tensor_count"])
        preclip_scalars += int(preclip["scalar_count"])
        postclip_scalars += int(postclip["scalar_count"])

        groups[before.name] = {
            "role": before.role,
            "parameter_partition": before.partition,
            "applied_lr": float(applied_learning_rates[before.name]),
            "parameter_tensor_count": len(parameters),
            "parameter_scalar_count": scalar_count,
            "pre_weight_norm": before_norm,
            "pre_weight_rms": before_norm / math.sqrt(scalar_count) if scalar_count else 0.0,
            "delta_weight_norm": delta_norm,
            "delta_weight_rms": delta_norm / math.sqrt(scalar_count) if scalar_count else 0.0,
            "delta_weight_over_pre_weight_norm": delta_norm / max(before_norm, floor),
            "weight_norm_denominator_floor": floor,
            "zero_weight_denominator_count": int(before_norm <= floor),
            "preclip_gradient_norm": preclip_norm,
            "preclip_gradient_parameter_tensor_count": int(preclip["parameter_tensor_count"]),
            "preclip_gradient_scalar_count": int(preclip["scalar_count"]),
            "postclip_gradient_norm": postclip_norm,
            "postclip_gradient_parameter_tensor_count": int(postclip["parameter_tensor_count"]),
            "postclip_gradient_scalar_count": int(postclip["scalar_count"]),
            **state,
        }
    return {
        "sampled_update_step": snapshot.update_step,
        "sampling_interval": snapshot.sampling_interval,
        "preclip_gradient_norm": preclip_gradient_norm,
        "preclip_gradient_parameter_tensor_count": preclip_tensors,
        "preclip_gradient_scalar_count": preclip_scalars,
        "postclip_gradient_norm": math.sqrt(postclip_squared),
        "postclip_gradient_parameter_tensor_count": postclip_tensors,
        "postclip_gradient_scalar_count": postclip_scalars,
        "applied_learning_rates": dict(applied_learning_rates),
        "groups": groups,
    }
