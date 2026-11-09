"""Explicit optimizer-role and head/body learning-rate factors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

LR_CONTROL_KEYS = frozenset(("lr_role_factors", "lr_head_factor", "lr_body_factor"))


def validate_lr_factor_config(config: Mapping[str, object]) -> None:
    """Validate optional factors before model or optimizer construction."""
    factors = config.get("lr_role_factors")
    if factors is not None:
        if not isinstance(factors, Mapping) or not factors:
            raise ValueError("lr_role_factors must be a nonempty role-to-factor mapping")
        for role, factor in factors.items():
            if not isinstance(role, str) or not role:
                raise ValueError("lr_role_factors keys must be nonempty role names")
            _positive_factor(factor, f"lr_role_factors.{role}")
    for key in ("lr_head_factor", "lr_body_factor"):
        if key in config:
            _positive_factor(config[key], key)


def _positive_factor(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _partition(name: str) -> str:
    return "head" if name.startswith(("head.", "output_head.")) else "body"


def apply_optimizer_lr_factors(
    groups: Sequence[dict[str, Any]],
    model: Any,
    config: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Return scaled groups, optionally splitting head/body within each optimizer role."""
    validate_lr_factor_config(config)
    if not any(key in config for key in LR_CONTROL_KEYS):
        return list(groups)
    factor_value = config.get("lr_role_factors", {})
    role_factors = dict(factor_value) if isinstance(factor_value, Mapping) else {}
    actual_roles = {
        str(group.get("role", group.get("optimizer_role", "adamw"))) for group in groups
    }
    unused_roles = set(role_factors) - actual_roles
    if unused_roles:
        raise ValueError(f"lr_role_factors names unknown or unused roles: {sorted(unused_roles)}")

    split_head_body = "lr_head_factor" in config or "lr_body_factor" in config
    head_factor = _positive_factor(config.get("lr_head_factor", 1.0), "lr_head_factor")
    body_factor = _positive_factor(config.get("lr_body_factor", 1.0), "lr_body_factor")
    name_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(name_by_id) != sum(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("optimizer LR controls require unique model parameter names")

    result: list[dict[str, Any]] = []
    found_head = False
    found_body = False
    for group in groups:
        role = str(group.get("role", group.get("optimizer_role", "adamw")))
        base_name = str(group.get("group_name", role))
        params = list(group["params"])
        names = list(group.get("param_names", ()))
        if not names:
            try:
                names = [name_by_id[id(parameter)] for parameter in params]
            except KeyError as error:
                raise ValueError(
                    "optimizer LR group contains a parameter outside the model"
                ) from error
        if len(names) != len(params):
            raise ValueError(f"optimizer group {base_name} parameter names do not align")
        if split_head_body:
            partitions = {"body": [], "head": []}
            partition_names = {"body": [], "head": []}
            for name, parameter in zip(names, params, strict=True):
                kind = _partition(name)
                partitions[kind].append(parameter)
                partition_names[kind].append(name)
                found_head |= kind == "head"
                found_body |= kind == "body"
            pieces = [
                (kind, partitions[kind], partition_names[kind])
                for kind in ("body", "head")
                if partitions[kind]
            ]
        else:
            pieces = [(None, params, names)]
        for partition, piece_params, piece_names in pieces:
            updated = dict(group)
            updated["params"] = piece_params
            updated["param_names"] = piece_names
            updated["role"] = role
            updated["optimizer_role"] = role
            updated["group_name"] = f"{base_name}_{partition}" if partition else base_name
            if partition is not None:
                updated["parameter_partition"] = partition
            multiplier = _positive_factor(role_factors.get(role, 1.0), f"lr_role_factors.{role}")
            multiplier *= head_factor if partition == "head" else body_factor
            base_lr = float(group["initial_lr"] if "initial_lr" in group else group["lr"])
            updated["lr"] = base_lr * multiplier
            if "initial_lr" in group:
                updated["initial_lr"] = base_lr * multiplier
            result.append(updated)
    if split_head_body and (not found_head or not found_body):
        raise ValueError("head/body LR factors require both head and body parameters")
    result_roles = {str(group["role"]) for group in result}
    if result_roles != actual_roles:
        raise ValueError("optimizer LR grouping changed the declared optimizer roles")
    return result
