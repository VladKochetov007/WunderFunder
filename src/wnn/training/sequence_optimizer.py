"""Strict-FP32 Muon groups for causal Transformer sequence fits."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import nn

from wnn.optim import MuonAdamW

_QKV_PATTERN = re.compile(r"^blocks\.(\d+)\.qkv\.weight$")
_MUON_ROLES: dict[str, int] = {
    "muon_qkv": 3,
    "muon_hidden": 1,
}


def _parameter_role(name: str, parameter: torch.Tensor) -> tuple[str, bool]:
    """Assign a named Transformer parameter to a semantic optimizer group."""
    if _QKV_PATTERN.fullmatch(name):
        return "muon_qkv", True
    if re.fullmatch(r"blocks\.\d+\.(attention_output|feedforward_(in|gate|out))\.weight", name):
        return "muon_hidden", True
    if any(
        marker in name
        for marker in ("output_norm.", ".norm_attention.", ".norm_feedforward.", ".normalizer.")
    ):
        return "adamw_no_decay", False
    if name.startswith("blocks.") and name.endswith(".weight"):
        raise ValueError(f"unsupported trainable Transformer block matrix {name}")
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return "adamw_no_decay", False
    return "adamw_matrix", True


def _group_specs(model: nn.Module) -> list[dict[str, object]]:
    """Build ordered groups with exact trainable-parameter ownership."""
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("sequence Muon requires a Transformer model with nonempty blocks")
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("sequence model has no trainable parameters")

    activation = getattr(blocks[0], "ff_activation", "gelu")
    if activation not in ("gelu", "swiglu") or any(
        getattr(block, "ff_activation", "gelu") != activation for block in blocks
    ):
        raise ValueError("Transformer blocks must share a supported feed-forward activation")
    feedforward_names = ["feedforward_in", "feedforward_out"]
    if activation == "swiglu":
        feedforward_names.insert(1, "feedforward_gate")

    roles: dict[str, list[tuple[str, torch.Tensor, bool]]] = defaultdict(list)
    qkv_indices: set[int] = set()
    muon_names: set[str] = set()
    for name, parameter in named_parameters:
        match = _QKV_PATTERN.fullmatch(name)
        role, decay_enabled = _parameter_role(name, parameter)
        if match:
            block_index = int(match.group(1))
            qkv_indices.add(block_index)
            if parameter.ndim != 2 or parameter.shape[0] != 3 * parameter.shape[1]:
                raise ValueError(f"{name} must stack three square D-by-D projection matrices")
        if role in _MUON_ROLES and parameter.ndim != 2:
            raise ValueError(f"Muon parameter {name} must be a matrix")
        if role in _MUON_ROLES:
            muon_names.add(name)
        roles[role].append((name, parameter, decay_enabled))
    if qkv_indices != set(range(len(blocks))):
        raise ValueError("every Transformer block must expose a trainable qkv.weight matrix")
    expected_muon_names = {
        f"blocks.{index}.{suffix}.weight"
        for index in range(len(blocks))
        for suffix in (
            "qkv",
            "attention_output",
            *feedforward_names,
        )
    }
    parameter_shapes = {name: tuple(parameter.shape) for name, parameter in named_parameters}
    for index, block in enumerate(blocks):
        hidden_width = block.feedforward_in.in_features
        feedforward_width = block.feedforward_in.out_features
        expected_shapes = {
            "feedforward_in": (feedforward_width, hidden_width),
            "feedforward_out": (hidden_width, feedforward_width),
        }
        if activation == "swiglu":
            expected_shapes["feedforward_gate"] = (feedforward_width, hidden_width)
        for suffix, shape in expected_shapes.items():
            name = f"blocks.{index}.{suffix}.weight"
            if parameter_shapes.get(name) != shape:
                raise ValueError(f"Muon projection {name} has an invalid separate matrix shape")
    if muon_names != expected_muon_names:
        raise ValueError("every Transformer block must expose the declared Muon matrix groups")

    expected_ids = {id(parameter) for _, parameter in named_parameters}
    owned_ids = [id(parameter) for entries in roles.values() for _, parameter, _ in entries]
    if len(owned_ids) != len(set(owned_ids)) or set(owned_ids) != expected_ids:
        raise ValueError("sequence Muon groups must own every trainable parameter exactly once")

    specs: list[dict[str, object]] = []
    role_order = (*_MUON_ROLES, "adamw_matrix", "adamw_no_decay")
    for role in role_order:
        entries = roles.get(role, [])
        if not entries:
            continue
        muon = role in _MUON_ROLES
        decay_enabled = entries[0][2]
        if any(entry[2] != decay_enabled for entry in entries):
            raise ValueError(f"optimizer role {role} has inconsistent weight-decay policy")
        specs.append(
            {
                "params": [parameter for _, parameter, _ in entries],
                "param_names": [name for name, _, _ in entries],
                "role": role,
                "group_name": role,
                "optimizer_implementation": "wnn.optim.MuonAdamW",
                "optimizer_group_schema": "wnn.sequence-muon-group.v1",
                "muon": muon,
                "blocks": _MUON_ROLES.get(role, 1),
                "weight_decay_enabled": decay_enabled,
                "strict_fp32": True,
            }
        )
    spec_ids = [id(parameter) for group in specs for parameter in group["params"]]
    if len(spec_ids) != len(set(spec_ids)) or set(spec_ids) != expected_ids:
        raise ValueError(
            "sequence optimizer groups must own every trainable parameter exactly once"
        )
    return specs


def build_sequence_muon_optimizer(model: nn.Module, config: Mapping[str, object]) -> MuonAdamW:
    """Create one resumable Muon/AdamW optimizer for a strict-FP32 Transformer.

    Args:
        model: A built sequence Transformer with named causal attention blocks.
        config: Validated sequence recipe with Muon, AdamW, and shared learning-rate settings.

    Returns:
        A ``MuonAdamW`` instance whose parameter groups record roles and names in state dictionaries.

    Raises:
        TypeError: If a configured optimizer value has the wrong type.
        ValueError: If the model groups or optimizer settings are invalid.
    """
    learning_rate = _positive_number(config, "lr")
    weight_decay = _nonnegative_number(config, "weight_decay")
    muon_momentum = _unit_interval(config, "muon_momentum")
    adamw_beta1 = _unit_interval(config, "adamw_beta1")
    adamw_beta2 = _unit_interval(config, "adamw_beta2")
    adamw_eps = _positive_number(config, "adamw_eps")
    ns_eps = _positive_number(config, "muon_ns_eps")
    ns_steps = config.get("muon_ns_steps")
    if type(ns_steps) is not int or ns_steps < 1:
        raise ValueError("muon_ns_steps must be a positive integer")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError("sequence Muon config requires a model mapping")
    if model_config.get("architecture") != "transformer":
        raise ValueError("sequence Muon is scoped to Transformer models")

    groups = _group_specs(model)
    for group in groups:
        role = str(group["role"])
        muon = bool(group["muon"])
        group.update(
            lr=learning_rate,
            weight_decay=weight_decay if group["weight_decay_enabled"] else 0.0,
            betas=(muon_momentum, muon_momentum) if muon else (adamw_beta1, adamw_beta2),
            eps=adamw_eps,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            rms=0.2,
            optimizer_role=role,
        )
    return MuonAdamW(
        groups,
        lr=learning_rate,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_eps,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
        rms=0.2,
        strict_fp32=True,
        ns_eps=ns_eps,
    )


def _number(config: Mapping[str, object], name: str) -> float:
    """Read one finite numeric optimizer setting while rejecting booleans."""
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_number(config: Mapping[str, object], name: str) -> float:
    """Read one finite, strictly positive optimizer setting."""
    value = _number(config, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_number(config: Mapping[str, object], name: str) -> float:
    """Read one finite, nonnegative optimizer setting."""
    value = _number(config, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _unit_interval(config: Mapping[str, object], name: str) -> float:
    """Read one finite optimizer coefficient in the open unit interval."""
    value = _number(config, name)
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1)")
    return value
