"""Named Muon/AdamW parameter groups for explicit recurrent adapter variants."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import nn

_GRU_GATE_PATTERN = re.compile(r"^gru\.weight_(ih|hh)_l\d+$")
_GALAM_ATTENTION_PATTERN = re.compile(
    r"^galam\.(local_attention|global_attention)\.\d+\."
    r"(query|key|value|output|relative_bias\.(hidden|output))\.weight$"
)
_GALAM_FEEDFORWARD_PATTERN = re.compile(r"^galam\.feed_forward\.(0|2)\.weight$")


def adapter_muon_group_specs(model: nn.Module, architecture: str) -> list[dict[str, object]]:
    """Build strict-FP32 hybrid groups for one approved recurrent adapter."""
    if architecture == "gru":
        return _gru_group_specs(model)
    if architecture == "galam":
        return _galam_group_specs(model)
    raise ValueError("adapter Muon groups support only GRU and GALAM models")


def optimizer_group_specs(
    roles: Mapping[str, list[tuple[str, torch.Tensor, bool]]],
    role_order: tuple[str, ...],
    muon_roles: Mapping[str, int],
    expected_ids: set[int],
) -> list[dict[str, object]]:
    """Create auditable groups and prove exact trainable-parameter ownership."""
    specs: list[dict[str, object]] = []
    for role in role_order:
        entries = roles.get(role, [])
        if not entries:
            continue
        muon = role in muon_roles
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
                "blocks": muon_roles.get(role, 1),
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


def _gru_group_specs(model: nn.Module) -> list[dict[str, object]]:
    """Use Muon for each packed three-gate GRU matrix and AdamW elsewhere."""
    recurrent = getattr(model, "gru", None)
    if not isinstance(recurrent, nn.GRU):
        raise ValueError("GRU Muon requires an nn.GRU at model.gru")
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("sequence model has no trainable parameters")
    roles: dict[str, list[tuple[str, torch.Tensor, bool]]] = defaultdict(list)
    muon_names: set[str] = set()
    for name, parameter in named_parameters:
        if _GRU_GATE_PATTERN.fullmatch(name):
            if parameter.ndim != 2 or parameter.shape[0] % 3:
                raise ValueError(f"GRU gate matrix {name} must contain three stacked gate blocks")
            role, decay_enabled = "muon_gru_gates", True
            muon_names.add(name)
        else:
            role, decay_enabled = "adamw_other", True
        roles[role].append((name, parameter, decay_enabled))
    expected_names = {
        f"gru.weight_{kind}_l{layer}"
        for layer in range(recurrent.num_layers)
        for kind in ("ih", "hh")
    }
    if muon_names != expected_names:
        raise ValueError("every trainable GRU layer must expose both gate matrices for Muon")
    expected_ids = {id(parameter) for _, parameter in named_parameters}
    return optimizer_group_specs(
        roles,
        ("muon_gru_gates", "adamw_other"),
        {"muon_gru_gates": 3},
        expected_ids,
    )


def _galam_group_specs(model: nn.Module) -> list[dict[str, object]]:
    """Use Muon only on full-GALAM temporal-attention and decoder FFN matrices."""
    variant = getattr(model, "variant", None)
    galam = getattr(model, "galam", None)
    if variant != "schematic_lstm_full" or not isinstance(galam, nn.Module):
        raise ValueError("GALAM hybrid Muon requires the full LSTM schematic variant")
    local_attention = getattr(galam, "local_attention", None)
    global_attention = getattr(galam, "global_attention", None)
    feed_forward = getattr(galam, "feed_forward", None)
    if not isinstance(local_attention, nn.ModuleList) or not isinstance(
        global_attention, nn.ModuleList
    ):
        raise ValueError("GALAM hybrid Muon requires local and global attention stacks")
    if (
        not len(local_attention)
        or not len(global_attention)
        or not isinstance(feed_forward, nn.Sequential)
    ):
        raise ValueError(
            "GALAM hybrid Muon requires attention blocks and a sequential feed-forward path"
        )
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("sequence model has no trainable parameters")
    roles: dict[str, list[tuple[str, torch.Tensor, bool]]] = defaultdict(list)
    muon_names: set[str] = set()
    for name, parameter in named_parameters:
        if _GALAM_ATTENTION_PATTERN.fullmatch(name):
            role, decay_enabled = "muon_attention", True
            muon_names.add(name)
        elif _GALAM_FEEDFORWARD_PATTERN.fullmatch(name):
            role, decay_enabled = "muon_feedforward", True
            muon_names.add(name)
        else:
            role, decay_enabled = "adamw_other", True
        if role in ("muon_attention", "muon_feedforward") and parameter.ndim != 2:
            raise ValueError(f"GALAM Muon parameter {name} must be a matrix")
        roles[role].append((name, parameter, decay_enabled))
    attention_prefixes = (
        *(f"galam.local_attention.{index}." for index in range(len(local_attention))),
        *(f"galam.global_attention.{index}." for index in range(len(global_attention))),
    )
    attention_suffixes = (
        "query.weight",
        "key.weight",
        "value.weight",
        "output.weight",
        "relative_bias.hidden.weight",
        "relative_bias.output.weight",
    )
    expected_names = {
        prefix + suffix for prefix in attention_prefixes for suffix in attention_suffixes
    }
    expected_names.update({"galam.feed_forward.0.weight", "galam.feed_forward.2.weight"})
    if muon_names != expected_names:
        raise ValueError("GALAM attention/FFN Muon matrix coverage differs from the declared graph")
    if any(name.startswith("galam.lstm.") for name in muon_names):
        raise ValueError("GALAM LSTM parameters must remain on AdamW")
    expected_ids = {id(parameter) for _, parameter in named_parameters}
    return optimizer_group_specs(
        roles,
        ("muon_attention", "muon_feedforward", "adamw_other"),
        {"muon_attention": 1, "muon_feedforward": 1},
        expected_ids,
    )
