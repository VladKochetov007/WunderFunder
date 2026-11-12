"""Server-only construction gate for canonical Muon/AdamW parameter ownership."""

from __future__ import annotations

from typing import Any

import torch

from bilinear_entmax.receiver_support import VARIANTS, build_probe_model

_INPUT_MATRIX_NAMES = (
    "input_projection.bilinear_u.weight",
    "input_projection.bilinear_v.weight",
    "input_projection.bilinear_output.weight",
)


def check_optimizer_ownership(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, dict[str, int]]:
    """Require exact ownership and declared AdamW roles for every adapter matrix."""
    from wnn.training.sequence_optimizer import build_sequence_muon_optimizer

    reports: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        model = build_probe_model(config, artifact, variant, seed, device)
        optimizer = build_sequence_muon_optimizer(model, config)
        groups = optimizer.param_groups
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        trainable_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != trainable_ids:
            raise AssertionError(f"{variant} Muon groups do not own every trainable parameter once")

        role_counts: dict[str, int] = {}
        name_roles: dict[str, str] = {}
        for group in groups:
            role = str(group.get("role"))
            role_counts[role] = role_counts.get(role, 0) + len(group["params"])
            names = group.get("param_names")
            if not isinstance(names, list) or len(names) != len(group["params"]):
                raise AssertionError(f"{variant} optimizer group lost parameter-name ownership")
            for name in names:
                if str(name) in name_roles:
                    raise AssertionError(f"{variant} optimizer repeats parameter name {name}")
                name_roles[str(name)] = role

        if variant in ("input_linear", "input_bilinear"):
            for name in _INPUT_MATRIX_NAMES:
                if name_roles.get(name) != "adamw_matrix":
                    raise AssertionError(f"{variant} adapter matrix {name} lacks AdamW ownership")
        elif any(name in name_roles for name in _INPUT_MATRIX_NAMES):
            raise AssertionError(f"{variant} attention arm unexpectedly owns input-route matrices")
        reports[variant] = role_counts
        del optimizer, model
    return reports
