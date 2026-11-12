"""Server-only construction gate for canonical Muon/AdamW parameter ownership."""

from __future__ import annotations

from typing import Any

import torch

from signal_routes.receiver_support import build_probe_model


def check_optimizer_ownership(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, dict[str, int]]:
    """Construct the real small-model optimizer for every variant and audit ownership."""
    from wnn.training.sequence_optimizer import build_sequence_muon_optimizer

    reports: dict[str, dict[str, int]] = {}
    for variant in (
        "plain",
        "attention_additive",
        "attention_differential",
        "ffn_static",
        "ffn_conditional",
    ):
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
            name_roles.update((str(name), role) for name in names)
        if variant.startswith("ffn_"):
            for layer in range(2):
                if name_roles.get(f"blocks.{layer}.routing_matrix") != "adamw_matrix":
                    raise AssertionError(f"{variant} router matrix lacks its declared AdamW role")
                if name_roles.get(f"blocks.{layer}.routing_bias") != "adamw_no_decay":
                    raise AssertionError(f"{variant} router bias lacks its declared no-decay role")
        reports[variant] = role_counts
        del optimizer, model
    return reports
