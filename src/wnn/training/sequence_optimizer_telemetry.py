from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from wnn.training.sequence_provenance import json_value


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Name every actual hybrid group instead of labelling all rates AdamW."""
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def optimizer_contract(
    optimizer: torch.optim.Optimizer, config: Mapping[str, object], total_updates: int
) -> dict[str, object]:
    """Bind actual parameter groups and the full update schedule into run identity."""
    groups: list[dict[str, object]] = []
    fields = (
        "group_name",
        "role",
        "param_names",
        "muon",
        "blocks",
        "lr",
        "initial_lr",
        "weight_decay",
        "betas",
        "eps",
        "ns_steps",
        "rms",
        "ns_eps",
        "strict_fp32",
    )
    for index, group in enumerate(optimizer.param_groups):
        record = {field: json_value(group[field]) for field in fields if field in group}
        record["group_name"] = str(group.get("group_name", f"group_{index}"))
        record["parameter_count"] = sum(parameter.numel() for parameter in group["params"])
        record["matrix_update_scale"] = (
            "0.2 * sqrt(max(rows_per_block, columns))" if group.get("muon") else "AdamW"
        )
        groups.append(record)
    return {
        "class": type(optimizer).__name__,
        "groups": groups,
        "total_updates": total_updates,
        "warmup_updates": min(
            total_updates, math.ceil(total_updates * float(config["warmup_fraction"]))
        ),
        "schedule": str(config.get("schedule", "legacy_warmup_cosine")),
        "peak_lr": float(config["lr"]),
        "min_lr_ratio": float(config.get("min_lr_ratio", 0.0)),
    }
