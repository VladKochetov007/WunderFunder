"""Construct sequence optimizers while preserving opt-in parameter grouping."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from wnn.training.sequence_fit_utils import sequence_lr_multiplier
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer
from wnn.training.sequence_optimizer_controls import (
    LR_CONTROL_KEYS,
    apply_optimizer_lr_factors,
)


def make_optimizer(
    model: torch.nn.Module, config: Mapping[str, object], total_updates: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create the configured sequence optimizer and update-indexed LR schedule."""
    optimizer_name = config.get("optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = _make_adamw_optimizer(model, config)
    elif optimizer_name == "muon":
        optimizer = build_sequence_muon_optimizer(model, config)
    else:
        raise ValueError("optimizer must be adamw or muon")
    multiplier = sequence_lr_multiplier(
        str(config.get("schedule", "legacy_warmup_cosine")),
        total_updates,
        float(config.get("warmup_fraction", 0.05)),
        float(config.get("min_lr_ratio", 0.0)),
    )
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _make_adamw_optimizer(
    model: torch.nn.Module, config: Mapping[str, object]
) -> torch.optim.Optimizer:
    """Retain the historical flat AdamW group unless explicit controls are present."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("sequence model has no trainable parameters")
    learning_rate = float(config["lr"])
    betas = (
        float(config.get("adamw_beta1", 0.9)),
        float(config.get("adamw_beta2", 0.999)),
    )
    optimizer_settings = {
        "lr": learning_rate,
        "betas": betas,
        "eps": float(config.get("adamw_eps", 1e-8)),
        "weight_decay": float(config["weight_decay"]),
    }
    if not LR_CONTROL_KEYS.intersection(config):
        return torch.optim.AdamW(parameters, **optimizer_settings)
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    groups: list[dict[str, object]] = [
        {
            "params": [parameter for _, parameter in named_parameters],
            "param_names": [name for name, _ in named_parameters],
            "group_name": "adamw",
            "role": "adamw",
            "optimizer_role": "adamw",
            "lr": learning_rate,
            "initial_lr": learning_rate,
        }
    ]
    grouped = apply_optimizer_lr_factors(groups, model, config)
    return torch.optim.AdamW(grouped, **optimizer_settings)
