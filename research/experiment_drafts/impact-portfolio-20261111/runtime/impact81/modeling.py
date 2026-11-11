from __future__ import annotations

from typing import Any

import torch
from torch import nn

from impact81.depth_models import build_depth_model
from impact81.representations import GroupInputProjection, PLEInputProjection
from impact81.target_head import TargetPrivateHead
from wnn.training.sequence_models import build_sequence_model

# Configs contain canonical runtime settings plus validated experiment metadata.


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    """Compose only the declared PR81 residual module onto the unchanged Transformer."""
    extension = config["impact81"]
    if extension.get("schema") != "impact81.model.v1":
        raise ValueError("unsupported PR81 model reconstruction schema")
    variant = extension["cell"]["variant"]
    if variant not in (
        "plain",
        "ple",
        "groups",
        "target_adapters",
        "deep_supervision",
        "copied_depth",
        "tied_depth",
        "pairwise_ranking",
    ):
        raise ValueError("unsupported PR81 experiment variant")
    if config["model"]["architecture"] != "transformer":
        raise ValueError("PR81 cards01–06 require the Transformer factory")
    model = (
        build_depth_model(config["model"], variant, int(extension["cell"]["seed"]))
        if variant in ("deep_supervision", "copied_depth", "tied_depth")
        else build_sequence_model(config["model"])
    )
    artifact = extension["preprocessing"]
    mean = torch.tensor(artifact["mean"], dtype=torch.float32)
    std = torch.tensor(artifact["std"], dtype=torch.float32)
    if mean.shape != (112,) or std.shape != (112,) or not bool((std > 0).all()):
        raise ValueError("invalid serialized normalization dimensions/scales")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("nonfinite serialized normalization")
    # New-module initialization must not perturb the common model or update RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(extension["cell"]["seed"]) + 8100)
        if variant == "ple":
            model.input_projection = PLEInputProjection(
                model.input_projection,
                torch.tensor(artifact["knots"], dtype=torch.float32),
                artifact["counts"],
            )
        elif variant == "groups":
            model.input_projection = GroupInputProjection(model.input_projection)
        elif variant == "target_adapters":
            model.output_head = TargetPrivateHead(model.output_head)
    with torch.no_grad():
        model.mean.copy_(mean)
        model.std.copy_(std)
    return model.to(device=device, dtype=torch.float32)
