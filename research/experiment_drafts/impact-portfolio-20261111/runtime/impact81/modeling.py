from __future__ import annotations

from typing import Any

import torch
from torch import nn

from impact81.representations import GroupInputProjection, PLEInputProjection
from impact81.target_head import TargetPrivateHead
from wnn.training.sequence_models import build_sequence_model

# Configs contain canonical runtime settings plus validated experiment metadata.


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    """Compose the declared mechanism using its family's pinned model factory."""
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
        "future_heads",
        "gru_plain",
        "perturbation_control",
        "state_perturbation",
        "nll_transfer",
        "clip_event",
    ):
        raise ValueError("unsupported PR81 experiment variant")
    if config["model"]["architecture"] == "gru":
        from impact81.gru_models import build_gru_model

        model = build_gru_model(config["model"], variant, int(extension["cell"]["seed"]))
    elif variant == "future_heads":
        from impact81.future_model import build_future_model

        model = build_future_model(config["model"], int(extension["cell"]["seed"]))
    elif variant in ("deep_supervision", "copied_depth", "tied_depth"):
        from impact81.depth_models import build_depth_model

        model = build_depth_model(config["model"], variant, int(extension["cell"]["seed"]))
    else:
        if config["model"]["architecture"] != "transformer":
            raise ValueError("unsupported model family for PR81")
        model = build_sequence_model(config["model"])
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
