from __future__ import annotations

from typing import Any

import torch

from wnn.training.sequence_data import SequenceData
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_models import build_sequence_model


def prepare_fit_model(
    config: dict[str, Any], data: SequenceData, context: DistributedContext
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Build strict-FP32 weights and install fitting-only normalization buffers."""
    model = build_sequence_model(config["model"])
    if not isinstance(model, torch.nn.Module):
        raise TypeError("sequence model builder must return a torch.nn.Module")
    model = model.to(device=context.device, dtype=torch.float32)
    mean, std = data.normalization()
    mean = mean.to(device=context.device, dtype=torch.float32).reshape(-1)
    std = std.to(device=context.device, dtype=torch.float32).reshape(-1)
    valid_normalization = (
        mean.shape == (112,)
        and std.shape == (112,)
        and bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(std).all())
        and bool((std > 0).all())
    )
    if not valid_normalization:
        raise ValueError("normalization must contain 112 finite means and positive FP32 scales")
    model_mean = getattr(model, "mean", None)
    model_std = getattr(model, "std", None)
    if not isinstance(model_mean, torch.Tensor) or not isinstance(model_std, torch.Tensor):
        raise TypeError("sequence model must expose mean and std normalization buffers")
    with torch.no_grad():
        model_mean.copy_(mean)
        model_std.copy_(std)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("all trainable and frozen model parameters must use strict FP32")
    if any(
        buffer.is_floating_point() and buffer.dtype != torch.float32 for buffer in model.buffers()
    ):
        raise TypeError("floating sequence-model buffers must use strict FP32")
    normalization = {"mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist()}
    return model, normalization
