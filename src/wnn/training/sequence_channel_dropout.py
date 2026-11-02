"""Training-only dropout of whole raw-input channels within a sequence."""

from __future__ import annotations

import torch
from torch import Tensor


def mask_sequence_channels(
    inputs: Tensor, mean: Tensor, probability: float, seed: int, epoch: int, batch: int
) -> Tensor:
    """Replace independently masked channels by fitting-only model means.

    One mask is held over the entire sequence. A private per-batch generator
    leaves model and data-loader RNG untouched, making replay reproducible for
    the same order, batch size, epoch and source identity.
    """
    if inputs.ndim != 3 or inputs.shape[-1] < 112 or mean.numel() != inputs.shape[-1]:
        raise ValueError("channel dropout requires 112 raw channels and matching fitting means")
    if inputs.dtype != torch.float32 or mean.dtype != torch.float32 or inputs.device != mean.device:
        raise ValueError("channel dropout requires FP32 inputs/means on the same device")
    if not 0 < probability < 1 or min(seed, epoch, batch) < 0:
        raise ValueError("invalid channel-dropout seed or probability")
    stream_seed = (seed + epoch * 1_000_003 + batch * 10_007) % (2**63 - 1)
    generator = torch.Generator(device=inputs.device).manual_seed(stream_seed)
    keep = (
        torch.rand(inputs.shape[0], 1, 112, device=inputs.device, generator=generator)
        >= probability
    )
    result = inputs.clone()
    result[..., :112] = torch.where(keep, inputs[..., :112], mean[:112].reshape(1, 1, -1))
    return result
