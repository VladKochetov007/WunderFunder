"""Parameter-free Q/K normalization for the optional Transformer variant."""

from __future__ import annotations

import torch
from torch import Tensor

QK_NORM_EPSILON = 1e-6


def canonical_qk_norm(value: object) -> str | None:
    """Validate the public setting and erase the legacy-equivalent ``none`` value."""
    if not isinstance(value, str):
        raise TypeError("model.qk_norm must be a string")
    if value == "none":
        return None
    if value == "rms":
        return value
    raise ValueError("model.qk_norm must be 'none' or 'rms'")


def headwise_rms_normalize(values: Tensor) -> Tensor:
    """Apply affine-free RMS normalization over the final head dimension in FP32."""
    fp32_values = values.float()
    normalized = fp32_values * torch.rsqrt(
        fp32_values.square().mean(dim=-1, keepdim=True) + QK_NORM_EPSILON
    )
    return normalized.to(dtype=values.dtype)
