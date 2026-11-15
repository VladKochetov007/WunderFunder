"""Matched angular/radial training perturbations of post-RoPE attention queries."""

from __future__ import annotations

import math

import torch
from torch import Tensor

ANGLE_RADIANS = 0.35
COSINE = math.cos(ANGLE_RADIANS)
SINE = math.sin(ANGLE_RADIANS)


def validate_query_perturbation(value: object) -> str:
    """Accept only the three fixed modes declared in the experiment panel."""
    if not isinstance(value, str) or value not in ("none", "angular", "radial"):
        raise ValueError("query_perturbation must be 'none', 'angular', or 'radial'")
    return value


def perturb_query(query: Tensor, mode: str, *, signs: Tensor | None = None) -> Tensor:
    """Apply one matched query perturbation without changing keys or values.

    Args:
        query: Post-RoPE float32/BF16 queries shaped [batch, heads, rows, head_dim].
        mode: The registered mode: none, angular, or radial.
        signs: Optional independently supplied Rademacher signs [B,H,1,D/2].
            Omission draws native device RNG; signs are shared over query rows.

    Returns:
        Query values in their original dtype and shape. None returns the same
        tensor and consumes no RNG. Active modes never mutate the input.

    The caller gates this function to training. For a rotary pair q=(a,b), let
    Jq=(-b,a), c=cos(.35), u=sin(.35), and s be uniform +/-1. Angular returns
    cq+suJq; radial returns (c+su)q. Both have conditional mean cq, expected
    squared norm ||q||^2, and expected perturbation energy 2(1-c)||q||^2.
    These are real-arithmetic identities, not a BF16 bit-parity assertion.
    """
    validate_query_perturbation(mode)
    if mode == "none":
        return query
    if query.ndim != 4 or min(query.shape) < 1 or query.shape[-1] % 2:
        raise ValueError("query must have nonempty [B,H,T,D] shape with even head width")
    if query.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("the registered query perturbation requires float32 or BF16")
    sign_shape = (query.shape[0], query.shape[1], 1, query.shape[-1] // 2)
    if signs is None:
        signs = torch.randint(0, 2, sign_shape, device=query.device).float().mul_(2).sub_(1)
    else:
        if signs.shape != sign_shape or signs.device != query.device:
            raise ValueError("explicit signs must have [B,H,1,D/2] shape on the query device")
        if not bool(((signs == -1) | (signs == 1)).all()):
            raise ValueError("explicit signs must contain only -1 and +1")
        signs = signs.float()
    pairs = query.float().reshape(*query.shape[:-1], -1, 2)
    signed_sine = signs.unsqueeze(-1) * SINE
    if mode == "angular":
        quarter_turn = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        perturbed = COSINE * pairs + signed_sine * quarter_turn
    else:
        perturbed = (COSINE + signed_sine) * pairs
    return perturbed.flatten(-2).to(dtype=query.dtype)
