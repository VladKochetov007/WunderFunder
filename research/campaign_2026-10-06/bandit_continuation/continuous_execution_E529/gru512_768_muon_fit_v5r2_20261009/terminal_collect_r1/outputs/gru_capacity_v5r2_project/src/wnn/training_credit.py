from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch


class _CreditContext(Protocol):
    factor: float


class _ScaleCredit(torch.autograd.Function):
    """Identity forward; scale only recurrent credit arriving from later chunks."""

    @staticmethod
    def forward(ctx: _CreditContext, state: torch.Tensor, factor: float) -> torch.Tensor:
        ctx.factor = factor
        return state

    @staticmethod
    def backward(ctx: _CreditContext, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.factor, None


def scale_hidden_credit(
    state: torch.Tensor | tuple[torch.Tensor, ...],
    factor: float,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Leave state values intact and apply a detached backward-only edge multiplier."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("surviving credit factors must be finite and positive")
    if isinstance(state, tuple):
        return tuple(_ScaleCredit.apply(value, factor) for value in state)
    return _ScaleCredit.apply(state, factor)


def credit_edge_scales(
    edges: int,
    keep_probability: float,
    seed: int,
    step: int,
    compensate: bool,
) -> tuple[float, ...]:
    """Draw independent positive-support gates using a stateless private generator.

    Compensation targets the fixed update-window graph conditional on its incoming
    detached state. No full-sequence or nonlinear-optimizer unbiasedness is claimed.
    The longest possible retained graph is that declared finite window.
    """
    if edges < 0 or not 0 < keep_probability <= 1 or seed < 0 or step < 0:
        raise ValueError(
            "credit sampling requires nonnegative indices and keep probability in (0,1]"
        )
    if not math.isfinite(1 / keep_probability):
        raise ValueError("inverse survival probability must be finite")
    if keep_probability == 1:
        return (1.0,) * edges
    generator = np.random.default_rng(np.random.SeedSequence([seed, step]))
    keep = generator.random(edges) < keep_probability
    scale = 1 / keep_probability if compensate else 1.0
    return tuple(scale if alive else 0.0 for alive in keep)
