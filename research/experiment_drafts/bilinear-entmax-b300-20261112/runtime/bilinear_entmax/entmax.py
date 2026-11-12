"""Finite, mask-aware exact 1.5-entmax normalizer for attention logits.

The threshold equations follow the sort-based algorithm in
``deep-spin/entmax/entmax/activations.py`` (MIT license); this implementation
keeps masked entries out of threshold statistics and uses ordinary PyTorch ops
so gradients remain differentiable away from support-boundary ties.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _entmax15_forward(logits: Tensor, mask: Tensor) -> Tensor:
    """Apply sort-based threshold equations to FP32 logits on their last axis."""
    has_allowed = mask.any(dim=-1, keepdim=True)
    safe_values = torch.where(mask, logits, torch.zeros_like(logits))
    finite_min = torch.finfo(logits.dtype).min
    masked_max = safe_values.masked_fill(~mask, finite_min).amax(dim=-1, keepdim=True)
    row_max = torch.where(has_allowed, masked_max, torch.zeros_like(masked_max))

    # Entmax is translation invariant. Centering and clipping below -2 logits
    # is exact because alpha=1.5 support cannot extend more than 2 below a max.
    centered = torch.where(mask, safe_values - row_max, torch.zeros_like(safe_values))
    scaled = centered.clamp(min=-2.0, max=0.0) * 0.5

    # Valid scaled scores are in [-1, 0]. The finite -2 sentinel sorts after
    # every valid score and contributes zero to both cumulative moments.
    sorted_input = torch.where(mask, scaled, torch.full_like(scaled, -2.0))
    sorted_values, _ = torch.sort(sorted_input, dim=-1, descending=True)
    valid_count = mask.sum(dim=-1, keepdim=True)
    ranks = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype)
    in_valid_prefix = ranks <= valid_count
    moment_values = torch.where(in_valid_prefix, sorted_values, torch.zeros_like(sorted_values))
    cumulative = moment_values.cumsum(dim=-1)
    cumulative_square = moment_values.square().cumsum(dim=-1)
    mean = cumulative / ranks
    mean_square = cumulative_square / ranks
    ss = ranks * (mean_square - mean.square())
    delta = ((1.0 - ss) / ranks).clamp_min(0.0)

    # Avoid sqrt's singular derivative at zero; the custom backward below uses
    # the closed-form entmax Jacobian on the active support instead.
    positive_delta = delta > 0.0
    safe_delta = torch.where(positive_delta, delta, torch.ones_like(delta))
    root_delta = torch.where(positive_delta, safe_delta.sqrt(), torch.zeros_like(delta))
    thresholds = mean - root_delta
    support = in_valid_prefix & (thresholds <= sorted_values)
    support_count = support.sum(dim=-1, keepdim=True)
    support_index = (support_count - 1).clamp_min(0)
    threshold = thresholds.gather(dim=-1, index=support_index)
    threshold = torch.where(support_count > 0, threshold, torch.zeros_like(threshold))

    probabilities = (scaled - threshold).clamp_min(0.0).square()
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


class _Entmax15(torch.autograd.Function):
    """Use the analytic 1.5-entmax Jacobian on the forward support."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, logits: Tensor, mask: Tensor) -> Tensor:
        """Save the active probabilities for the backward pass."""
        probabilities = _entmax15_forward(logits, mask)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, gradient: Tensor) -> tuple[Tensor, None]:
        """Multiply the upstream gradient by the active-support Jacobian."""
        (probabilities,) = ctx.saved_tensors
        active_slope = probabilities.sqrt()
        denominator = active_slope.sum(dim=-1, keepdim=True)
        centered_gradient = (gradient * active_slope).sum(dim=-1, keepdim=True)
        shift = centered_gradient / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
        input_gradient = active_slope * (gradient - shift)
        input_gradient = torch.where(
            denominator > 0.0, input_gradient, torch.zeros_like(input_gradient)
        )
        return input_gradient, None


def entmax15(logits: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    """Normalize logits with alpha=1.5 entmax, returning exact zeros off-mask.

    Args:
        logits: Floating point scores, such as ``[batch, heads, query, key]``.
        allowed: Boolean mask broadcastable to ``logits``. All-masked rows are
            defined to return zeros; normal causal attention rows include self.
        dim: Key dimension to normalize.

    Returns:
        FP32 probabilities with unit mass on every nonempty row and exact zeros
        in masked slots. Rows with no allowed slots return all zeros.

    Raises:
        TypeError: If logits are not floating point or the mask is not boolean.
        ValueError: If the dimension or mask shape is invalid.
    """
    if logits.ndim == 0 or not logits.is_floating_point():
        raise TypeError("entmax15 logits must be a non-scalar floating-point tensor")
    if allowed.dtype != torch.bool:
        raise TypeError("entmax15 allowed mask must be boolean")
    if allowed.device != logits.device:
        raise ValueError("entmax15 logits and mask must be on the same device")
    if type(dim) is not int or not -logits.ndim <= dim < logits.ndim:
        raise ValueError("entmax15 dim is outside the logits rank")
    axis = dim % logits.ndim
    if logits.shape[axis] == 0:
        raise ValueError("entmax15 cannot normalize an empty dimension")
    try:
        mask = allowed.expand_as(logits)
    except RuntimeError as error:
        raise ValueError("entmax15 mask must be broadcastable to logits") from error

    values = logits.float().movedim(axis, -1)
    expanded_mask = mask.movedim(axis, -1)
    probabilities = _Entmax15.apply(values, expanded_mask)
    return probabilities.movedim(-1, axis)
