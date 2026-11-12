"""Server-only analytical checks for the paired split-Q/K attention variants."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from signal_routes.attention_block import (
    AttentionVariant,
    SplitQKLocalAttention,
    combine_attention_maps,
    split_query_key,
)
from signal_routes.receiver_support import synthetic_tensor
from wnn.training.sequence_attention_backends import LocalCausalAttention


def _direct_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
) -> Tensor:
    """Compute exact causal sliding-window attention in FP32 for a tiny fixture."""
    q, k, v = query.float(), key.float(), value.float()
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    query_positions = torch.arange(query_start, query_start + q.shape[-2], device=q.device)
    key_positions = torch.arange(key_start, key_start + k.shape[-2], device=q.device)
    distance = query_positions[:, None] - key_positions[None, :]
    allowed = (distance >= 0) & (distance < context_rows)
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _manual_mix(first: Tensor, second: Tensor, variant: AttentionVariant) -> Tensor:
    """Apply the declared sign, affine-free FP32 RMS normalization and fixed scale."""
    mixture = first + 0.5 * second if variant == "attention_additive" else first - 0.5 * second
    fp32 = mixture.float()
    return 0.5 * fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + 1e-6)


def _check_variant(query: Tensor, key: Tensor, value: Tensor, variant: AttentionVariant) -> float:
    """Compare the fused causal wrapper to explicit maps and signed arithmetic."""
    query_start, key_start, context_rows = 5, 2, 4
    q1, k1, q2, k2 = split_query_key(query, key)
    half = query.shape[-1] // 2
    scale = 2.0**0.25
    if not torch.allclose(q1[..., :half].float(), query[..., :half].float() * scale, atol=0.03):
        raise AssertionError("first split query does not preserve half-width SDPA scale")
    if not torch.allclose(k1[..., :half].float(), key[..., :half].float() * scale, atol=0.03):
        raise AssertionError("first split key does not preserve half-width SDPA scale")
    if not torch.allclose(q2[..., half:].float(), query[..., half:].float() * scale, atol=0.03):
        raise AssertionError("second split query does not preserve half-width SDPA scale")
    if not torch.allclose(k2[..., half:].float(), key[..., half:].float() * scale, atol=0.03):
        raise AssertionError("second split key does not preserve half-width SDPA scale")
    if bool(q1[..., half:].count_nonzero()) or bool(k1[..., half:].count_nonzero()):
        raise AssertionError("first split attention map must zero-pad the unused channels")
    if bool(q2[..., :half].count_nonzero()) or bool(k2[..., :half].count_nonzero()):
        raise AssertionError("second split attention map must zero-pad the unused channels")

    first = _direct_attention(
        query[..., :half], key[..., :half], value, query_start, key_start, context_rows
    )
    second = _direct_attention(
        query[..., half:], key[..., half:], value, query_start, key_start, context_rows
    )
    expected = _manual_mix(first, second, variant)
    helper = combine_attention_maps(first, second, variant)
    if not torch.allclose(helper.float(), expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{variant} map helper differs from its signed formula")
    actual = SplitQKLocalAttention(LocalCausalAttention("efficient"), variant)(
        query, key, value, query_start, key_start, context_rows, 0.0
    )
    gap = float((actual.float() - expected).abs().max().item())
    if gap > 2e-2:
        raise AssertionError(
            f"{variant} fused causal attention differs from direct reference: {gap}"
        )
    return gap


def check_attention_maps(device: torch.device, seed: int) -> dict[str, float | bool]:
    """Check both signs, shared values, half-width scaling and exact causal masks."""
    query = synthetic_tensor((1, 2, 3, 16), device, seed).to(torch.bfloat16)
    key = synthetic_tensor((1, 2, 6, 16), device, seed + 1).to(torch.bfloat16)
    value = synthetic_tensor((1, 2, 6, 16), device, seed + 2).to(torch.bfloat16)
    additive_gap = _check_variant(query, key, value, "attention_additive")
    differential_gap = _check_variant(query, key, value, "attention_differential")
    half = query.shape[-1] // 2
    direct_first = _direct_attention(query[..., :half], key[..., :half], value, 5, 2, 4)
    direct_second = _direct_attention(query[..., half:], key[..., half:], value, 5, 2, 4)
    if not bool(torch.isfinite(direct_first).all() and torch.isfinite(direct_second).all()):
        raise AssertionError("causal reference contains nonfinite outputs")
    return {
        "additive_max_abs_vs_direct": additive_gap,
        "differential_max_abs_vs_direct": differential_gap,
        "shared_value_and_causal_window": True,
        "half_width_scale_verified": True,
    }
