"""Explicit attention backend dispatch for bounded causal Transformer chunks."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from wnn.training.sequence_attention_capabilities import (
    _efficient_mask_format,
    attention_backend_capabilities,
)
from wnn.training.sequence_attention_masks import (
    _additive_window_mask,
    _relative_window_mask,
    _trim_window_keys,
)

__all__ = ("LocalCausalAttention", "attention_backend_capabilities")

MATH_QUERY_ROWS = 256
FLEX_BLOCK_MASK_CACHE_SIZE = 128

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # FlexAttention is optional on older or stripped PyTorch builds.
    create_block_mask = None
    flex_attention = None


def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float,
    recency_bias: float = 0.0,
) -> Tensor:
    """Run fixed-query-chunk causal math SDPA, optionally biasing past keys."""
    query_rows = query.shape[-2]
    key_rows = key.shape[-2]
    outputs: list[Tensor] = []
    for query_offset in range(0, query_rows, MATH_QUERY_ROWS):
        query_end = min(query_offset + MATH_QUERY_ROWS, query_rows)
        absolute_start = query_start + query_offset
        absolute_end = query_start + query_end
        first_key = max(0, absolute_start - (context_rows - 1) - key_start)
        end_key = min(key_rows, absolute_end - key_start)
        key_offset = torch.arange(first_key, end_key, device=query.device)
        query_positions = torch.arange(absolute_start, absolute_end, device=query.device)
        key_positions = key_offset + key_start
        distances = query_positions[:, None] - key_positions[None, :]
        allowed = (distances >= 0) & (distances < context_rows)
        mask = allowed[None, None, :, :]
        if recency_bias:
            bias = -recency_bias * distances.to(query.dtype) / max(context_rows - 1, 1)
            mask = bias.masked_fill(~allowed, float("-inf"))[None, None, :, :]
        with sdpa_kernel(SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(
                query[..., query_offset:query_end, :],
                key[..., first_key:end_key, :],
                value[..., first_key:end_key, :],
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        outputs.append(attended)
    return torch.cat(outputs, dim=-2)


def _efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float,
) -> Tensor:
    """Force efficient SDPA with a probed boolean or additive exact rolling mask."""
    key, value, past_rows = _trim_window_keys(
        query, key, value, query_start, key_start, context_rows
    )
    if torch.compiler.is_compiling():
        # Keep OrderedDict capability-cache mutations out of Dynamo's trace.
        mask = _additive_window_mask(
            query.shape[-2],
            key.shape[-2],
            past_rows,
            context_rows,
            query.device,
            query.dtype,
        )
    else:
        _, mask = _efficient_mask_format(query, key, value, past_rows, context_rows, dropout_p)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
        )


@lru_cache(maxsize=FLEX_BLOCK_MASK_CACHE_SIZE)
def _cached_flex_block_mask(
    batch: int,
    heads: int,
    query_rows: int,
    key_rows: int,
    past_rows: int,
    context_rows: int,
    device_type: str,
    device_index: int | None,
) -> object:
    """Cache BlockMasks by bounded shape and past length, never absolute position."""
    if create_block_mask is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )

    def local_mask(_batch: Tensor, _head: Tensor, q_index: Tensor, kv_index: Tensor) -> Tensor:
        """Return one causal sliding-window element of the block mask."""
        return _relative_window_mask(q_index, kv_index, past_rows, context_rows)

    return create_block_mask(
        local_mask,
        batch,
        heads,
        query_rows,
        key_rows,
        device=device,
    )


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Callable[..., Tensor]:
    """Compile the installed FlexAttention callable once without max-autotune overhead."""
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
    return torch.compile(flex_attention, mode="default")


def _flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float,
) -> Tensor:
    """Run FlexAttention with a bounded relative-offset causal BlockMask."""
    if create_block_mask is None or flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
    if not query.is_cuda:
        raise RuntimeError("FlexAttention requires a CUDA tensor for this backend path")
    if dropout_p != 0.0:
        raise ValueError("FlexAttention backend requires attention dropout_p=0")
    key, value, past_rows = _trim_window_keys(
        query, key, value, query_start, key_start, context_rows
    )
    block_mask = _cached_flex_block_mask(
        query.shape[0],
        query.shape[1],
        query.shape[-2],
        key.shape[-2],
        past_rows,
        context_rows,
        query.device.type,
        query.device.index,
    )
    return _compiled_flex_attention()(
        query,
        key,
        value,
        block_mask=block_mask,
        kernel_options={"BACKEND": "TRITON"},
    )


class LocalCausalAttention(nn.Module):
    """Dispatch exact rolling causal attention to one explicitly selected backend."""

    def __init__(self, attention_backend: str = "math", recency_bias: float = 0.0) -> None:
        super().__init__()
        if attention_backend not in ("math", "efficient", "flex"):
            raise ValueError("attention_backend must be 'math', 'efficient', or 'flex'")
        if (
            not math.isfinite(recency_bias)
            or not 0 <= recency_bias <= 4
            or (recency_bias and attention_backend != "math")
        ):
            raise ValueError("recency bias must be in [0,4] and requires math attention")
        self.recency_bias = recency_bias
        self.attention_backend = attention_backend

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_start: int,
        key_start: int,
        context_rows: int,
        dropout_p: float = 0.0,
    ) -> Tensor:
        """Attend over current and cached keys using the selected exact mask."""
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape [batch, heads, rows, dim]")
        if (
            key.shape != value.shape
            or query.shape[:2] != key.shape[:2]
            or query.shape[-1] != key.shape[-1]
        ):
            raise ValueError("query, key, and value batch/head dimensions must match")
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise TypeError("query, key, and value must use the same dtype")
        if query.device != key.device or key.device != value.device:
            raise ValueError("query, key, and value must use the same device")
        if context_rows < 1 or query_start < key_start:
            raise ValueError(
                "context_rows must be positive and key_start cannot exceed query_start"
            )

        if self.attention_backend == "math":
            return _math_attention(
                query,
                key,
                value,
                query_start,
                key_start,
                context_rows,
                dropout_p,
                self.recency_bias,
            )
        if self.attention_backend == "efficient":
            if not query.is_cuda:
                raise RuntimeError("efficient attention requires CUDA; no fallback was attempted")
            return _efficient_attention(
                query, key, value, query_start, key_start, context_rows, dropout_p
            )
        return _flex_attention(query, key, value, query_start, key_start, context_rows, dropout_p)
