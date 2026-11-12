"""Split-Q/K local attention adapters for the causal routing screen."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from wnn.training.sequence_qk_norm import headwise_rms_normalize

from wnn.training.sequence_attention_backends import LocalCausalAttention

AttentionVariant = Literal["attention_additive", "attention_differential"]


def split_query_key(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split rotated per-head Q/K into two half-width maps and pad to base width.

    Scaling each half by ``2**0.25`` preserves the original SDPA scale after
    zero-padding: the product of Q and K scales is ``sqrt(2)``, changing
    ``1/sqrt(d)`` into ``1/sqrt(d/2)``.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, heads, rows, dim]")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key batch, head, and feature dimensions must match")
    head_width = query.shape[-1]
    if head_width < 4 or head_width % 4 != 0:
        raise ValueError("split Q/K attention requires a head width divisible by four")

    half = head_width // 2
    scale = 2.0**0.25
    query_low, query_high = query[..., :half], query[..., half:]
    key_low, key_high = key[..., :half], key[..., half:]
    query_first = torch.cat((query_low * scale, torch.zeros_like(query_high)), dim=-1)
    key_first = torch.cat((key_low * scale, torch.zeros_like(key_high)), dim=-1)
    query_second = torch.cat((torch.zeros_like(query_low), query_high * scale), dim=-1)
    key_second = torch.cat((torch.zeros_like(key_low), key_high * scale), dim=-1)
    return query_first, key_first, query_second, key_second


def combine_attention_maps(
    first_map: Tensor, second_map: Tensor, variant: AttentionVariant
) -> Tensor:
    """Combine split attention maps, then apply the shared RMS/scale contract."""
    if first_map.shape != second_map.shape:
        raise ValueError("split attention maps must have identical shapes")
    if variant == "attention_additive":
        mixture = first_map + 0.5 * second_map
    elif variant == "attention_differential":
        mixture = first_map - 0.5 * second_map
    else:
        raise ValueError("unsupported split-Q/K attention variant")
    return 0.5 * headwise_rms_normalize(mixture)


class SplitQKLocalAttention(nn.Module):
    """Run two exact-window attention maps with the canonical shared V cache."""

    def __init__(self, base: LocalCausalAttention, variant: AttentionVariant) -> None:
        """Wrap the pinned local-attention backend without adding parameters."""
        super().__init__()
        if variant not in ("attention_additive", "attention_differential"):
            raise ValueError("unsupported split-Q/K attention variant")
        self.base = base
        self.variant = variant

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
        """Return the selected split-map combination over the unchanged V cache."""
        if key.shape != value.shape:
            raise ValueError("split-Q/K attention requires the canonical shared V shape")
        query_first, key_first, query_second, key_second = split_query_key(query, key)
        first_map = self.base(
            query_first,
            key_first,
            value,
            query_start,
            key_start,
            context_rows,
            dropout_p,
        )
        second_map = self.base(
            query_second,
            key_second,
            value,
            query_start,
            key_start,
            context_rows,
            dropout_p,
        )
        return combine_attention_maps(first_map, second_map, self.variant)
