"""Explicit, matched softmax and 1.5-entmax local causal attention maps."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bilinear_entmax.entmax import entmax15

AttentionNormalizer = Literal["softmax", "entmax"]


class ExplicitLocalAttention(nn.Module):
    """Apply one normalizer to the same FP32 QK and dtype-matched probability-V path."""

    def __init__(self, normalizer: AttentionNormalizer) -> None:
        """Select the paired attention normalizer without adding parameters."""
        super().__init__()
        if normalizer not in ("softmax", "entmax"):
            raise ValueError("normalizer must be 'softmax' or 'entmax'")
        self.normalizer = normalizer

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
        """Attend over the exact global-offset causal window and return V dtype."""
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape [batch, heads, rows, dim]")
        if (
            key.shape != value.shape
            or query.shape[:2] != key.shape[:2]
            or query.shape[-1] != key.shape[-1]
        ):
            raise ValueError("query, key, and value dimensions do not match")
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise TypeError("query, key, and value must use the same dtype")
        if query.device != key.device or key.device != value.device:
            raise ValueError("query, key, and value must use the same device")
        if (
            type(query_start) is not int
            or type(key_start) is not int
            or type(context_rows) is not int
            or context_rows < 1
            or key_start > query_start
        ):
            raise ValueError("causal attention offsets and context are invalid")
        if (
            query.shape[-2] == 0
            or key.shape[-2] == 0
            or key_start + key.shape[-2] < query_start + query.shape[-2]
        ):
            raise ValueError("every causal query row must include its current key")
        if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
            raise ValueError("attention dropout must be finite and in [0, 1)")

        query_positions = torch.arange(
            query_start, query_start + query.shape[-2], device=query.device
        )
        key_positions = torch.arange(key_start, key_start + key.shape[-2], device=query.device)
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < context_rows)

        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
            scores = scores / math.sqrt(query.shape[-1])
            broadcast_mask = allowed[None, None, :, :]
            if self.normalizer == "softmax":
                probabilities = torch.softmax(
                    scores.masked_fill(~broadcast_mask, float("-inf")), dim=-1
                )
            else:
                probabilities = entmax15(scores, broadcast_mask)
        if dropout_p > 0.0:
            probabilities = F.dropout(probabilities, p=dropout_p, training=True)
        return torch.matmul(probabilities.to(dtype=value.dtype), value)
