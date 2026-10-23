"""Bounded causal self-attention primitives for the B300 sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wnn.training.sequence_attention_backends import LocalCausalAttention


@dataclass(frozen=True)
class TransformerState:
    """Per-layer rotated key/value cache and the next absolute sequence row."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    next_position: int


def _rotate(values: Tensor, positions: Tensor, inverse_frequency: Tensor) -> Tensor:
    """Apply rotary position embeddings to a batch of head vectors."""
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]
    paired = values.float().reshape(*values.shape[:-1], -1, 2)
    even = paired[..., 0]
    odd = paired[..., 1]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(
        -2
    )
    return rotated.to(dtype=values.dtype)


class CausalTransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE and a bounded rolling KV cache."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        ff_width: int,
        dropout: float,
        attention_backend: str = "math",
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm_attention = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attention_output = nn.Linear(hidden, hidden)
        self.norm_feedforward = nn.LayerNorm(hidden)
        self.feedforward_in = nn.Linear(hidden, ff_width)
        self.feedforward_out = nn.Linear(ff_width, hidden)
        self.residual_dropout = nn.Dropout(dropout)
        self.local_attention = LocalCausalAttention(attention_backend)
        self.register_buffer(
            "rotary_inverse_frequency",
            1.0
            / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)),
        )

    def forward(
        self,
        values: Tensor,
        past_key: Tensor,
        past_value: Tensor,
        start_position: int,
        context_rows: int,
        dropout_p: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the block and return its output plus the bounded cache."""
        batch, rows, _ = values.shape
        query, current_key, current_value = self.qkv(self.norm_attention(values)).chunk(3, dim=-1)
        query = query.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        current_key = current_key.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        current_value = current_value.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(start_position, start_position + rows, device=values.device)
        query = _rotate(query, positions, self.rotary_inverse_frequency)
        current_key = _rotate(current_key, positions, self.rotary_inverse_frequency)
        all_key = torch.cat((past_key.float(), current_key.float()), dim=-2)
        all_value = torch.cat((past_value.float(), current_value.float()), dim=-2)
        key_start = start_position - past_key.shape[-2]
        attended = self.local_attention(
            query,
            all_key.to(dtype=query.dtype),
            all_value.to(dtype=query.dtype),
            start_position,
            key_start,
            context_rows,
            dropout_p,
        )
        attended = attended.transpose(1, 2).reshape(batch, rows, self.hidden)
        values = values + self.residual_dropout(self.attention_output(attended))
        feedforward = self.feedforward_out(
            F.gelu(self.feedforward_in(self.norm_feedforward(values)))
        )
        values = values + self.residual_dropout(feedforward)
        cache_rows = min(context_rows - 1, all_key.shape[-2])
        if cache_rows == 0:
            next_key = all_key[..., :0, :].clone()
            next_value = all_value[..., :0, :].clone()
        else:
            next_key = all_key[..., -cache_rows:, :].clone()
            next_value = all_value[..., -cache_rows:, :].clone()
        return values, next_key, next_value
