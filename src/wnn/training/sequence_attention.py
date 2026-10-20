"""Bounded causal self-attention primitives for the B300 sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ATTENTION_QUERY_ROWS = 256


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
    paired = values.reshape(*values.shape[:-1], -1, 2)
    even = paired[..., 0]
    odd = paired[..., 1]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def _bounded_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float,
) -> Tensor:
    """Attend in fixed query blocks with an independent row-distance causal mask."""
    query_rows = query.shape[-2]
    key_rows = key.shape[-2]
    outputs: list[Tensor] = []
    for query_offset in range(0, query_rows, ATTENTION_QUERY_ROWS):
        query_end = min(query_offset + ATTENTION_QUERY_ROWS, query_rows)
        absolute_start = query_start + query_offset
        absolute_end = query_start + query_end
        first_key = max(0, absolute_start - (context_rows - 1) - key_start)
        end_key = min(key_rows, absolute_end - key_start)
        key_offset = torch.arange(first_key, end_key, device=query.device)
        query_positions = torch.arange(absolute_start, absolute_end, device=query.device)
        key_positions = key_offset + key_start
        distances = query_positions[:, None] - key_positions[None, :]
        allowed = (distances >= 0) & (distances < context_rows)
        with sdpa_kernel(SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(
                query[..., query_offset:query_end, :],
                key[..., first_key:end_key, :],
                value[..., first_key:end_key, :],
                attn_mask=allowed[None, None, :, :],
                dropout_p=dropout_p,
                is_causal=False,
            )
        outputs.append(attended)
    return torch.cat(outputs, dim=-2)


class CausalTransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE and a bounded rolling KV cache."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        ff_width: int,
        dropout: float,
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
        all_key = torch.cat((past_key, current_key), dim=-2)
        all_value = torch.cat((past_value, current_value), dim=-2)
        key_start = start_position - past_key.shape[-2]
        attended = _bounded_attention(
            query, all_key, all_value, start_position, key_start, context_rows, dropout_p
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
