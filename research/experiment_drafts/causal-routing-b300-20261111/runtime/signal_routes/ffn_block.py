"""Dense two-part feed-forward routing for the causal Transformer screen."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from wnn.training.sequence_qk_norm import headwise_rms_normalize

from wnn.training.sequence_attention import CausalTransformerBlock, _rotate

FFNVariant = Literal["ffn_static", "ffn_conditional"]
_BLOCK_MODULES = (
    "norm_attention",
    "qkv",
    "attention_output",
    "norm_feedforward",
    "feedforward_in",
    "feedforward_out",
    "residual_dropout",
    "local_attention",
)


def router_weights(
    normalized_hidden: Tensor,
    routing_matrix: Tensor,
    routing_bias: Tensor,
    variant: FFNVariant,
) -> Tensor:
    """Return the two dense expert weights from static or token-local logits."""
    if normalized_hidden.ndim < 2:
        raise ValueError("router input must have shape [..., rows, hidden]")
    if routing_matrix.shape != (2, normalized_hidden.shape[-1]):
        raise ValueError("routing matrix must have shape [2, hidden]")
    if routing_bias.shape != (2,):
        raise ValueError("routing bias must have shape [2]")
    if variant not in ("ffn_static", "ffn_conditional"):
        raise ValueError("unsupported feed-forward routing variant")

    router_input = (
        normalized_hidden if variant == "ffn_conditional" else torch.zeros_like(normalized_hidden)
    )
    logits = F.linear(router_input, routing_matrix, routing_bias)
    return torch.softmax(logits.float(), dim=-1).to(dtype=normalized_hidden.dtype)


def route_feedforward_activations(activations: Tensor, weights: Tensor) -> Tensor:
    """Scale two equal FFN channel partitions for one canonical output projection.

    The returned linear input is algebraically equivalent to
    ``2 * (g1 * E1 + g2 * E2)`` before adding the original output bias. Uniform
    gates therefore produce the exact unmodified FFN input at initialization.
    """
    if activations.ndim < 2 or activations.shape[-1] % 2 != 0:
        raise ValueError("feed-forward activations must have an even final dimension")
    if weights.shape != (*activations.shape[:-1], 2):
        raise ValueError("expert weights must match the activation leading dimensions")
    gates = (2.0 * weights).to(dtype=activations.dtype)
    first, second = activations.chunk(2, dim=-1)
    return torch.cat((first * gates[..., :1], second * gates[..., 1:]), dim=-1)


class SignalRouteBlock(CausalTransformerBlock):
    """Canonical block with a dense two-way gate over its existing FFN channels."""

    def __init__(self, source: CausalTransformerBlock, variant: FFNVariant) -> None:
        """Reuse every canonical submodule and add a zero-initialized router."""
        if variant not in ("ffn_static", "ffn_conditional"):
            raise ValueError("unsupported feed-forward routing variant")
        if source.feedforward_in.out_features % 2 != 0:
            raise ValueError("feed-forward width must split into two equal partitions")
        nn.Module.__init__(self)
        self.hidden = source.hidden
        self.heads = source.heads
        self.head_dim = source.head_dim
        self.qk_norm = source.qk_norm
        for name in _BLOCK_MODULES:
            self.add_module(name, getattr(source, name))
        self.register_buffer("rotary_inverse_frequency", source.rotary_inverse_frequency)
        self.routing_matrix = nn.Parameter(torch.zeros(2, self.hidden, dtype=torch.float32))
        self.routing_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.signal_route_variant = variant

    def forward(
        self,
        values: Tensor,
        past_key: Tensor,
        past_value: Tensor,
        start_position: int,
        context_rows: int,
        dropout_p: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run canonical attention/cache logic and route only the FFN channel halves."""
        batch, rows, _ = values.shape
        query, current_key, current_value = self.qkv(self.norm_attention(values)).chunk(3, dim=-1)
        query = query.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        current_key = current_key.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        current_value = current_value.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            query = headwise_rms_normalize(query)
            current_key = headwise_rms_normalize(current_key)
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

        normalized = self.norm_feedforward(values)
        activations = F.gelu(self.feedforward_in(normalized))
        weights = router_weights(
            normalized,
            self.routing_matrix,
            self.routing_bias,
            self.signal_route_variant,
        )
        routed = route_feedforward_activations(activations, weights)
        values = values + self.residual_dropout(self.feedforward_out(routed))

        cache_rows = min(context_rows - 1, all_key.shape[-2])
        if cache_rows == 0:
            next_key = all_key[..., :0, :].clone()
            next_value = all_value[..., :0, :].clone()
        else:
            next_key = all_key[..., -cache_rows:, :].clone()
            next_value = all_value[..., -cache_rows:, :].clone()
        return values, next_key, next_value
