"""Compact GRU and parallel causal local-attention sequence regressor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_gru_adapter import GRUSequenceAdapter
from wnn.training.sequence_model_helpers import HintProjector, detach_state


@dataclass(frozen=True)
class GRULocalState:
    """Independent recurrent and local-attention caches for one complete sequence."""

    recurrent: Tensor
    attention: TransformerState

    def detach(self) -> GRULocalState:
        """Cut both carried graphs at a TBPTT boundary."""
        attention = detach_state(self.attention)
        if not isinstance(attention, TransformerState):
            raise TypeError("local-attention state detachment returned an invalid type")
        return GRULocalState(self.recurrent.detach(), attention)


class GRULocalSequenceAdapter(nn.Module):
    """Canonical GRU trunk with two local-attention blocks and gated residual fusion.

    Attention has its own raw-input stem and pre-norm RoPE blocks. Both branches
    carry state across chunks; the two-target direct linear path uses only the
    current standardized input. This is a new adaptation, not a GALAM replica.
    """

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        # Import lazily to keep the public factory's dependencies acyclic.
        from wnn.training.sequence_models import _TransformerSequenceAdapter

        hidden = int(config["hidden"])
        attention_hidden = int(config["local_hidden"])
        self.register_buffer("mean", torch.zeros(112))
        self.register_buffer("std", torch.ones(112))
        self.recurrent = GRUSequenceAdapter(
            hidden,
            int(config["layers"]),
            0.0,
            0,
            "learned",
            "fp32",
        )
        # The canonical GRU's public head is replaced by the declared fused head.
        self.recurrent.head = nn.Identity()
        self.attention = _TransformerSequenceAdapter(
            attention_hidden,
            int(config["local_layers"]),
            int(config["heads"]),
            attention_hidden * int(config["ff_multiplier"]),
            int(config["context_rows"]),
            0.0,
            bool(config["checkpoint_blocks"]),
            attention_hidden,
            "learned",
            "fp32",
            "math",
            "raw112",
        )
        self.attention.output_head = nn.Identity()
        self.attention.hint_projector = HintProjector(attention_hidden, 0, "learned")
        self.local_projection = nn.Linear(attention_hidden, hidden)
        self.fusion_gate = nn.Linear(hidden + attention_hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 2)
        self.direct_head = nn.Linear(112, 2)
        self.hint_projector = HintProjector(hidden, int(config["hint_dim"]), "learned")

    def forward(
        self, values: Tensor, state: GRULocalState | None = None
    ) -> tuple[Tensor, Tensor, GRULocalState]:
        """Predict all rows from current/past inputs and return both branch caches."""
        if state is not None and not isinstance(state, GRULocalState):
            raise TypeError("GRU/local-attention requires GRULocalState")
        # Normalize once before the independent branches. Their identity buffers
        # remain independent across .to(device); cross-module buffer aliases do not.
        normalized = (values - self.mean) / self.std
        recurrent, _, next_recurrent = self.recurrent(
            normalized, None if state is None else state.recurrent
        )
        # Identity canonical head concatenates recurrent and current encodings.
        recurrent = recurrent[..., : self.recurrent.hidden]
        local, _, next_attention = self.attention(
            normalized, None if state is None else state.attention
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat((recurrent, local), dim=-1)))
        hidden = self.output_norm(recurrent + gate * self.local_projection(local))
        prediction = self.output_head(hidden) + self.direct_head(normalized)
        return (
            prediction,
            self.hint_projector(hidden),
            GRULocalState(next_recurrent, next_attention),
        )
