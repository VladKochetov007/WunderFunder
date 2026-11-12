"""Small shared hint and streaming-state helpers for sequence adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_input_stem import RAW_INPUT_WIDTH
from wnn.training.sequence_precision import validate_forward_state


@dataclass(frozen=True)
class GRUSequenceState:
    """Recurrent state plus bounded normalized raw-input history for one sequence."""

    hidden: Tensor
    input_history: Tensor
    next_position: int


SequenceState = Tensor | TransformerState | GRUSequenceState | None


def check_sequence_input(model: nn.Module, values: Tensor, precision_mode: str) -> None:
    """Validate raw-row shape, FP32 master tensors, and selected forward scope."""
    if values.ndim != 3 or values.shape[-1] != RAW_INPUT_WIDTH:
        raise ValueError("sequence models accept only raw input shaped [batch, rows, 112]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("sequence batches and row chunks must be non-empty")
    if values.dtype != torch.float32:
        raise TypeError("sequence models require float32 inputs")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("sequence models require float32 parameters")
    if any(buffer is not None and buffer.dtype != torch.float32 for buffer in model.buffers()):
        raise TypeError("sequence models require float32 buffers")
    validate_forward_state(precision_mode, values.device.type)


class HintProjector(nn.Module):
    """Produce learned hints or a fixed, normalized random representation."""

    def __init__(self, hidden: int, hint_dim: int, mode: str) -> None:
        super().__init__()
        self.hint_dim = hint_dim
        self.linear: nn.Linear | None = None
        self.normalizer: nn.LayerNorm | None = None
        self.register_buffer("fixed_projection", None)
        if hint_dim == 0:
            return
        if mode == "learned":
            self.linear = nn.Linear(hidden, hint_dim)
            return
        self.normalizer = nn.LayerNorm(hidden, elementwise_affine=False)
        generator = torch.Generator(device="cpu").manual_seed(1729)
        projection = torch.randn(hint_dim, hidden, generator=generator, dtype=torch.float32)
        self.fixed_projection = projection / math.sqrt(hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        """Project hidden row vectors when hints are enabled."""
        if self.hint_dim == 0:
            return hidden.new_empty((*hidden.shape[:-1], 0))
        if self.linear is not None:
            return self.linear(hidden)
        if self.normalizer is None or self.fixed_projection is None:
            raise RuntimeError("fixed hint projection was not initialized")
        return F.linear(self.normalizer(hidden), self.fixed_projection)


def detach_state(state: SequenceState) -> SequenceState:
    """Detach recurrent or Transformer state at an explicit TBPTT boundary."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, GRUSequenceState):
        return GRUSequenceState(
            state.hidden.detach(), state.input_history.detach(), state.next_position
        )
    if isinstance(state, TransformerState):
        return TransformerState(
            tuple(key.detach() for key in state.keys),
            tuple(value.detach() for value in state.values),
            state.next_position,
        )
    raise TypeError(f"unsupported sequence state: {type(state).__name__}")
