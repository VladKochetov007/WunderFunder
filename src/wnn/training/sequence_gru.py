"""Canonical GRU adapter with optional causal per-channel raw-input filtering."""

from __future__ import annotations

import torch
from torch import Tensor

from wnn.nn import RNNModel
from wnn.training.sequence_input_stem import (
    CausalInputStem,
    validate_input_history,
    validate_input_stem_kernel,
)
from wnn.training.sequence_model_helpers import (
    GRUSequenceState,
    HintProjector,
    check_sequence_input,
)


class GRUSequenceAdapter(RNNModel):
    """Expose the canonical GRU output, hints, and detached-streaming state."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        dropout: float,
        hint_dim: int,
        hint_mode: str,
        precision_mode: str,
        input_stem_kernel: int = 0,
    ) -> None:
        super().__init__(112, hidden, layers, dropout, n_out=2, cell="gru", enc="linear")
        self.precision_mode = precision_mode
        self.input_stem_kernel = validate_input_stem_kernel(input_stem_kernel)
        self.input_stem = CausalInputStem(input_stem_kernel) if input_stem_kernel else None
        self.hint_projector = HintProjector(hidden, hint_dim, hint_mode)

    def _hidden_and_history(
        self, values: Tensor, state: Tensor | GRUSequenceState | None
    ) -> tuple[Tensor | None, Tensor | None, int]:
        if self.input_stem_kernel > 1:
            if state is None:
                return None, None, 0
            if not isinstance(state, GRUSequenceState):
                raise TypeError(
                    "stateful GRU stem state must include hidden state and input history"
                )
            position = state.next_position
            history = validate_input_history(
                state.input_history, values, position, self.input_stem_kernel
            )
            hidden = state.hidden
            if not isinstance(hidden, Tensor):
                raise TypeError("stateful GRU hidden state must be a tensor")
        else:
            if isinstance(state, GRUSequenceState):
                raise TypeError(
                    "GRU input-history state requires input_stem_kernel larger than one"
                )
            if state is not None and not isinstance(state, Tensor):
                raise TypeError("GRU state must be a tensor")
            hidden, history, position = state, None, 0
        if hidden is not None and (
            hidden.shape != (self.gru.num_layers, values.shape[0], self.hidden)
            or hidden.device != values.device
            or hidden.dtype != torch.float32
        ):
            raise ValueError(
                "GRU state must match the model layers, batch, device, and float32 dtype"
            )
        return hidden, history, position

    def forward(
        self, values: Tensor, state: Tensor | GRUSequenceState | None = None
    ) -> tuple[Tensor, Tensor, Tensor | GRUSequenceState]:
        """Predict rows from current/past normalized inputs and return streaming state."""
        check_sequence_input(self, values, self.precision_mode)
        hidden, history, position = self._hidden_and_history(values, state)
        standardized = (values - self.mean) / self.std
        next_history: Tensor | None = None
        if self.input_stem is not None:
            standardized, next_history = self.input_stem(standardized, history)
        encoded = self.encode(standardized)
        recurrent, next_hidden = self.gru(encoded, hidden)
        prediction = self.head(torch.cat((recurrent, encoded), dim=-1))
        next_hidden = next_hidden.float()
        if self.input_stem_kernel > 1:
            if next_history is None:
                raise RuntimeError("stateful input stem failed to return its causal history")
            next_state: Tensor | GRUSequenceState = GRUSequenceState(
                next_hidden, next_history, position + values.shape[1]
            )
        else:
            next_state = next_hidden
        return prediction, self.hint_projector(recurrent), next_state
