"""Canonical FP32 LSTM adapter for source-bound sequence fitting."""

from __future__ import annotations

import torch
from torch import Tensor

from wnn.nn import RNNModel
from wnn.training.sequence_features import RAW_INPUT_MODE, sequence_input_width
from wnn.training.sequence_model_helpers import (
    HintProjector,
    validate_sequence_fp32_environment,
    validate_sequence_input_shape,
)


class LSTMSequenceAdapter(RNNModel):
    """Expose the legacy LSTM cell through the resumable sequence-fit API."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        dropout: float,
        hint_dim: int,
        hint_mode: str,
        precision_mode: str,
    ) -> None:
        super().__init__(
            sequence_input_width(RAW_INPUT_MODE),
            hidden,
            layers,
            dropout,
            n_out=2,
            cell="lstm",
            enc="linear",
        )
        self.precision_mode = precision_mode
        self.hint_projector = HintProjector(hidden, hint_dim, hint_mode)

    def forward(
        self,
        values: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Return predictions, representation hints, and the next hidden/cell state."""
        validate_sequence_input_shape(values, self.mean.numel())
        validate_sequence_fp32_environment(self, values, self.precision_mode)
        if state is not None:
            expected = (self.gru.num_layers, values.shape[0], self.hidden)
            if any(value.shape != expected or value.device != values.device for value in state):
                raise ValueError("LSTM state must match layers, batch, width, and device")
            if any(value.dtype != torch.float32 for value in state):
                raise ValueError("LSTM state must use float32")
        encoded = self.encode((values - self.mean) / self.std)
        recurrent, next_state = self.gru(encoded, state)
        prediction = self.head(torch.cat((recurrent, encoded), dim=-1))
        return prediction, self.hint_projector(recurrent), next_state
