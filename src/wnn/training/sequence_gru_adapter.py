"""Canonical recurrent adapter for raw or deterministic event-expanded inputs."""

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


class GRUSequenceAdapter(RNNModel):
    """Expose canonical GRU predictions, hints, and resumable hidden state."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        dropout: float,
        hint_dim: int,
        hint_mode: str,
        precision_mode: str,
        input_feature_mode: str = RAW_INPUT_MODE,
    ) -> None:
        input_width = sequence_input_width(input_feature_mode)
        super().__init__(input_width, hidden, layers, dropout, n_out=2, cell="gru", enc="linear")
        self.input_feature_mode = input_feature_mode
        self.precision_mode = precision_mode
        self.hint_projector = HintProjector(hidden, hint_dim, hint_mode)

    def forward(self, values: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return predictions, representation hints, and the next GRU hidden state."""
        validate_sequence_input_shape(values, self.mean.numel())
        validate_sequence_fp32_environment(self, values, self.precision_mode)
        if state is not None and (
            state.shape != (self.gru.num_layers, values.shape[0], self.hidden)
            or state.device != values.device
            or state.dtype != torch.float32
        ):
            raise ValueError(
                "GRU state must match the model layers, batch, device, and float32 dtype"
            )
        encoded = self.encode((values - self.mean) / self.std)
        recurrent, next_state = self.gru(encoded, state)
        prediction = self.head(torch.cat((recurrent, encoded), dim=-1))
        return prediction, self.hint_projector(recurrent), next_state.float()
