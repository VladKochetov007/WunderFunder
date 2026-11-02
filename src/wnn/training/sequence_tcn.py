"""Sequence-runtime adapter for the project's existing causal TCN teacher."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from wnn.nn import TCNModel
from wnn.training.sequence_model_helpers import HintProjector
from wnn.training.sequence_precision import validate_forward_state


class CausalTCN(TCNModel):
    """Adapt the existing left-context TCN to the sequence trainer's tensor state."""

    def __init__(self, hidden: int, layers: int, hint_dim: int, hint_mode: str) -> None:
        if hidden <= 0 or not 1 <= layers <= 12:
            raise ValueError("TCN hidden must be positive and layers must be between 1 and 12")
        super().__init__(112, hidden, layers, n_out=2, mix=False)
        self.hidden = hidden
        self.history_lengths = tuple(2**i for i in range(layers))
        self.hint_projector = HintProjector(hidden, hint_dim, hint_mode)

    def forward(self, values: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return predictions, representation hints and per-layer history packed in a tensor."""
        if values.ndim != 3 or values.shape[-1] != 112 or min(values.shape[:2]) <= 0:
            raise ValueError("TCN inputs must have shape [batch, nonempty rows, 112]")
        if values.dtype != torch.float32:
            raise TypeError("TCN inputs must be float32")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("TCN parameters must be float32")
        validate_forward_state("fp32", values.device.type)
        expected_shape = (values.shape[0], sum(self.history_lengths), self.hidden)
        if state is None:
            tails = None
        else:
            if state.shape != expected_shape or state.device != values.device or state.dtype != values.dtype:
                raise ValueError("TCN history shape, device or dtype is incompatible")
            tails = torch.split(state, self.history_lengths, dim=1)
        projected = F.gelu(self.inp((values - self.mean) / self.std))
        hidden, next_tails = self.tcn(projected, tails)
        prediction = self.head(torch.cat([hidden, projected], dim=-1))
        return prediction, self.hint_projector(hidden), torch.cat(next_tails, dim=1)
