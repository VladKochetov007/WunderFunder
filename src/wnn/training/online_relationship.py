from __future__ import annotations

import torch
from torch import nn

from wnn.online_relationship import MODEL_INPUT_WIDTH


class OnlineRelationshipResidualHead(nn.Module):
    """Add a zero-initialized, bounded-scale residual to parent predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(MODEL_INPUT_WIDTH, 64)
        self.output = nn.Linear(64, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, parent_prediction: torch.Tensor) -> torch.Tensor:
        """Return parent predictions plus one tenth of the learned residual."""
        residual = self.output(torch.tanh(self.hidden(features)))
        return parent_prediction + 0.1 * residual
