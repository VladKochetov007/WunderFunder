"""Small zero-initialized parent-residual model used by the cached age screen."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AgeResidualHead(nn.Module):
    """Return the fixed parent prediction plus one tenth of a tanh residual."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(323, 64)
        self.output = nn.Linear(64, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor, parent: Tensor) -> Tensor:
        """Return the cached parent prediction plus the bounded learned residual."""
        if features.shape[-1] != 323 or parent.shape[-1] != 2:
            raise ValueError("age readout expects 323 features and two parent outputs")
        residual = self.output(torch.tanh(self.hidden(features)))
        return parent + 0.1 * residual


def zero_head_parity() -> bool:
    """Prove that a fresh head returns its parent prediction exactly."""
    head = AgeResidualHead()
    features = torch.linspace(-1.0, 1.0, 323 * 5, dtype=torch.float32).reshape(5, 323)
    parent = torch.linspace(-2.0, 2.0, 10, dtype=torch.float32).reshape(5, 2)
    if not torch.equal(head(features, parent), parent):
        raise AssertionError("zero-initialized residual changed the parent prediction")
    return True
