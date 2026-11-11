"""Target-private low-rank readout adapters for impact experiment card 03."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TargetPrivateHead(nn.Module):
    """Apply separate rank-eight nonlinear adapters before a shared linear head.

    The supplied two-row ``base`` layer remains the only output readout. Each
    target receives an adapted hidden vector, and its original row contributes
    the corresponding correction to the unchanged base prediction.
    """

    def __init__(self, base: nn.Linear, rank: int = 8) -> None:
        """Create two target-specific bottleneck adapters around ``base``.

        Args:
            base: Existing linear output head with exactly two target rows.
            rank: Bottleneck width for each target-private adapter.

        Raises:
            TypeError: If rank is not an integer.
            ValueError: If the base head or rank has invalid dimensions.
        """
        super().__init__()
        if base.out_features != 2:
            raise ValueError("target-private head requires a base Linear with two outputs")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        self.rank = rank
        self.down = nn.ModuleList(nn.Linear(base.in_features, rank) for _ in range(2))
        self.up = nn.ModuleList(nn.Linear(rank, base.in_features) for _ in range(2))
        for layer in self.up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return the base prediction plus each target's low-rank correction."""
        if hidden.ndim < 1 or hidden.shape[-1] != self.base.in_features:
            raise ValueError("hidden states must end in the base head's input width")
        base_prediction = self.base(hidden)
        outputs: list[Tensor] = []
        for target in range(2):
            residual = self.up[target](F.gelu(self.down[target](hidden)))
            correction = F.linear(
                residual,
                self.base.weight[target : target + 1],
                bias=None,
            )
            outputs.append(base_prediction[..., target : target + 1] + correction)
        return torch.cat(outputs, dim=-1)
