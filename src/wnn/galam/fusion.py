from __future__ import annotations

import torch
from torch import nn

from wnn.galam.config import GALAMConfig


class TwoWayFusionGate(nn.Module):
    """Fuse global and local streams with two learned sigmoid gates."""

    def __init__(self, width: int, operand_order: tuple[str, str]) -> None:
        """Create the gate context and projected local hidden/cell value."""
        super().__init__()
        self.operand_order = operand_order
        self.local_value = nn.Linear(3 * width, width)
        self.gates = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )

    def forward(
        self,
        global_memory: torch.Tensor,
        local_memory: torch.Tensor,
        hidden_projection: torch.Tensor,
        cell_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Return a gate-weighted sum with configured global/local operand order."""
        local_value = self.local_value(
            torch.cat((local_memory, hidden_projection, cell_projection), dim=-1)
        )
        context = torch.cat(
            (global_memory, local_memory, hidden_projection, cell_projection), dim=-1
        )
        gates = torch.sigmoid(self.gates(context)).chunk(2, dim=-1)
        values = {"global": global_memory, "local": local_value}
        return gates[0] * values[self.operand_order[0]] + gates[1] * values[self.operand_order[1]]


def make_fusion(config: GALAMConfig) -> TwoWayFusionGate:
    """Build the explicitly ordered two-stream fusion specified in a config."""
    return TwoWayFusionGate(config.width, config.fusion_operand_order)
