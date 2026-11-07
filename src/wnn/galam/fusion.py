from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from wnn.galam.config import GALAMConfig


class DiagramChannelProductGate(nn.Module):
    """Apply a learned sigmoid gate to the ordered decoder channel bypass."""

    def __init__(
        self,
        width: int,
        mode: Literal["diagram_channel_product"],
        channel_order: tuple[str, str, str, str],
    ) -> None:
        """Create a 4-width Linear-SiLU-Linear gate from the schematic detail."""
        super().__init__()
        if mode != "diagram_channel_product":
            raise ValueError("fusion mode must be diagram_channel_product")
        self.width = width
        self.mode = mode
        self.channel_order = channel_order
        self.gates = nn.Sequential(
            nn.Linear(4 * width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, 4 * width),
        )

    def forward(
        self,
        global_memory: torch.Tensor,
        local_memory: torch.Tensor,
        hidden_projection: torch.Tensor,
        cell_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the ordered concatenated bypass by its sigmoid gate channels."""
        sources = {
            "global_memory": global_memory,
            "local_memory": local_memory,
            "hidden_projection": hidden_projection,
            "cell_projection": cell_projection,
        }
        bypass = torch.cat(tuple(sources[name] for name in self.channel_order), dim=-1)
        return bypass * torch.sigmoid(self.gates(bypass))


def make_fusion(config: GALAMConfig) -> DiagramChannelProductGate:
    """Build the configured diagram-channel product gate."""
    return DiagramChannelProductGate(config.width, config.fusion_mode, config.fusion_channel_order)
