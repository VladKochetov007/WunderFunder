from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class BranchChannelProductGate(nn.Module):
    """Use a three-branch decoder gate for an explicitly adapted GALAM graph."""

    def __init__(self, width: int, channel_order: tuple[str, str, str]) -> None:
        """Create a gate whose width follows the selected branch ablation."""
        super().__init__()
        allowed = {
            ("global_memory", "local_memory", "hidden_projection"),
            ("local_memory", "hidden_projection", "cell_projection"),
        }
        if channel_order not in allowed:
            raise ValueError("adapted GALAM fusion requires a declared three-branch ordering")
        self.channel_order = channel_order
        decoder_width = len(channel_order) * width
        self.gates = nn.Sequential(
            nn.Linear(decoder_width, decoder_width),
            nn.SiLU(),
            nn.Linear(decoder_width, decoder_width),
        )

    def forward(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Multiply the ordered three-branch decoder values by their learned sigmoid gate."""
        if set(sources) != set(self.channel_order):
            raise ValueError("fusion inputs must match the selected GALAM branch variant")
        bypass = torch.cat(tuple(sources[name] for name in self.channel_order), dim=-1)
        return bypass * torch.sigmoid(self.gates(bypass))
