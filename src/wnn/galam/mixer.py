from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from wnn.galam.state import ConvMixerState


class PerTimeBatchNorm1d(nn.Module):
    """Apply BatchNorm1d to each timestamp independently across the batch axis."""

    def __init__(self, features: int) -> None:
        """Create channel statistics that never combine different time positions."""
        super().__init__()
        self.norm = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one time position shaped batch by features."""
        if x.ndim != 2:
            raise ValueError("per-time BatchNorm input must have shape [batch, features]")
        return self.norm(x)


class CausalConv1d(nn.Module):
    """A left-padded Conv1d with a bounded history API for one-step inference."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        """Create a same-width causal convolution."""
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size)

    def forward_step(
        self, x: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve one feature vector and return the updated left context."""
        sequence = torch.cat((history, x.unsqueeze(-1)), dim=-1)
        output = self.conv(sequence)
        next_history = (
            sequence[:, :, -(self.kernel_size - 1) :]
            if self.kernel_size > 1
            else sequence[:, :, :0]
        )
        return output.squeeze(-1), next_history

    def empty_history(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate a zero causal tail compatible with one batch of vectors."""
        return x.new_zeros((x.shape[0], x.shape[1], max(0, self.kernel_size - 1)))


class CausalConvMixer(nn.Module):
    """Two causal Conv1d layers with per-timestamp BatchNorm1d and GELU."""

    def __init__(self, width: int, kernel_size: int) -> None:
        """Create the pictured BatchNorm-GELU-Conv sequence twice."""
        super().__init__()
        self.norm1 = PerTimeBatchNorm1d(width)
        self.norm2 = PerTimeBatchNorm1d(width)
        self.conv1 = CausalConv1d(width, kernel_size)
        self.conv2 = CausalConv1d(width, kernel_size)

    def forward_step(
        self, x: torch.Tensor, state: ConvMixerState | None
    ) -> tuple[torch.Tensor, ConvMixerState]:
        """Run one causal mixer step and update both bounded convolution tails."""
        first_history = self.conv1.empty_history(x) if state is None else state.first_inputs
        second_history = self.conv2.empty_history(x) if state is None else state.second_inputs
        first, first_history = self.conv1.forward_step(F.gelu(self.norm1(x)), first_history)
        second, second_history = self.conv2.forward_step(F.gelu(self.norm2(first)), second_history)
        return second, ConvMixerState(first_history, second_history)
