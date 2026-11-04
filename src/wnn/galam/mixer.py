from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from wnn.galam.state import ConvMixerState


class PerTimeBatchNorm1d(nn.Module):
    """Apply BatchNorm1d per timestamp; training requires at least two batch rows."""

    def __init__(self, features: int) -> None:
        """Create channel statistics that never combine different time positions."""
        super().__init__()
        self.norm = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one time position shaped batch by features."""
        if x.ndim != 2:
            raise ValueError("per-time BatchNorm input must have shape [batch, features]")
        if self.training and x.shape[0] < 2:
            raise ValueError("training-time BatchNorm requires batch size >= 2")
        return self.norm(x)


class CausalConv1d(nn.Module):
    """A left-padded Conv1d with a bounded history API for one-step inference."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int) -> None:
        """Create a causal convolution with an explicit channel transition."""
        super().__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)

    def forward_step(
        self, x: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve one feature vector and return the updated left context."""
        if x.ndim != 2 or x.shape[1] != self.input_channels:
            raise ValueError("causal convolution input must match its configured channels")
        if history.shape != (x.shape[0], self.input_channels, self.kernel_size - 1):
            raise ValueError("causal convolution history has an incompatible shape")
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
    """Apply Conv-GELU-BatchNorm twice with an explicit hidden channel width."""

    def __init__(self, width: int, hidden_width: int, kernel_size: int) -> None:
        """Create the schematic Conv1d-GELU-BatchNorm1d module."""
        super().__init__()
        self.norm1 = PerTimeBatchNorm1d(hidden_width)
        self.norm2 = PerTimeBatchNorm1d(width)
        self.conv1 = CausalConv1d(width, hidden_width, kernel_size)
        self.conv2 = CausalConv1d(hidden_width, width, kernel_size)

    def forward_step(
        self, x: torch.Tensor, state: ConvMixerState | None
    ) -> tuple[torch.Tensor, ConvMixerState]:
        """Run one causal mixer step and update both bounded convolution tails."""
        first_history = self.conv1.empty_history(x) if state is None else state.first_inputs
        first_conv, first_history = self.conv1.forward_step(x, first_history)
        first = self.norm1(F.gelu(first_conv))
        second_history = self.conv2.empty_history(first) if state is None else state.second_inputs
        second_conv, second_history = self.conv2.forward_step(first, second_history)
        second = self.norm2(F.gelu(second_conv))
        return second, ConvMixerState(first_history, second_history)
