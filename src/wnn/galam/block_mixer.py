from __future__ import annotations

import torch
from torch.nn import functional as F

from wnn.galam.mixer import CausalConv1d, CausalConvMixer
from wnn.galam.state import ConvMixerState


def convolution_chunk(
    module: CausalConv1d, x: torch.Tensor, history: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve the chunk using exactly the carried left tail and no right padding."""
    if x.ndim != 3 or x.shape[-1] != module.input_channels:
        raise ValueError("block convolution requires [batch, time, configured channels]")
    if history.shape != (x.shape[0], module.input_channels, module.kernel_size - 1):
        raise ValueError("block convolution history has an incompatible shape")
    sequence = torch.cat((history, x.transpose(1, 2)), dim=-1)
    output = module.conv(sequence).transpose(1, 2)
    tail = (
        sequence[:, :, -(module.kernel_size - 1) :]
        if module.kernel_size > 1
        else sequence[:, :, :0]
    )
    return output, tail


def mixer_chunk(
    module: CausalConvMixer, x: torch.Tensor, state: ConvMixerState | None
) -> tuple[torch.Tensor, ConvMixerState]:
    """Batch both causal convolutions and preserve per-time BatchNorm semantics."""
    first_tail = module.conv1.empty_history(x[:, 0]) if state is None else state.first_inputs
    first, first_tail = convolution_chunk(module.conv1, x, first_tail)
    first = module.norm1(F.gelu(first))
    second_tail = module.conv2.empty_history(first[:, 0]) if state is None else state.second_inputs
    second, second_tail = convolution_chunk(module.conv2, first, second_tail)
    second = module.norm2(F.gelu(second))
    return second, ConvMixerState(first_tail, second_tail)
