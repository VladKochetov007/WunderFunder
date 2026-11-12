"""Matched additive and bilinear residual adapters for the raw-112 input map."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

InputVariant = Literal["input_linear", "input_bilinear"]
RAW_INPUT_WIDTH = 112
STREAM_WIDTH = 52
AUXILIARY_START = 104
BILINEAR_RANK = 32


class InputBlock(nn.Module):
    """Keep the native projection and add a matched two-stream residual branch.

    ``input_linear`` combines the two rank-32 stream projections additively;
    ``input_bilinear`` combines the same projections elementwise. The output
    matrix starts at zero, so both adapters preserve the native initial function.
    """

    def __init__(self, native_projection: nn.Linear, variant: InputVariant) -> None:
        """Reuse native projection tensors and initialize only the extra branch."""
        super().__init__()
        if variant not in ("input_linear", "input_bilinear"):
            raise ValueError("unsupported input adapter variant")
        if native_projection.in_features != RAW_INPUT_WIDTH or native_projection.bias is None:
            raise ValueError("input adapter requires the canonical biased raw-112 projection")

        # Register the existing parameters directly so canonical checkpoint
        # names and the pinned initialization remain unchanged.
        self.weight = native_projection.weight
        self.bias = native_projection.bias
        branch_device = native_projection.weight.device
        branch_dtype = native_projection.weight.dtype
        self.bilinear_u = nn.Linear(
            STREAM_WIDTH, BILINEAR_RANK, bias=False, device=branch_device, dtype=branch_dtype
        )
        self.bilinear_v = nn.Linear(
            STREAM_WIDTH, BILINEAR_RANK, bias=False, device=branch_device, dtype=branch_dtype
        )
        self.bilinear_output = nn.Linear(
            BILINEAR_RANK,
            native_projection.out_features,
            bias=False,
            device=branch_device,
            dtype=branch_dtype,
        )
        with torch.no_grad():
            self.bilinear_output.weight.zero_()
        self.variant = variant
        self.in_features = native_projection.in_features
        self.out_features = native_projection.out_features

    def forward(self, normalized_values: Tensor) -> Tensor:
        """Project all normalized raw features and add the selected stream route."""
        if normalized_values.ndim != 3 or normalized_values.shape[-1] != RAW_INPUT_WIDTH:
            raise ValueError("input adapter accepts only [batch, rows, 112] values")
        native = F.linear(normalized_values, self.weight, self.bias)
        stream_zero = self.bilinear_u(normalized_values[..., :STREAM_WIDTH])
        stream_one = self.bilinear_v(normalized_values[..., STREAM_WIDTH:AUXILIARY_START])
        if self.variant == "input_linear":
            routed = (stream_zero + stream_one) / math.sqrt(2.0)
        else:
            routed = stream_zero * stream_one
        return native + self.bilinear_output(routed)
