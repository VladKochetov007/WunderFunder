"""Causal residual filtering over normalized raw sequence inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RAW_INPUT_WIDTH = 112
INPUT_STEM_HISTORY_ROWS = 15
INPUT_STEM_KERNELS = (0, 1, 16)


def validate_input_stem_kernel(value: object) -> int:
    """Validate a supported causal raw-input stem kernel size."""
    if type(value) is not int:
        raise TypeError("input_stem_kernel must be an integer")
    if value not in INPUT_STEM_KERNELS:
        raise ValueError("input_stem_kernel must be 0, 1, or 16")
    return value


def validate_input_history(
    history: Tensor | None, values: Tensor, next_position: int, kernel_size: int
) -> Tensor | None:
    """Check a carried normalized-input tail against its causal position."""
    if kernel_size != 16:
        if history is not None:
            raise ValueError("input history is valid only for input_stem_kernel=16")
        return None
    if history is None:
        if next_position != 0:
            raise ValueError("kernel-16 Transformer state is missing input history")
        return None
    if not isinstance(history, Tensor):
        raise TypeError("Transformer input history must be a tensor")
    if (
        history.ndim != 3
        or history.shape[0] != values.shape[0]
        or history.shape[-1] != RAW_INPUT_WIDTH
    ):
        raise ValueError("Transformer input history has an invalid batch or feature shape")
    expected_rows = min(next_position, INPUT_STEM_HISTORY_ROWS)
    if history.shape[1] != expected_rows:
        raise ValueError("Transformer input history has an invalid row count")
    if history.device != values.device or history.dtype != torch.float32:
        raise ValueError("Transformer input history must use the input device and float32 dtype")
    return history


class CausalInputStem(nn.Module):
    """Filter standardized raw rows with an identity-initialized depthwise kernel."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if type(kernel_size) is not int:
            raise TypeError("causal input stem kernel must be an integer")
        if kernel_size not in (1, 16):
            raise ValueError("causal input stem kernel must be 1 or 16")
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(RAW_INPUT_WIDTH, 1, kernel_size))

    def forward(
        self, values: Tensor, history: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return residual-filtered rows and the bounded standardized raw-input tail."""
        if self.kernel_size == 1:
            if history is not None:
                raise ValueError("kernel-1 input stems do not carry input history")
            source = values.transpose(1, 2)
            next_history = None
        else:
            past = values[:, :0] if history is None else history
            joined = torch.cat((past, values), dim=1)
            source = joined.transpose(1, 2)
            source = F.pad(source, (INPUT_STEM_HISTORY_ROWS, 0))
            next_history = joined[:, -INPUT_STEM_HISTORY_ROWS:].clone()
        filtered = F.conv1d(source, self.weight, groups=RAW_INPUT_WIDTH)
        filtered = filtered[..., -values.shape[1] :].transpose(1, 2)
        return values + filtered, next_history
