"""Causal residual FIR filtering over normalized raw sequence inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RAW_INPUT_WIDTH = 112
INPUT_STEM_HISTORY_ROWS = 15
INPUT_STEM_KERNELS = (0, 1, 16, 64)


def validate_input_stem_kernel(value: object) -> int:
    """Return a supported raw-input stem kernel size."""
    if type(value) is not int:
        raise TypeError("input_stem_kernel must be an integer")
    if value not in INPUT_STEM_KERNELS:
        raise ValueError("input_stem_kernel must be 0, 1, 16, or 64")
    return value


def validate_input_history(
    history: Tensor | None, values: Tensor, next_position: int, kernel_size: int = 16
) -> Tensor | None:
    """Validate the normalized-input tail for a stateful causal kernel."""
    if type(next_position) is not int or next_position < 1:
        raise ValueError("GRU input history has an invalid absolute position")
    if history is None:
        raise ValueError("stateful GRU stem is missing input history")
    if not isinstance(history, Tensor):
        raise TypeError("GRU input history must be a tensor")
    kernel_size = validate_input_stem_kernel(kernel_size)
    if kernel_size <= 1:
        raise ValueError("input history requires a kernel larger than one")
    history_rows = kernel_size - 1
    if (
        history.ndim != 3
        or history.shape[0] != values.shape[0]
        or history.shape[-1] != RAW_INPUT_WIDTH
    ):
        raise ValueError("GRU input history has an invalid batch or feature shape")
    if history.shape[1] != min(next_position, history_rows):
        raise ValueError("GRU input history has an invalid row count")
    if history.device != values.device or history.dtype != torch.float32:
        raise ValueError("GRU input history must use the input device and float32 dtype")
    return history


class CausalInputStem(nn.Module):
    """Filter normalized rows with an identity-initialized per-channel FIR kernel."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        kernel_size = validate_input_stem_kernel(kernel_size)
        if kernel_size not in (1, 16, 64):
            raise ValueError("causal input stem kernel must be 1, 16, or 64")
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(RAW_INPUT_WIDTH, 1, kernel_size))

    def forward(
        self, values: Tensor, history: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return residual-filtered rows and the bounded input tail for the next chunk."""
        if values.ndim != 3 or values.shape[-1] != RAW_INPUT_WIDTH:
            raise ValueError("input stem values must have shape [batch, rows, 112]")
        if values.shape[0] <= 0 or values.shape[1] <= 0 or values.dtype != torch.float32:
            raise ValueError("input stem values must be non-empty float32 batches")
        if values.device != self.weight.device:
            raise ValueError("input stem values and parameters must use the same device")
        if self.kernel_size == 1:
            if history is not None:
                raise ValueError("kernel-1 input stems do not carry input history")
            source = values.transpose(1, 2)
            next_history = None
        else:
            history_rows = self.kernel_size - 1
            if history is None:
                past = values[:, :0]
            else:
                if not isinstance(history, Tensor):
                    raise TypeError("causal input history must be a tensor")
                if (
                    history.ndim != 3
                    or history.shape[0] != values.shape[0]
                    or history.shape[-1] != RAW_INPUT_WIDTH
                    or history.device != values.device
                    or history.dtype != torch.float32
                    or history.shape[1] > history_rows
                ):
                    raise ValueError("causal input history has an invalid shape or dtype")
                past = history
            joined = torch.cat((past, values), dim=1)
            source = F.pad(joined.transpose(1, 2), (history_rows, 0))
            next_history = joined[:, -history_rows:].clone()
        filtered = F.conv1d(source, self.weight, groups=RAW_INPUT_WIDTH)
        filtered = filtered[..., -values.shape[1] :].transpose(1, 2)
        return values + filtered, next_history
