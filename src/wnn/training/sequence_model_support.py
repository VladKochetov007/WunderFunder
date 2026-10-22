from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _HintProjector(nn.Module):
    """Produce learned hints or a fixed, normalized random representation."""

    def __init__(self, hidden: int, hint_dim: int, mode: str) -> None:
        super().__init__()
        self.hint_dim = hint_dim
        self.linear: nn.Linear | None = None
        self.normalizer: nn.LayerNorm | None = None
        self.register_buffer("fixed_projection", None)
        if hint_dim == 0:
            return
        if mode == "learned":
            self.linear = nn.Linear(hidden, hint_dim)
            return
        self.normalizer = nn.LayerNorm(hidden, elementwise_affine=False)
        generator = torch.Generator(device="cpu").manual_seed(1729)
        projection = torch.randn(hint_dim, hidden, generator=generator, dtype=torch.float32)
        self.fixed_projection = projection / math.sqrt(hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        """Project hidden row vectors when hints are enabled."""
        if self.hint_dim == 0:
            return hidden.new_empty((*hidden.shape[:-1], 0))
        if self.linear is not None:
            return self.linear(hidden)
        if self.normalizer is None or self.fixed_projection is None:
            raise RuntimeError("fixed hint projection was not initialized")
        return F.linear(self.normalizer(hidden), self.fixed_projection)


def _check_fp32_environment(model: nn.Module, values: Tensor) -> None:
    if values.dtype != torch.float32:
        raise TypeError("sequence models require float32 inputs")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("sequence models require float32 parameters")
    if any(buffer is not None and buffer.dtype != torch.float32 for buffer in model.buffers()):
        raise TypeError("sequence models require float32 buffers")
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("sequence models require autocast to be disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("sequence models require CUDA matmul and cuDNN TF32 to be disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("sequence models require float32 matmul precision 'highest'")


def _check_input_shape(values: Tensor, width: int = 112) -> None:
    if values.ndim != 3 or values.shape[-1] != width:
        raise ValueError(f"sequence model input must have shape [batch, rows, {width}]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("sequence batches and row chunks must be non-empty")
