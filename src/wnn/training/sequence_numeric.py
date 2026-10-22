from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES, StreamingEventFeatures, event_features

NUMERIC_INPUT_WIDTH = N_FEATURES * 2 + N_EVENT_FEATURES
NUMERIC_CONTINUOUS_WIDTH = N_FEATURES + N_EVENT_FEATURES
NUMERIC_MISSING_OFFSET = NUMERIC_CONTINUOUS_WIDTH
NUMERIC_BASIS_BINS = 16
NUMERIC_KNOT_COUNT = NUMERIC_BASIS_BINS + 1
NumericBasisMode = Literal["uniform16", "quantile16"]


def numeric_inputs(raw: np.ndarray) -> np.ndarray:
    """Impute nonfinite raw values, then append causal events and missing flags.

    Args:
        raw: One sequence shaped ``[rows, 112]``.

    Returns:
        Float32 values shaped ``[rows, 323]`` in raw, event, indicator order.

    Raises:
        ValueError: If the input shape is wrong or derived event values overflow.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] != N_FEATURES:
        raise ValueError("numeric inputs require an array shaped [rows, 112]")
    missing = ~np.isfinite(values)
    imputed = values.copy()
    imputed[missing] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        events = event_features(imputed)
    if events.shape != (len(imputed), N_EVENT_FEATURES) or not np.isfinite(events).all():
        raise ValueError("causal event features overflowed or have an invalid shape")
    result = np.concatenate((imputed, events, missing.astype(np.float32)), axis=-1)
    if result.shape != (len(imputed), NUMERIC_INPUT_WIDTH) or not np.isfinite(result).all():
        raise ValueError("numeric feature construction produced invalid values")
    return np.ascontiguousarray(result, dtype=np.float32)


class StreamingNumericInputs:
    """Make the same imputed 323-wide representation one row at a time."""

    def __init__(self) -> None:
        self._events = StreamingEventFeatures()

    def reset(self) -> None:
        """Reset event history at a new sequence boundary."""
        self._events.reset()

    def update(self, row: np.ndarray) -> np.ndarray:
        """Return one raw/event/missing vector while retaining causal state.

        Args:
            row: A raw row of exactly 112 values.

        Returns:
            A new float32 vector shaped ``[323]``.

        Raises:
            ValueError: If the input shape is wrong or event values overflow.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(row, dtype=np.float32)
        if values.shape != (N_FEATURES,):
            raise ValueError("streaming numeric input requires one 112-value row")
        missing = ~np.isfinite(values)
        imputed = values.copy()
        imputed[missing] = 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            events = self._events.update(imputed)
        if events.shape != (N_EVENT_FEATURES,) or not np.isfinite(events).all():
            raise ValueError("causal event features overflowed or have an invalid shape")
        result = np.concatenate((imputed, events, missing.astype(np.float32)))
        if not np.isfinite(result).all():
            raise ValueError("streaming numeric feature construction produced invalid values")
        return result


class NumericBasis(nn.Module):
    """Project sixteen clamped linear pieces per normalized raw input."""

    def __init__(self, hidden: int, mode: NumericBasisMode) -> None:
        super().__init__()
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        if mode not in ("uniform16", "quantile16"):
            raise ValueError("numeric basis mode must be uniform16 or quantile16")
        self.mode = mode
        self.register_buffer("knots", torch.zeros((N_FEATURES, NUMERIC_KNOT_COUNT)))
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(N_FEATURES * NUMERIC_BASIS_BINS, hidden, bias=False)
            nn.init.zeros_(self.projection.weight)

    def install_knots(self, knots: Tensor) -> None:
        """Validate and install fitting-only normalized raw-space knots.

        Args:
            knots: Float32 tensor shaped ``[112, 17]``, monotone by feature.

        Raises:
            TypeError: If knots are not a float32 tensor.
            ValueError: If knot shape, values, or ordering is invalid.
        """
        if not isinstance(knots, Tensor) or knots.dtype != torch.float32:
            raise TypeError("numeric basis knots must be a float32 tensor")
        if knots.shape != (N_FEATURES, NUMERIC_KNOT_COUNT):
            raise ValueError("numeric basis knots must have shape [112, 17]")
        if not torch.isfinite(knots).all():
            raise ValueError("numeric basis knots must be finite")
        if torch.any(knots[:, 1:] < knots[:, :-1]):
            raise ValueError("numeric basis knots must be nondecreasing")
        with torch.no_grad():
            self.knots.copy_(knots.to(device=self.knots.device))

    def forward(self, normalized_values: Tensor) -> Tensor:
        """Expand each normalized raw value and apply the zero-initialized map."""
        if normalized_values.ndim < 1 or normalized_values.shape[-1] != N_FEATURES:
            raise ValueError("numeric basis input must end in 112 raw features")
        if (
            normalized_values.dtype != torch.float32
            or self.knots.dtype != torch.float32
            or self.projection.weight.dtype != torch.float32
        ):
            raise TypeError("numeric basis requires float32 inputs, knots, and weights")
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("numeric basis requires autocast to be disabled")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("numeric basis requires CUDA matmul and cuDNN TF32 to be disabled")
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("numeric basis requires float32 matmul precision 'highest'")
        lower, upper = self.knots[:, :-1], self.knots[:, 1:]
        width = upper - lower
        nonzero_width = width > 0
        safe_width = torch.where(nonzero_width, width, torch.ones_like(width))
        positions = (normalized_values.unsqueeze(-1) - lower) / safe_width
        pieces = torch.where(nonzero_width, positions.clamp(0.0, 1.0), 0.0)
        return self.projection(pieces.flatten(start_dim=-2))
