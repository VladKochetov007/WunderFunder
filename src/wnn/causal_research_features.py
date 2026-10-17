"""Small causal features for research models; no fitted or future inputs."""

from __future__ import annotations

import math

import torch
from torch import nn


def _affine_prefix_scan(
    decay: torch.Tensor, offset: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``y[t] = decay[t] * y[t-1] + offset[t]`` by prefix composition."""
    if decay.shape != offset.shape or decay.ndim != 3:
        raise ValueError("decay and offset must have the same (batch, rows, features) shape")
    batch_size, rows, width = decay.shape
    if rows == 0 or initial.shape != (batch_size, width):
        raise ValueError("nonempty rows and a matching (batch, features) initial state required")

    prefix_decay = decay
    prefix_offset = offset
    distance = 1
    while distance < rows:
        previous_decay = prefix_decay
        previous_offset = prefix_offset
        shifted_decay = torch.cat(
            (torch.ones_like(previous_decay[:, :distance]), previous_decay[:, :-distance]),
            dim=1,
        )
        shifted_offset = torch.cat(
            (torch.zeros_like(previous_offset[:, :distance]), previous_offset[:, :-distance]),
            dim=1,
        )
        prefix_offset = previous_offset + previous_decay * shifted_offset
        prefix_decay = previous_decay * shifted_decay
        distance *= 2
    return prefix_offset + prefix_decay * initial.unsqueeze(1)


class CausalExponentialSmoother(nn.Module):
    """Smooth each feature with a fixed half-life and an explicit sequence reset.

    A call without ``state`` starts a sequence by setting its initial value to
    the first observed row. Pass the returned state to continue a chunk. Reset
    by omitting it at the next sequence boundary.
    """

    def __init__(self, half_life: float) -> None:
        super().__init__()
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be finite and positive")
        self.half_life = float(half_life)
        self.decay = 2.0 ** (-1.0 / self.half_life)

    def forward(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smoothed ``(batch, rows, features)`` values and the final state."""
        if values.ndim != 3 or values.shape[1] == 0:
            raise ValueError("values must have nonempty (batch, rows, features) shape")
        if state is None:
            state = values[:, 0]
        elif state.shape != (values.shape[0], values.shape[2]):
            raise ValueError("state must match the input batch and feature dimensions")
        elif state.device != values.device or state.dtype != values.dtype:
            raise ValueError("state must match the input device and dtype")

        decay = torch.full_like(values, self.decay)
        offset = values * (1.0 - self.decay)
        smoothed = _affine_prefix_scan(decay, offset, state)
        return smoothed, smoothed[:, -1]


class TrailingCircleResidual(nn.Module):
    """Fit a regularized 2D circle to prior rows and measure the current residual.

    The fit for row ``t`` uses only rows ``[max(0, t-window), t)``. The first
    three rows and geometrically degenerate windows return zero with a false
    validity flag. This is a diagnostic geometric feature, not a financial
    interpretation of either selected input column.
    """

    def __init__(
        self,
        columns: tuple[int, int],
        window: int,
        ridge: float = 1e-3,
    ) -> None:
        super().__init__()
        if len(columns) != 2 or columns[0] == columns[1]:
            raise ValueError("columns must contain two distinct feature indices")
        if min(columns) < 0 or max(columns) >= 112:
            raise ValueError("column indices must be in [0, 111]")
        if window < 3:
            raise ValueError("window must be at least three rows")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.columns = columns
        self.window = window
        self.ridge = float(ridge)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized radial residuals and a mask for usable past fits."""
        if values.ndim != 3 or values.shape[-1] != 112 or values.shape[1] == 0:
            raise ValueError("values must have nonempty (batch, rows, 112) shape")

        # Float64 keeps the small normal-equation solve stable for raw columns
        # with different magnitudes. The returned feature uses the input dtype.
        points = values[..., list(self.columns)].to(dtype=torch.float64)
        x, y = points.unbind(dim=-1)
        x2, y2 = x.square(), y.square()
        basis = torch.stack((x, y, x2, y2, x * y, x2 * x, x * y2, x2 * y, y2 * y), dim=-1)
        prefix = torch.cat(
            (
                torch.zeros_like(basis[:, :1]),
                basis.cumsum(dim=1),
            ),
            dim=1,
        )
        rows = values.shape[1]
        ends = torch.arange(rows, device=values.device)
        starts = (ends - self.window).clamp_min(0)
        window_sums = prefix.index_select(1, ends) - prefix.index_select(1, starts)
        count = (ends - starts).to(dtype=torch.float64).view(1, rows)

        sum_x, sum_y = window_sums[..., 0], window_sums[..., 1]
        sum_x2, sum_y2 = window_sums[..., 2], window_sums[..., 3]
        sum_xy, sum_x3 = window_sums[..., 4], window_sums[..., 5]
        sum_xy2, sum_x2y, sum_y3 = (
            window_sums[..., 6],
            window_sums[..., 7],
            window_sums[..., 8],
        )
        safe_count = count.clamp_min(1.0)
        mean_x, mean_y = sum_x / safe_count, sum_y / safe_count
        centered_x2 = (sum_x2 - count * mean_x.square()).clamp_min(0.0)
        centered_y2 = (sum_y2 - count * mean_y.square()).clamp_min(0.0)
        centered_xy = sum_xy - count * mean_x * mean_y
        centered_x3 = sum_x3 - 3.0 * mean_x * sum_x2 + 2.0 * count * mean_x.pow(3)
        centered_xy2 = (
            sum_xy2
            - 2.0 * mean_y * sum_xy
            - mean_x * sum_y2
            + 2.0 * count * mean_x * mean_y.square()
        )
        centered_x2y = (
            sum_x2y
            - 2.0 * mean_x * sum_xy
            - mean_y * sum_x2
            + 2.0 * count * mean_x.square() * mean_y
        )
        centered_y3 = sum_y3 - 3.0 * mean_y * sum_y2 + 2.0 * count * mean_y.pow(3)

        centered_radius_sq = centered_x2 + centered_y2
        scale = torch.sqrt((centered_radius_sq / (2.0 * safe_count)).clamp_min(1e-24))
        scale2, scale3 = scale.square(), scale.pow(3)
        matrix_xx = centered_x2 / scale2 + self.ridge * count
        matrix_yy = centered_y2 / scale2 + self.ridge * count
        matrix_xy = centered_xy / scale2
        rhs_x = -(centered_x3 + centered_xy2) / scale3
        rhs_y = -(centered_x2y + centered_y3) / scale3
        determinant = (matrix_xx * matrix_yy - matrix_xy.square()).clamp_min(1e-24)
        coefficient_x = (rhs_x * matrix_yy - matrix_xy * rhs_y) / determinant
        coefficient_y = (matrix_xx * rhs_y - matrix_xy * rhs_x) / determinant
        radius_sq = (coefficient_x.square() + coefficient_y.square()) / 4.0 + 2.0
        radius = torch.sqrt(radius_sq.clamp_min(0.0))

        current_x = (x - mean_x) / scale + coefficient_x / 2.0
        current_y = (y - mean_y) / scale + coefficient_y / 2.0
        residual = torch.sqrt(current_x.square() + current_y.square()) - radius
        covariance_det = (centered_x2 * centered_y2 - centered_xy.square()).clamp_min(0.0)
        covariance_scale = (centered_x2 + centered_y2).square().clamp_min(1e-24)
        valid = (
            (count >= 3) & (centered_radius_sq > 1e-12) & (covariance_det > 1e-8 * covariance_scale)
        )
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual.to(dtype=values.dtype).unsqueeze(-1), valid
