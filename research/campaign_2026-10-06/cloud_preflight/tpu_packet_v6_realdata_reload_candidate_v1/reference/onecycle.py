"""Pure-Python port of the production PyTorch OneCycleLR scalar schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OneCyclePoint:
    """Learning rate and Adam beta values at one scheduler index."""

    learning_rate: float
    beta1: float
    beta2: float


def onecycle_point(
    step: int,
    total_steps: int,
    *,
    max_lr: float = 1e-3,
    pct_start: float = 0.05,
    div_factor: float = 25.0,
    final_div_factor: float = 100.0,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
    beta2: float = 0.999,
) -> OneCyclePoint:
    """Return production OneCycleLR values, including its final unused step.

    The accepted index range is ``0..total_steps``. The final index is useful
    when reproducing a trainer that calls ``scheduler.step()`` after its last
    optimizer update: PyTorch permits exactly that extra index and evaluates
    the cosine phase just beyond its endpoint. Optimizer updates use only
    indices ``0..total_steps - 1``.

    Args:
        step: Zero-based scheduler index.
        total_steps: Number of optimizer updates in the planned cycle.
        max_lr: Peak learning rate.
        pct_start: Fraction of the cycle assigned to the rising phase.
        div_factor: Initial learning-rate divisor.
        final_div_factor: Final learning-rate divisor relative to initial LR.
        base_momentum: Minimum beta1 during the cycle.
        max_momentum: Maximum beta1 during the cycle.
        beta2: Constant Adam beta2.

    Returns:
        The learning rate and Adam betas used at ``step``.

    Raises:
        TypeError: If step or total_steps is not an integer.
        ValueError: If any schedule input is outside the supported range.
    """
    if type(step) is not int or type(total_steps) is not int:
        raise TypeError("step and total_steps must be integers")
    if total_steps < 2 or not 0 <= step <= total_steps:
        raise ValueError("require total_steps >= 2 and 0 <= step <= total_steps")
    finite_values = (
        max_lr,
        pct_start,
        div_factor,
        final_div_factor,
        base_momentum,
        max_momentum,
        beta2,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("schedule parameters must be finite")
    if max_lr <= 0 or div_factor <= 0 or final_div_factor <= 0:
        raise ValueError("learning-rate scale parameters must be positive")
    if not 0 <= pct_start < 1:
        raise ValueError("pct_start must be in [0, 1)")
    if not 0 <= base_momentum <= max_momentum < 1:
        raise ValueError("momentum must satisfy 0 <= base <= max < 1")
    if not 0 <= beta2 < 1:
        raise ValueError("beta2 must be in [0, 1)")

    initial_lr = max_lr / div_factor
    minimum_lr = initial_lr / final_div_factor
    warm_end = pct_start * total_steps - 1.0
    if warm_end == 0.0:
        raise ValueError("pct_start and total_steps produce a degenerate warmup")
    final_index = float(total_steps - 1)

    if step <= warm_end:
        pct = step / warm_end
        learning_rate = _cosine_anneal(initial_lr, max_lr, pct)
        beta1 = _cosine_anneal(max_momentum, base_momentum, pct)
    else:
        cooling_width = final_index - warm_end
        if cooling_width <= 0:
            raise ValueError("pct_start leaves no cooling phase")
        pct = (step - warm_end) / cooling_width
        learning_rate = _cosine_anneal(max_lr, minimum_lr, pct)
        beta1 = _cosine_anneal(base_momentum, max_momentum, pct)
    return OneCyclePoint(learning_rate, beta1, beta2)


def _cosine_anneal(start: float, end: float, fraction: float) -> float:
    """Match torch.optim.lr_scheduler._annealing_cos without clamping."""
    return end + (start - end) / 2.0 * (math.cos(math.pi * fraction) + 1.0)
