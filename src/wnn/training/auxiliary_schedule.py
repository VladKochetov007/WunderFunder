from __future__ import annotations


def auxiliary_loss_coefficients(
    schedule: str,
    structural_weight: float,
    next_weight: float,
    step: int,
    total_steps: int,
) -> tuple[float, float]:
    """Return scheduled structural and next-change loss coefficients.

    The default schedule returns the configured floats unchanged. The opt-in late
    schedule holds the configured weights through half of training, then decays them
    linearly to zero on the final optimizer update.
    """
    if schedule == "none":
        return structural_weight, next_weight
    if schedule != "late_linear_to_zero":
        raise ValueError(f"unsupported auxiliary schedule: {schedule}")
    if total_steps < 2:
        raise ValueError("late_linear_to_zero requires at least two optimizer updates")
    if not 0 <= step < total_steps:
        raise ValueError(f"optimizer step {step} is outside [0, {total_steps})")
    progress = step / (total_steps - 1)
    scale = 1.0 if progress <= 0.5 else 2.0 * (1.0 - progress)
    return structural_weight * scale, next_weight * scale
