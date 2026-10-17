"""Descriptive geometry models; these never feed full-sequence data into predictors."""

from __future__ import annotations

import numpy as np


def robust_arc(
    mean: np.ndarray, variance: np.ndarray, tolerance: float = 0.002
) -> dict[str, float]:
    """Find a train-only fixed-curvature consensus, retaining its selection limitations."""
    rng = np.random.default_rng(20261017)
    left, right = rng.integers(0, len(mean), size=(2, 2048))
    difference = mean[right] - mean[left]
    usable = np.abs(difference) > 0.05
    left, right, difference = left[usable], right[usable], difference[usable]
    second = variance + mean**2
    if not len(left):
        return {"beta": 0.0, "intercept": 0.0, "fraction": 0.0}
    beta = (second[right] - second[left]) / difference
    intercept = second[left] - beta * mean[left]
    errors = np.abs(second[:, None] - mean[:, None] * beta - intercept)
    score = (errors < tolerance).sum(axis=0)
    best = int(np.argmax(score))
    selected = errors[:, best] < tolerance
    coefficients = np.linalg.lstsq(
        np.column_stack([mean[selected], np.ones(selected.sum())]), second[selected], rcond=None
    )[0]
    return {
        "beta": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "fraction": float(selected.mean()),
        "tolerance": tolerance,
    }


def prefix_errors(moments: np.lib.npyio.NpzFile, column: int) -> list[dict[str, float | int]]:
    """Compare descriptive prefix estimates to final moments without treating them as causal targets."""
    rows = []
    for split in ("train", "valid"):
        mean, variance = (
            moments[f"{split}_mean"][:, :, column],
            moments[f"{split}_var"][:, :, column],
        )
        for index, length in enumerate(moments["prefixes"]):
            rows.append(
                {
                    "split": split,
                    "length": int(length),
                    "mean_mae": float(np.abs(mean[:, index] - mean[:, -1]).mean()),
                    "std_mae": float(
                        np.abs(np.sqrt(variance[:, index]) - np.sqrt(variance[:, -1])).mean()
                    ),
                }
            )
    return rows
