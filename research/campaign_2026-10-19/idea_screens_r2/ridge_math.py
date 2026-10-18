"""Numerical helpers for the bounded fixed-grid ridge screen."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .views import VIEW_NAMES, build_views

FIT_ROW_STRIDE = 40
FIT_BLOCK_ROWS = 2048
PREDICT_BLOCK_ROWS = 1024
FOLD_SEED = 19_020


@dataclass(frozen=True)
class RidgeModel:
    """Fit-only standardized coefficients for the two clipped targets."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    norms: tuple[float, float]


def component_folds(components: list[int], seed: int = FOLD_SEED) -> list[set[int]]:
    """Assign sorted components to three deterministic, disjoint validation folds."""
    if len(set(components)) != 14:
        raise ValueError("the fixed R2 screen requires exactly 14 fit components")
    order = sorted(set(components))
    random.Random(seed).shuffle(order)
    folds = [set(order[index::3]) for index in range(3)]
    if set.union(*folds) != set(order) or any(
        folds[i] & folds[j] for i in range(3) for j in range(i)
    ):
        raise RuntimeError("component fold assignment is not a partition")
    return folds


def collect_fit_rows(
    sequences: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """Build full causal histories, then retain stride rows from fit sequences."""
    parts = {name: [] for name in VIEW_NAMES}
    targets: list[np.ndarray] = []
    components: list[np.ndarray] = []
    fit_sequence_count = 0
    for sequence in sequences:
        if sequence["role"] != "fit":
            continue
        fit_sequence_count += 1
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        scored = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or scored.shape != (20_000,):
            raise ValueError("R2 requires complete source-pinned 20,000-row sequences")
        views = build_views(x)
        rows = np.arange(0, len(x), FIT_ROW_STRIDE)
        rows = rows[scored[rows]]
        if rows.size == 0 or not np.isfinite(y[rows]).all():
            raise ValueError("fit sequence has no selected stride rows or nonfinite labels")
        for name in VIEW_NAMES:
            parts[name].append(views[name][rows])
        targets.append(y[rows])
        components.append(np.full(rows.size, int(sequence["component"]), dtype=np.int32))
    if fit_sequence_count == 0:
        raise ValueError("sample has no fitting sequences")
    return (
        {name: np.concatenate(value, axis=0) for name, value in parts.items()},
        np.concatenate(targets),
        np.concatenate(components),
        fit_sequence_count,
    )


def fit_candidates(
    features: np.ndarray, targets: np.ndarray, alphas: tuple[int, ...]
) -> dict[int, RidgeModel]:
    """Fit weighted heads with one training-only normalizer and reused Gram matrices."""
    x = features.astype(np.float64)
    mean = x.mean(axis=0)
    scale = np.sqrt(x.var(axis=0))
    scale[scale < 1e-12] = 1.0
    x -= mean
    x /= scale
    clipped = np.clip(targets.astype(np.float64), -2.0, 2.0)
    coefficients = {alpha: np.zeros((x.shape[1], 2), dtype=np.float64) for alpha in alphas}
    intercepts = {alpha: np.zeros(2, dtype=np.float64) for alpha in alphas}
    for target_index in range(2):
        label = clipped[:, target_index]
        weight = np.abs(label)
        total = float(weight.sum())
        if total <= 0.0:
            continue
        x_mean = (weight @ x) / total
        y_mean = float(weight @ label) / total
        gram = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        rhs = np.zeros(x.shape[1], dtype=np.float64)
        for start in range(0, len(x), FIT_BLOCK_ROWS):
            end = min(start + FIT_BLOCK_ROWS, len(x))
            centered = x[start:end] - x_mean
            row_weight = weight[start:end]
            target_centered = label[start:end] - y_mean
            gram += centered.T @ (centered * row_weight[:, None])
            rhs += centered.T @ (row_weight * target_centered)
        for alpha in alphas:
            regularized = gram.copy()
            regularized.flat[:: len(gram) + 1] += alpha
            coefficient = np.linalg.solve(regularized, rhs)
            coefficients[alpha][:, target_index] = coefficient
            intercepts[alpha][target_index] = y_mean - float(x_mean @ coefficient)
    return {
        alpha: RidgeModel(
            mean.copy(),
            scale.copy(),
            coefficients[alpha],
            intercepts[alpha],
            (
                float(np.linalg.norm(coefficients[alpha][:, 0])),
                float(np.linalg.norm(coefficients[alpha][:, 1])),
            ),
        )
        for alpha in alphas
    }


def predict(features: np.ndarray, model: RidgeModel) -> np.ndarray:
    """Predict all rows with bounded float64 analysis matrix blocks."""
    output = np.empty((len(features), 2), dtype=np.float32)
    for start in range(0, len(features), PREDICT_BLOCK_ROWS):
        end = min(start + PREDICT_BLOCK_ROWS, len(features))
        block = features[start:end].astype(np.float64)
        block -= model.mean
        block /= model.scale
        output[start:end] = (block @ model.coefficients + model.intercepts).astype(np.float32)
    return output
