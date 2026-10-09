from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum

import numpy as np


def component_sums(
    prediction: np.ndarray,
    batch: Mapping[str, np.ndarray],
    ema_prediction: np.ndarray,
) -> dict[str, float]:
    """Compute combo-loss numerators and denominators independently in NumPy.

    Args:
        prediction: Model outputs with shape ``[batch, time, 23]``.
        batch: Synthetic or held-out supervision arrays with the frozen loss keys.
        ema_prediction: Fixed incoming EMA targets with shape ``[batch, time, 2]``.

    Returns:
        The raw component numerators and their exact normalization masses.

    Raises:
        ValueError: If output or supervision shapes do not match the frozen objective.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != 23:
        raise ValueError("prediction must have shape [batch, time, 23]")
    expected_shape = prediction.shape[:2]
    if ema_prediction.shape != (*expected_shape, 2):
        raise ValueError("EMA targets must have shape [batch, time, 2]")

    labels = np.clip(np.asarray(batch["labels"], dtype=np.float64), -2.0, 2.0)
    row_weights = np.asarray(batch["main_weights"], dtype=np.float64)
    fixed_target = np.clip(np.asarray(batch["fixed_target"], dtype=np.float64), -2.0, 2.0)
    structural_labels = np.asarray(batch["structural_labels"], dtype=np.int64)
    next_weights = np.asarray(batch["next_weights"], dtype=np.float64)
    next_rows = np.asarray(batch["next_rows"], dtype=np.float64)
    next_up = np.asarray(batch["next_up"], dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    ema64 = np.asarray(ema_prediction, dtype=np.float64)
    if labels.shape != (*expected_shape, 2) or fixed_target.shape != labels.shape:
        raise ValueError("labels and fixed targets must have shape [batch, time, 2]")
    if structural_labels.shape != (*expected_shape, 3):
        raise ValueError("structural labels must have shape [batch, time, 3]")
    if any(
        value.shape != expected_shape for value in (row_weights, next_weights, next_rows, next_up)
    ):
        raise ValueError("row-valued supervision must match prediction batch/time dimensions")

    primary_weights = np.abs(labels) * row_weights[..., None]
    primary_num = np.sum(
        primary_weights * (prediction64[..., :2] - fixed_target) ** 2,
        dtype=np.float64,
    )
    consistency_num = np.sum(
        row_weights[..., None] * (prediction64[..., :2] - ema64) ** 2,
        dtype=np.float64,
    )

    structural_num = 0.0
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        logits = prediction64[..., start : start + width]
        max_logits = np.max(logits, axis=-1, keepdims=True)
        log_normalizer = max_logits[..., 0] + np.log(np.sum(np.exp(logits - max_logits), axis=-1))
        class_indices = structural_labels[..., target_index]
        selected_logits = np.take_along_axis(logits, class_indices[..., None], axis=-1)[..., 0]
        structural_num += float(np.sum((log_normalizer - selected_logits) * row_weights))

    next_distance = (next_rows - 3.0) / 2.0
    next_mse = (prediction64[..., 21] - next_distance) ** 2
    next_logit = prediction64[..., 22]
    next_bce = np.logaddexp(0.0, next_logit) - next_up * next_logit
    next_num = np.sum((next_mse + next_bce) * next_weights, dtype=np.float64)
    return {
        "primary_num": float(primary_num),
        "primary_den": float(np.sum(primary_weights, dtype=np.float64)),
        "consistency_num": float(consistency_num),
        "row_den": float(np.sum(row_weights, dtype=np.float64)),
        "structural_num": structural_num,
        "next_num": float(next_num),
        "next_den": float(np.sum(next_weights, dtype=np.float64)),
    }


def loss_from_sums(sums: Mapping[str, float]) -> float:
    """Combine raw sums using the frozen four-term combo objective.

    Args:
        sums: Raw numerators and denominators from :func:`component_sums`.

    Returns:
        The weighted combo objective value.
    """
    primary = sums["primary_num"] / max(sums["primary_den"], 1e-8)
    consistency = sums["consistency_num"] / max(sums["row_den"], 1e-8)
    structural = sums["structural_num"] / max(sums["row_den"], 1e-8)
    next_aux = sums["next_num"] / max(sums["next_den"], 1e-8)
    return primary + consistency + 0.3 * structural + 0.3 * next_aux


def aggregate_shards(shards: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Sum component numerators and denominators before global normalization.

    Args:
        shards: One raw component-sum mapping per data shard.

    Returns:
        A single mapping representing the concatenated global batch.

    Raises:
        ValueError: If no shard values are supplied.
    """
    if not shards:
        raise ValueError("at least one shard is required")
    keys = tuple(shards[0])
    if any(tuple(shard) != keys for shard in shards[1:]):
        raise ValueError("all shard mappings must have the same ordered keys")
    return {key: fsum(shard[key] for shard in shards) for key in keys}
