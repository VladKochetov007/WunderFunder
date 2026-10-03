from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
FEATURES = (
    [f"i0_p{i}" for i in range(22)]
    + [f"i0_v{i}" for i in range(22)]
    + [f"i0_dp{i}" for i in range(4)]
    + [f"i0_dv{i}" for i in range(4)]
    + [f"i1_p{i}" for i in range(22)]
    + [f"i1_v{i}" for i in range(22)]
    + [f"i1_dp{i}" for i in range(4)]
    + [f"i1_dv{i}" for i in range(4)]
    + [f"a{i}" for i in range(8)]
)
MODELS = ["ridge", "affine", "offset", "gain", "offset_gain", "threshold"]


def sample_ids(total: int, count: int, seed: int) -> np.ndarray:
    """Reproduce the predecessor's row-group sampling convention."""
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False))


def ewma(values: np.ndarray, half_life: float) -> np.ndarray:
    """Causal zero-initialized EWMA, equivalent to predecessor recurrence."""
    alpha = 1 - 0.5 ** (1 / half_life)
    return lfilter([alpha], [1, -(1 - alpha)], values).astype(np.float32)


def causal_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build original 243-column ridge design and five causal state inputs."""
    delta = np.zeros_like(values)
    delta[1:] = values[1:] - values[:-1]
    bid = np.any(delta[:, :11] != 0, axis=1).astype(np.float32)
    ask = np.any(delta[:, 11:22] != 0, axis=1).astype(np.float32)
    change = np.maximum(bid, ask)
    trade = np.any(delta[:, 48:52] != 0, axis=1).astype(np.float32)
    change1 = np.any(delta[:, 52:74] != 0, axis=1).astype(np.float32)
    trade1 = np.any(delta[:, 100:104] != 0, axis=1).astype(np.float32)
    rows = np.arange(len(values), dtype=np.float32)
    ages = []
    for event in [change, trade]:
        last = np.maximum.accumulate(np.where(event > 0, rows, -10001))
        ages.append(np.log1p(rows - last).astype(np.float32))
    magnitude = np.abs(delta[:, :44]).sum(axis=1)
    states = np.column_stack(
        [change, bid, ask, trade, change1, trade1]
        + [ewma(change, h) for h in [10, 100, 1000]]
        + [ewma(trade, h) for h in [10, 100, 1000]]
        + [ewma(change1, 100), ewma(trade1, 100)]
        + ages
        + [ewma(magnitude, 10), ewma(magnitude, 100), rows / 20000]
    )
    small = np.column_stack([values[:, 106], values[:, 107], states[:, 7], states[:, 10], ages[0]])
    return np.column_stack([values, delta, states]), small


def read_sequence(
    parquet: pq.ParquetFile, row_group: int, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Read one row group, recording actual ID and content fingerprints."""
    columns = FEATURES + ["t0", "t1", "seq_ix", "need_prediction"]
    if split == "valid":
        columns += ["is_scored"]
    table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
    values = np.column_stack([table[c].to_numpy() for c in FEATURES])
    targets = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]])
    required = table["need_prediction"].to_numpy()
    mask = required.copy()
    if split == "valid":
        mask &= table["is_scored"].to_numpy()
    digest = hashlib.sha256()
    for array in [values, targets, mask]:
        digest.update(array.tobytes())
    provenance = {
        "row_group": int(row_group),
        "seq_ix": int(table["seq_ix"][0].as_py()),
        "source_split": split,
        "rows": len(values),
        "scored_or_required_rows": int(mask.sum()),
        "selected_columns_sha256": digest.hexdigest(),
        "warmup_inputs_sha256": hashlib.sha256(values[:99].tobytes()).hexdigest(),
    }
    return values, targets, mask, provenance


def calibration_design(predictions: np.ndarray, states: np.ndarray, model: str) -> np.ndarray:
    """Matched designs for center, gain, and additive threshold hypotheses."""
    intercept = np.ones((len(predictions), 1))
    affine = np.column_stack([intercept, predictions])
    if model == "affine":
        return affine
    if model == "offset":
        return np.column_stack([affine, states])
    if model == "gain":
        return np.column_stack([affine, states * predictions[:, None]])
    if model == "offset_gain":
        return np.column_stack([affine, states, states * predictions[:, None]])
    if model == "threshold":
        return np.column_stack([affine, states])
    raise ValueError(f"Unknown calibration model: {model}")


def threshold_states(states: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Two indicators per feature yield three additive bins."""
    return np.column_stack(
        [states[:, j] > thresholds[j, k] for j in range(5) for k in range(2)]
    ).astype(np.float64)


def sufficient_stats(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Six additive moments reproduce clipped global weighted Pearson."""
    target = np.clip(targets.astype(np.float64), -2, 2)
    prediction = np.clip(predictions.astype(np.float64), -2, 2)
    weight = np.abs(target)
    return np.array(
        [
            weight.sum(),
            (weight * target).sum(),
            (weight * prediction).sum(),
            (weight * target**2).sum(),
            (weight * prediction**2).sum(),
            (weight * target * prediction).sum(),
        ]
    )


def weighted_pearson(moments: np.ndarray) -> np.ndarray:
    """Compute weighted correlation from moments, pooling before division."""
    weight, target, prediction, tt, pp, tp = np.moveaxis(moments, -1, 0)
    covariance = tp - target * prediction / weight
    variance = (tt - target**2 / weight) * (pp - prediction**2 / weight)
    return covariance / np.sqrt(np.maximum(variance, 1e-30))
