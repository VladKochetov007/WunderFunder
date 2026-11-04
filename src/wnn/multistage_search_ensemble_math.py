from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.gmetric import GlobalAccumulator
from wnn.multistage_search import file_sha256
from wnn.splits import CampaignSplit


def accumulate_quadratic(
    data_root: Path, split: CampaignSplit, prediction_refs: Sequence[Mapping[str, object]]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Accumulate the exact convex weighted-MSE quadratic over inner-control rows."""
    count = len(prediction_refs)
    quadratic = np.zeros((count, count), dtype=np.float64)
    linear = np.zeros(count, dtype=np.float64)
    constant = total_weight = 0.0
    source = SequenceFile(data_root / "datasets/train.parquet")
    entry_maps = [_entries_by_group(item) for item in prediction_refs]
    for group in split.evaluation_groups:
        sequence = source.read(group)
        if sequence.targets is None:
            raise ValueError("inner control labels are unavailable")
        target = np.clip(sequence.targets[WARMUP:], -2.0, 2.0).astype(np.float64)
        prediction_rows = [
            load_prediction(ref, entry_maps[index][group], group)[WARMUP:].astype(np.float64)
            for index, ref in enumerate(prediction_refs)
        ]
        matrix = np.stack(prediction_rows, axis=-1).reshape(-1, count)
        flattened_target = target.reshape(-1)
        weight = np.abs(flattened_target)
        quadratic += matrix.T @ (matrix * weight[:, None])
        linear += matrix.T @ (weight * flattened_target)
        constant += float(np.dot(weight, flattened_target * flattened_target))
        total_weight += float(weight.sum())
    if total_weight <= 0 or not math.isfinite(total_weight):
        raise ValueError("inner control has no positive clipped-target weight")
    return quadratic, linear, constant, total_weight


def score_ensemble(
    data_root: Path,
    split: CampaignSplit,
    prediction_refs: Sequence[Mapping[str, object]],
    weights: np.ndarray,
    *,
    deadline_epoch: float | None = None,
) -> dict[str, object]:
    """Evaluate pooled clipped WP of one fixed simplex vector on the inner control."""
    files = SequenceFile(data_root / "datasets/train.parquet")
    entry_maps = [_entries_by_group(item) for item in prediction_refs]
    accumulator = GlobalAccumulator()
    components = {
        group: component for source, group, component in split.components if source == "train"
    }
    for group in split.evaluation_groups:
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("stage 4 WP trial reached its hard wall deadline")
        sequence = files.read(group)
        if sequence.targets is None:
            raise ValueError("inner control labels are unavailable")
        values = [
            load_prediction(ref, entry_maps[index][group], group)
            for index, ref in enumerate(prediction_refs)
        ]
        mixed = np.einsum("m,tmk->tk", weights, np.stack(values, axis=1))
        mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        mask[WARMUP:] = True
        accumulator.add(sequence.targets[None], mixed[None], mask[None], group=components[group])
    return accumulator.result()


def solve_simplex(quadratic: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Solve the nonnegative unit-sum convex quadratic with deterministic SLSQP."""
    count = len(linear)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ quadratic @ weights - 2 * weights @ linear)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2 * (quadratic @ weights - linear)

    solution = minimize(
        objective,
        np.full(count, 1.0 / count),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda values: float(values.sum() - 1.0),
                "jac": lambda values: np.ones_like(values),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"convex simplex weight search failed: {solution.message}")
    values = np.clip(solution.x, 0.0, 1.0)
    total = float(values.sum())
    if total <= 0:
        raise RuntimeError("simplex solver returned a zero-mass weight vector")
    return values / total


def _entries_by_group(reference: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    entries = reference.get("entries")
    if not isinstance(entries, list):
        raise ValueError("prediction manifest entries are malformed")
    return {int(entry["row_group"]): entry for entry in entries}


def load_prediction(
    reference: Mapping[str, object], entry: Mapping[str, object], group: int
) -> np.ndarray:
    """Load one hashed two-target prediction shard without object deserialization."""
    directory = Path(str(reference["directory"]))
    path = (directory / str(entry["file"])).resolve(strict=True)
    if path.parent != directory.resolve() or file_sha256(path) != entry.get("sha256"):
        raise ValueError(f"candidate prediction shard changed for row group {group}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2):
        raise ValueError("candidate prediction shard has an unexpected shape or dtype")
    if not np.isfinite(values).all():
        raise ValueError("candidate prediction shard contains nonfinite values")
    return values
