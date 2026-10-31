"""Re-score ordered primary-valid predictions from labels and the official row mask.

This checker reads only t0/t1, is_scored, seq_ix, and step_in_seq from the
validation parquet. It preserves physical row-group order; seq_ix is never sorted.
The prediction array must contain one float32 (t0, t1) pair per selected scored
row in that same order. No model or GPU is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
CLIP = 2.0
EPSILON = 1e-8


# hashlib exposes a shared update API but no public digest-object protocol.
def _update_hash(digest: Any, values: np.ndarray, dtype: str) -> None:
    """Hash an array using a fixed byte order and contiguous row-major bytes."""
    digest.update(np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes())


def _sha256_file(path: Path) -> str:
    """Hash one report file without reading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_valid_targets(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read selected targets/masks in physical parquet row-group order."""
    parquet = pq.ParquetFile(path)
    required = {"seq_ix", "step_in_seq", "is_scored", "t0", "t1"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"validation parquet is missing columns: {sorted(missing)}")
    if parquet.metadata.num_rows != parquet.num_row_groups * SEQUENCE_LENGTH:
        raise ValueError("validation parquet does not contain complete fixed-length sequences")

    selected_targets: list[np.ndarray] = []
    is_scored_hash = hashlib.sha256()
    official_mask_hash = hashlib.sha256()
    sequence_order_hash = hashlib.sha256()
    flat_index_hash = hashlib.sha256()
    row_group_step_hash = hashlib.sha256()
    sequence_step_hash = hashlib.sha256()
    seen_sequence_ids: set[int] = set()
    selected_rows = 0
    is_scored_rows = 0
    total_rows = 0

    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            row_group,
            columns=["seq_ix", "step_in_seq", "is_scored", "t0", "t1"],
            use_threads=False,
        )
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {row_group} is not one complete sequence")
        seq_ix = table.column("seq_ix").to_numpy(zero_copy_only=False)
        if seq_ix.shape != (SEQUENCE_LENGTH,) or not np.all(seq_ix == seq_ix[0]):
            raise ValueError(f"seq_ix is not constant in row group {row_group}")
        sequence_id = int(seq_ix[0])
        if sequence_id in seen_sequence_ids:
            raise ValueError(f"duplicate seq_ix {sequence_id} in validation parquet")
        seen_sequence_ids.add(sequence_id)
        _update_hash(sequence_order_hash, np.asarray([sequence_id]), "<i8")

        steps = table.column("step_in_seq").to_numpy(zero_copy_only=False)
        expected_steps = np.arange(SEQUENCE_LENGTH, dtype=np.int64)
        if steps.shape != (SEQUENCE_LENGTH,) or not np.array_equal(
            steps.astype(np.int64, copy=False), expected_steps
        ):
            raise ValueError(f"step_in_seq is not zero-based row order in row group {row_group}")
        scored = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool)
        if scored.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"is_scored has an invalid shape in row group {row_group}")
        official_mask = scored.copy()
        official_mask[:WARMUP] = False
        _update_hash(is_scored_hash, scored, "u1")
        _update_hash(official_mask_hash, official_mask, "u1")
        is_scored_rows += int(scored.sum())

        if official_mask.any():
            targets = np.column_stack(
                (
                    table.column("t0").to_numpy(zero_copy_only=False)[official_mask],
                    table.column("t1").to_numpy(zero_copy_only=False)[official_mask],
                )
            ).astype(np.float32, copy=False)
            selected_targets.append(targets)
            chosen_steps = expected_steps[official_mask]
            count = int(official_mask.sum())
            flat_rows = row_group * SEQUENCE_LENGTH + chosen_steps
            row_group_steps = np.column_stack(
                (np.full(count, row_group, dtype=np.int64), chosen_steps)
            )
            sequence_steps = np.column_stack(
                (np.full(count, sequence_id, dtype=np.int64), chosen_steps)
            )
            _update_hash(flat_index_hash, flat_rows, "<i8")
            _update_hash(row_group_step_hash, row_group_steps, "<i8")
            _update_hash(sequence_step_hash, sequence_steps, "<i8")
            selected_rows += count
        total_rows += table.num_rows

    if not selected_targets:
        raise ValueError("validation mask selected no rows")
    return np.concatenate(selected_targets, axis=0), {
        "sequence_count": parquet.num_row_groups,
        "total_rows": total_rows,
        "is_scored_rows_before_warmup": is_scored_rows,
        "scored_rows_after_warmup": selected_rows,
        "is_scored_mask_sha256": is_scored_hash.hexdigest(),
        "official_mask_sha256": official_mask_hash.hexdigest(),
        "sequence_order_sha256": sequence_order_hash.hexdigest(),
        "flat_row_index_sha256": flat_index_hash.hexdigest(),
        "row_group_step_index_sha256": row_group_step_hash.hexdigest(),
        "sequence_step_index_sha256": sequence_step_hash.hexdigest(),
        "mask_encoding": "one uint8 value per source row, parquet row-group order",
        "official_mask": "is_scored AND step_in_seq>=99",
        "index_encoding": "little-endian int64; ordered keys are named by hash field",
    }


def _weighted_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute stable pooled clipped weighted Pearson for one target column."""
    y = np.clip(np.asarray(target, dtype=np.float32), -CLIP, CLIP).astype(np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float32), -CLIP, CLIP).astype(np.float64)
    weights = np.abs(y)
    positive = weights > 0
    if not positive.any():
        return 0.0
    y, p, weights = y[positive], p[positive], weights[positive]
    total = float(weights.sum(dtype=np.float64))
    if total < EPSILON:
        return 0.0
    shifted_y = y - y[0]
    shifted_p = p - p[0]
    centered_y = shifted_y - float(np.dot(weights, shifted_y) / total)
    centered_p = shifted_p - float(np.dot(weights, shifted_p) / total)
    variance_y = float(np.dot(weights, centered_y * centered_y) / total)
    variance_p = float(np.dot(weights, centered_p * centered_p) / total)
    if math.sqrt(max(variance_y, 0.0)) <= EPSILON or math.sqrt(max(variance_p, 0.0)) <= EPSILON:
        return 0.0
    covariance = float(np.dot(weights, centered_y * centered_p) / total)
    return float(np.clip(covariance / math.sqrt(variance_y * variance_p), -1.0, 1.0))


def _score(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Recompute pooled clipped WP and raw-prediction clipped-target WMSE."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape:
        raise ValueError("targets and predictions must both have shape (scored_rows, 2)")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("selected targets or predictions contain nonfinite values")

    wp_t0 = _weighted_pearson(y[:, 0], p[:, 0])
    wp_t1 = _weighted_pearson(y[:, 1], p[:, 1])
    clipped_targets = np.clip(y.astype(np.float64), -CLIP, CLIP)
    weights = np.abs(clipped_targets)
    squared_error = np.square(p.astype(np.float64) - clipped_targets)
    numerator = np.sum(weights * squared_error, axis=0, dtype=np.float64)
    denominator = np.sum(weights, axis=0, dtype=np.float64)
    if denominator.sum() <= 0:
        raise ValueError("selected clipped targets have no positive scoring weight")
    wmse_by_target = numerator / np.where(denominator > 0, denominator, 1.0)
    return {
        "wp_t0": wp_t0,
        "wp_t1": wp_t1,
        "weighted_pearson": (wp_t0 + wp_t1) / 2.0,
        "wmse_t0": float(wmse_by_target[0]),
        "wmse_t1": float(wmse_by_target[1]),
        "wmse": float(numerator.sum() / denominator.sum()),
        "prediction_fraction_abs_gt_2": float((np.abs(p) > CLIP).mean()),
    }


def _parse_args() -> argparse.Namespace:
    """Parse the minimal report paths and optional expected values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--predictions-npy", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int)
    parser.add_argument("--expected-scored-rows", type=int)
    parser.add_argument("--expected-is-scored-mask-sha256")
    parser.add_argument("--expected-official-mask-sha256")
    parser.add_argument("--expected-sequence-order-sha256")
    parser.add_argument("--expected-sequence-step-index-sha256")
    parser.add_argument("--expected-prediction-sha256")
    parser.add_argument("--expected-weighted-pearson", type=float)
    parser.add_argument("--expected-wmse", type=float)
    parser.add_argument("--score-absolute-tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> int:
    """Validate report alignment and print independent counts and pooled scores."""
    args = _parse_args()
    targets, source = _load_valid_targets(args.valid_parquet)
    predictions = np.load(args.predictions_npy, mmap_mode="r", allow_pickle=False)
    if predictions.dtype != np.float32 or predictions.shape != targets.shape:
        raise ValueError(
            f"prediction array must be float32 with shape {targets.shape}; "
            f"received {predictions.dtype} {predictions.shape}"
        )
    scores = _score(targets, predictions)
    prediction_sha256 = _sha256_file(args.predictions_npy)
    checks = {
        "prediction_rows_match_mask": predictions.shape[0] == source["scored_rows_after_warmup"],
        "expected_sequence_count": args.expected_sequences is None
        or source["sequence_count"] == args.expected_sequences,
        "expected_scored_row_count": args.expected_scored_rows is None
        or source["scored_rows_after_warmup"] == args.expected_scored_rows,
        "expected_is_scored_mask_hash": args.expected_is_scored_mask_sha256 is None
        or source["is_scored_mask_sha256"] == args.expected_is_scored_mask_sha256,
        "expected_official_mask_hash": args.expected_official_mask_sha256 is None
        or source["official_mask_sha256"] == args.expected_official_mask_sha256,
        "expected_sequence_order_hash": args.expected_sequence_order_sha256 is None
        or source["sequence_order_sha256"] == args.expected_sequence_order_sha256,
        "expected_sequence_step_index_hash": args.expected_sequence_step_index_sha256 is None
        or source["sequence_step_index_sha256"] == args.expected_sequence_step_index_sha256,
        "expected_prediction_hash": args.expected_prediction_sha256 is None
        or prediction_sha256 == args.expected_prediction_sha256,
        "expected_weighted_pearson": args.expected_weighted_pearson is None
        or math.isclose(
            scores["weighted_pearson"],
            args.expected_weighted_pearson,
            rel_tol=0.0,
            abs_tol=args.score_absolute_tolerance,
        ),
        "expected_wmse": args.expected_wmse is None
        or math.isclose(
            scores["wmse"], args.expected_wmse, rel_tol=0.0, abs_tol=args.score_absolute_tolerance
        ),
    }
    result = {
        "schema": "wnn.independent-primary-report-score.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "valid_parquet": str(args.valid_parquet),
        "prediction_array": str(args.predictions_npy),
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": str(predictions.dtype),
        "prediction_sha256": prediction_sha256,
        "source_order": "parquet row-group order, then zero-based step_in_seq; no seq_ix sorting",
        "counts": source,
        "scores": scores,
        "checks": checks,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
