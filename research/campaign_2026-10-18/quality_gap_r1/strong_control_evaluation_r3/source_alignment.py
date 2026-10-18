"""Validate source identities and align full-valid control rows to the F0 bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
from pins import ROW_GROUPS, ROWS_PER_GROUP

SOURCE_COLUMNS = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    """Hash a file in bounded chunks and optionally enforce a file-size limit."""
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"file exceeds byte bound: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(path: Path) -> tuple[int, int, int, int]:
    """Return filesystem identity fields used to detect changes during bounded reads."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def sha256_array(array: np.ndarray) -> str:
    """Hash contiguous little-endian bytes, matching the frozen diagnostic input pins."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError("object arrays are not accepted")
    value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def read_pinned_json(path: Path, expected_sha256: str, name: str) -> dict[str, object]:
    """Read a bounded JSON object only after verifying its exact byte hash."""
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the configured JSON byte bound")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the configured JSON byte bound")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, object], value)


def resolve_manifest_path(base: Path, value: object) -> Path:
    """Resolve an absolute or manifest-relative path without guessing the working directory."""
    path = Path(str(value))
    return path.resolve(strict=True) if path.is_absolute() else (base / path).resolve(strict=True)


def prediction_row_map(groups: list[int], expected_group_count: int = ROW_GROUPS) -> dict[int, int]:
    """Build explicit source-row-group to prediction-row mapping from the export sidecar."""
    if len(groups) != expected_group_count or len(set(groups)) != expected_group_count:
        raise ValueError("export sidecar row_groups must be a unique full-valid permutation")
    if set(groups) != set(range(expected_group_count)):
        raise ValueError("export sidecar row_groups do not cover the full validation source")
    return {group: index for index, group in enumerate(groups)}


def require_sha256(actual: str, expected: str, name: str) -> None:
    """Require a caller-supplied arrival hash to match an exact captured input."""
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{name} expected SHA-256 must be 64 lowercase hexadecimal characters")
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs from the arrival pin")


def write_npy_create_only(path: Path, array: np.ndarray) -> None:
    """Write one NumPy array without replacing an existing artifact."""
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def extract_group(
    seq_ix: np.ndarray,
    steps: np.ndarray,
    need_prediction: np.ndarray,
    is_scored: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    predictions: np.ndarray,
    expected_seq_ix: int,
    *,
    warmup: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one source group and select exactly the official need-and-score rows."""
    row_count = len(steps)
    fields = (seq_ix, need_prediction, is_scored, t0, t1)
    if row_count == 0 or any(value.shape != (row_count,) for value in fields):
        raise ValueError("source group fields must be nonempty aligned vectors")
    if seq_ix.dtype.kind not in "iu" or steps.dtype.kind not in "iu":
        raise ValueError("source sequence and step identities must be integer vectors")
    if predictions.shape != (row_count, 2) or predictions.dtype != np.float32:
        raise ValueError("group predictions must be float32 with shape (rows, 2)")
    if not np.array_equal(steps, np.arange(row_count, dtype=steps.dtype)):
        raise ValueError("source step order is not contiguous from zero")
    if not np.all(seq_ix == expected_seq_ix):
        raise ValueError("source sequence identity differs from the checkpoint split")
    if need_prediction.dtype != np.bool_ or is_scored.dtype != np.bool_:
        raise ValueError("need_prediction and is_scored source columns must be boolean")
    mask = need_prediction & is_scored
    if warmup > 0 and mask[:warmup].any():
        raise ValueError("official mask includes a scored warmup row")
    labels = np.column_stack((t0, t1)).astype(np.float32, copy=False)
    selected_labels = labels[mask]
    selected_predictions = np.asarray(predictions[mask], dtype=np.float32)
    positions = steps[mask].astype(np.int64, copy=False)
    if not np.isfinite(selected_labels).all() or not np.isfinite(selected_predictions).all():
        raise ValueError("selected source labels or control predictions contain nonfinite values")
    return selected_labels, selected_predictions, positions, mask


def collect_evaluation_rows(
    parquet: pq.ParquetFile,
    evaluation_groups: list[int],
    identities: dict[int, dict[str, int]],
    prediction: np.ndarray,
    row_map: dict[int, int],
    *,
    rows_per_group: int = ROWS_PER_GROUP,
    warmup: int = 99,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Select original labels and control predictions in the checkpoint split's group order."""
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    row_groups: list[np.ndarray] = []
    components: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    group_audit: list[dict[str, object]] = []
    for group in evaluation_groups:
        identity = identities[group]
        table = parquet.read_row_group(group, columns=SOURCE_COLUMNS, use_threads=False)
        if table.num_rows != rows_per_group:
            raise ValueError(f"validation source row-group size changed: {group}")
        values_by_column: dict[str, np.ndarray] = {}
        for name in SOURCE_COLUMNS:
            column = table[name]
            if column.null_count:
                raise ValueError(f"validation source column contains nulls: {name}")
            values_by_column[name] = np.asarray(column.to_numpy(zero_copy_only=False))
        seq, steps, need, scored, t0, t1 = (values_by_column[name] for name in SOURCE_COLUMNS)
        prediction_index = row_map[group]
        labels, selected, selected_positions, mask = extract_group(
            seq,
            steps,
            need,
            scored,
            t0,
            t1,
            prediction[prediction_index],
            identity["seq_ix"],
            warmup=warmup,
        )
        targets.append(labels)
        predictions.append(selected)
        row_groups.append(np.full(len(labels), group, dtype=np.int64))
        components.append(np.full(len(labels), identity["component"], dtype=np.int64))
        positions.append(selected_positions)
        group_audit.append(
            {
                "row_group": group,
                "prediction_row_index": prediction_index,
                "seq_ix": identity["seq_ix"],
                "component": identity["component"],
                "selected_rows": len(labels),
                "mask_sha256": sha256_array(mask.astype(np.uint8)),
                "selected_positions_sha256": sha256_array(selected_positions),
            }
        )
    arrays = {
        "target": np.concatenate(targets).astype(np.float32, copy=False),
        "prediction": np.concatenate(predictions).astype(np.float32, copy=False),
        "row_group": np.concatenate(row_groups),
        "component": np.concatenate(components),
        "position": np.concatenate(positions),
    }
    return arrays, group_audit
