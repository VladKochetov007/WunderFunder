"""Hash, schema, chronology, target and mutation checks for official sources."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wnn.chronological_source_audit_io import assert_stat_unchanged, file_sha256_with_progress
from wnn.chronological_source_index import SourcePin, SourceRowGroupReadError
from wnn.chronological_splits import _footer_identity
from wnn.data import FEATURE_COLUMNS, N_FEATURES


class SourceIdentity(TypedDict):
    """The exact bounded source entry consumed by the thin-index producer."""

    bytes: int
    sha256: str
    mtime_ns: int
    footer_bytes: int
    footer_sha256: str
    row_group_count: int
    sequence_length: int
    has_is_scored: bool


ProgressCallback = Callable[[dict[str, int | str]], None]


def audit_source(
    path: Path,
    split: str,
    pin: SourcePin,
    sequence_length: int,
    progress_interval: int,
    report: ProgressCallback,
) -> tuple[SourceIdentity, dict[str, int]]:
    """Verify one source file and stream its thin sequence/target audit."""
    if pin.split != split:
        raise ValueError(f"source pin split {pin.split!r} does not match {split!r}")
    before = path.stat()
    if before.st_size != pin.byte_count:
        raise ValueError(
            f"source byte count mismatch for {path}: observed {before.st_size}, expected {pin.byte_count}"
        )

    report({"phase": "hashing_sha256", "bytes_hashed": 0, "bytes_total": before.st_size})

    def hash_progress(byte_count: int) -> None:
        report({"phase": "hashing_sha256", "bytes_hashed": byte_count})

    actual_sha256, bytes_hashed = file_sha256_with_progress(path, hash_progress)
    assert_stat_unchanged(path, before, "full-file SHA-256")
    if bytes_hashed != before.st_size:
        raise ValueError(
            f"full-file SHA-256 read {bytes_hashed} bytes from {path}; expected {before.st_size}"
        )
    report(
        {
            "phase": "sha256_computed",
            "bytes_hashed": bytes_hashed,
            "observed_bytes": before.st_size,
            "observed_sha256": actual_sha256,
            "mtime_ns": before.st_mtime_ns,
        }
    )
    if actual_sha256 != pin.sha256:
        raise ValueError(
            f"source SHA-256 mismatch for {path}: observed {actual_sha256}, expected {pin.sha256}"
        )

    footer_before = _footer_identity(path)
    assert_stat_unchanged(path, before, "footer identity read")
    if footer_before["bytes"] != before.st_size or footer_before["mtime_ns"] != before.st_mtime_ns:
        raise ValueError(f"source changed while its footer identity was read: {path}")

    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(
            f"cannot open parquet metadata for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _validate_schema(parquet.schema_arrow, path, pin)
    if parquet.num_row_groups != pin.row_group_count:
        raise ValueError(
            f"row-group count mismatch for {path}: observed {parquet.num_row_groups}, "
            f"expected {pin.row_group_count}"
        )
    for row_group in range(parquet.num_row_groups):
        try:
            rows = parquet.metadata.row_group(row_group).num_rows
        except Exception as exc:
            raise SourceRowGroupReadError(
                split, row_group, str(path), f"metadata error: {type(exc).__name__}: {exc}"
            ) from exc
        if rows != sequence_length:
            raise SourceRowGroupReadError(
                split,
                row_group,
                str(path),
                f"metadata row count is {rows}; expected {sequence_length}",
            )
    if parquet.metadata.num_rows != pin.row_group_count * sequence_length:
        raise ValueError(f"total row count mismatch for {path}")

    assert_stat_unchanged(path, before, "parquet metadata read")
    has_mask = "is_scored" in parquet.schema_arrow.names
    seen_ids: dict[int, int] = {}
    scored_rows = 0
    rows_scanned = 0
    report(
        {
            "phase": "scanning_sequence_ids_steps_targets",
            "row_groups_scanned": 0,
            "row_groups_expected": parquet.num_row_groups,
        }
    )
    for row_group in range(parquet.num_row_groups):
        try:
            scored_rows += _audit_row_group(
                parquet, path, split, row_group, sequence_length, has_mask, seen_ids
            )
        except SourceRowGroupReadError:
            raise
        except Exception as exc:
            raise SourceRowGroupReadError(
                split, row_group, str(path), f"{type(exc).__name__}: {exc}"
            ) from exc
        rows_scanned += sequence_length
        if (row_group + 1) % progress_interval == 0 or row_group + 1 == parquet.num_row_groups:
            report(
                {
                    "phase": "scanning_sequence_ids_steps_targets",
                    "row_groups_scanned": row_group + 1,
                    "rows_scanned": rows_scanned,
                    "unique_sequence_ids": len(seen_ids),
                    "scored_rows": scored_rows,
                }
            )

    assert_stat_unchanged(path, before, "label and sequence-identity scan")
    footer_after = _footer_identity(path)
    assert_stat_unchanged(path, before, "final footer identity read")
    if footer_after != footer_before:
        raise ValueError(f"source footer changed during audit: {path}")

    identity: SourceIdentity = {
        "bytes": before.st_size,
        "sha256": actual_sha256,
        "mtime_ns": before.st_mtime_ns,
        "footer_bytes": int(footer_before["footer_bytes"]),
        "footer_sha256": str(footer_before["footer_sha256"]),
        "row_group_count": parquet.num_row_groups,
        "sequence_length": sequence_length,
        "has_is_scored": has_mask,
    }
    summary = {
        "rows_scanned": rows_scanned,
        "unique_sequence_ids": len(seen_ids),
        "scored_rows": scored_rows,
    }
    return identity, summary


def _audit_row_group(
    parquet: pq.ParquetFile,
    path: Path,
    split: str,
    row_group: int,
    sequence_length: int,
    has_mask: bool,
    seen_ids: dict[int, int],
) -> int:
    """Read one small identity/target group and validate its complete sequence order."""
    table = _read_row_group(parquet, path, split, row_group, has_mask)
    if table.num_rows != sequence_length:
        raise SourceRowGroupReadError(
            split,
            row_group,
            str(path),
            f"read row count is {table.num_rows}; expected {sequence_length}",
        )
    for name in ("seq_ix", "step_in_seq", "t0", "t1"):
        if table.column(name).null_count:
            raise SourceRowGroupReadError(
                split, row_group, str(path), f"{name} contains null values"
            )
    if has_mask and table.column("is_scored").null_count:
        raise SourceRowGroupReadError(split, row_group, str(path), "is_scored contains null values")

    seq_values = table.column("seq_ix").to_numpy(zero_copy_only=False)
    if seq_values.ndim != 1 or len(seq_values) != sequence_length:
        raise SourceRowGroupReadError(split, row_group, str(path), "seq_ix has an invalid shape")
    if not np.all(seq_values == seq_values[0]):
        raise SourceRowGroupReadError(
            split, row_group, str(path), "seq_ix is not constant within the complete sequence"
        )
    sequence_id = int(seq_values[0])
    if sequence_id in seen_ids:
        raise SourceRowGroupReadError(
            split,
            row_group,
            str(path),
            f"duplicate seq_ix {sequence_id}; first appeared in row_group={seen_ids[sequence_id]}",
        )
    seen_ids[sequence_id] = row_group

    steps = table.column("step_in_seq").to_numpy(zero_copy_only=False)
    if steps.shape != (sequence_length,):
        raise SourceRowGroupReadError(
            split, row_group, str(path), "step_in_seq has an invalid shape"
        )
    expected_steps = np.arange(sequence_length, dtype=np.int64)
    observed_steps = steps.astype(np.int64, copy=False)
    if not np.array_equal(observed_steps, expected_steps):
        mismatch = int(np.flatnonzero(observed_steps != expected_steps)[0])
        raise SourceRowGroupReadError(
            split,
            row_group,
            str(path),
            f"step_in_seq mismatch at row={mismatch}: observed {steps[mismatch]}, "
            f"expected {expected_steps[mismatch]}",
        )

    for name in ("t0", "t1"):
        values = table.column(name).to_numpy(zero_copy_only=False)
        finite = np.isfinite(values)
        if not finite.all():
            bad_row = int(np.argmax(~finite))
            raise SourceRowGroupReadError(
                split, row_group, str(path), f"nonfinite target {name} at row={bad_row}"
            )
    if has_mask:
        mask = table.column("is_scored").to_numpy(zero_copy_only=False)
        if mask.dtype.kind != "b":
            raise SourceRowGroupReadError(split, row_group, str(path), "is_scored is not boolean")
        return int(np.count_nonzero(mask))
    return 0


def _read_row_group(
    parquet: pq.ParquetFile, path: Path, split: str, row_group: int, has_mask: bool
) -> pa.Table:
    """Read only target and sequence-order columns from one source-local row group."""
    columns = ["seq_ix", "step_in_seq", "t0", "t1"]
    if has_mask:
        columns.append("is_scored")
    try:
        return parquet.read_row_group(row_group, columns=columns, use_threads=False)
    except Exception as exc:
        raise SourceRowGroupReadError(
            split, row_group, str(path), f"{type(exc).__name__}: {exc}"
        ) from exc


def _validate_schema(schema: pa.Schema, path: Path, pin: SourcePin) -> None:
    """Require the official feature, sequence, label, and scoring-mask schema."""
    names = schema.names
    if len(names) != len(set(names)):
        raise ValueError(f"parquet schema has duplicate column names: {path}")
    if len(FEATURE_COLUMNS) != N_FEATURES:
        raise RuntimeError("canonical feature schema no longer contains 112 columns")
    required = {"seq_ix", "step_in_seq", "t0", "t1", *FEATURE_COLUMNS}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"source schema is missing required columns for {path}: {missing}")
    has_mask = "is_scored" in names
    if has_mask != pin.has_scoring_mask:
        raise ValueError(f"source scoring-mask presence differs from the official pin: {path}")
    if not pa.types.is_integer(schema.field("seq_ix").type):
        raise ValueError(f"seq_ix must have an integer schema: {path}")
    if not pa.types.is_signed_integer(schema.field("step_in_seq").type):
        raise ValueError(f"step_in_seq must have a signed integer schema: {path}")
    for name in FEATURE_COLUMNS + ["t0", "t1"]:
        dtype = schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f"{name} must have a numeric schema: {path}")
    if has_mask and not pa.types.is_boolean(schema.field("is_scored").type):
        raise ValueError(f"is_scored must have a boolean schema: {path}")
