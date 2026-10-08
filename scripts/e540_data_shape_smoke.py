#!/usr/bin/env python3
"""Inspect one feature-only E540 parquet row group and persist phase receipts."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path


def _write_receipt(path: Path, phase: str, started: float, **details: object) -> None:
    """Append a flushed JSONL receipt so a resource kill preserves prior phases."""
    record = {
        "experiment_id": "E540",
        "phase": phase,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        **details,
    }
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(encoded + "\n")
        destination.flush()
    print(encoded, flush=True)


def inspect_row_group(parquet_path: Path, row_group: int, receipt: Path) -> None:
    """Read raw features only; never request target or score-mask columns."""
    if row_group < 0:
        raise ValueError("row_group must be nonnegative")
    parquet_path = parquet_path.resolve(strict=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite smoke receipt: {receipt}")
    started = time.perf_counter()
    stat = parquet_path.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    _write_receipt(
        receipt,
        "started",
        started,
        data_path=str(parquet_path),
        data_identity=identity,
        row_group=row_group,
        requested_columns="seq_ix plus the repository's 112 raw feature columns only",
        targets_requested=False,
        scored_mask_requested=False,
    )

    import numpy as np

    _write_receipt(receipt, "numpy_imported", started)
    from wnn.data import N_FEATURES, SEQUENCE_LENGTH, SequenceFile

    _write_receipt(receipt, "project_reader_imported", started)
    from wnn.features import N_EVENT_FEATURES, event_features

    _write_receipt(receipt, "causal_feature_builder_imported", started)
    _write_receipt(receipt, "metadata_open_started", started)
    source = SequenceFile(parquet_path)
    metadata = source._pf.metadata
    _write_receipt(
        receipt,
        "metadata_opened",
        started,
        row_group_count=metadata.num_row_groups,
        column_count=metadata.num_columns,
    )
    if row_group >= len(source):
        raise IndexError("requested row_group is outside the parquet source")

    _write_receipt(receipt, "feature_read_started", started)
    sequence = source.read(row_group, targets=False, mask=False)
    if sequence.targets is not None or sequence.scored is not None:
        raise AssertionError("feature-only reader unexpectedly returned targets or scored mask")
    raw = sequence.features
    if raw.shape != (SEQUENCE_LENGTH, N_FEATURES):
        raise AssertionError(f"raw feature shape differs: {raw.shape}")
    finite = bool(np.isfinite(raw).all())
    if not finite:
        raise ValueError("supplied raw feature row group contains nonfinite values")
    _write_receipt(
        receipt,
        "raw_feature_read_complete",
        started,
        seq_ix=sequence.seq_ix,
        row_group_rows=int(raw.shape[0]),
        raw_feature_width=int(raw.shape[1]),
        raw_dtype=str(raw.dtype),
        all_raw_features_finite=finite,
        targets_loaded=sequence.targets is not None,
        scored_mask_loaded=sequence.scored is not None,
        feature_bytes=int(raw.nbytes),
    )
    _write_receipt(receipt, "event_feature_build_started", started)
    event = event_features(raw)
    combined = np.concatenate((raw, event), axis=-1)
    if event.shape != (SEQUENCE_LENGTH, N_EVENT_FEATURES):
        raise AssertionError(f"event feature shape differs: {event.shape}")
    if combined.shape != (SEQUENCE_LENGTH, N_FEATURES + N_EVENT_FEATURES):
        raise AssertionError(f"combined input shape differs: {combined.shape}")
    if not np.isfinite(event).all():
        raise ValueError("derived causal event features contain nonfinite values")
    _write_receipt(
        receipt,
        "combined_shape_complete",
        started,
        raw_feature_width=int(raw.shape[1]),
        event_feature_width=int(event.shape[1]),
        combined_input_width=int(combined.shape[1]),
        combined_rows=int(combined.shape[0]),
        event_features_finite=bool(np.isfinite(event).all()),
        target_columns_used=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    inspect_row_group(args.train_parquet, args.row_group, args.receipt)


if __name__ == "__main__":
    main()
