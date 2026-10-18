"""Validate the official source split and producer schema for TL3 replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class F0Source:
    """Validated F0 row-group order and feature-only Parquet reader."""

    split: Any
    manifest: dict[str, Any]
    parquet: pq.ParquetFile
    data_path: Path
    groups: tuple[int, ...]
    sequence_by_group: dict[int, int]
    feature_columns: tuple[str, ...]
    sequence_length: int
    columns: tuple[str, ...]
    data_identity: dict[str, int | str]


def validate_f0_source(
    repo_root: Path,
    split_path: Path,
    max_groups: int | None,
) -> F0Source:
    """Load source-qualified fold zero and verify its producer metadata."""
    from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, repo_root)
    if split.fold != 0:
        raise ValueError(f"this replay requires fold 0, got fold {split.fold}")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    sequence_by_group = {
        int(item["row_group"]): int(item["seq_ix"]) for item in manifest["sequences"]["valid"]
    }
    all_groups = tuple(map(int, split.evaluation_groups))
    groups = all_groups[:max_groups] if max_groups is not None else all_groups
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("source-qualified F0 groups must be nonempty and unique")

    data_path = repo_root / "datasets" / "valid.parquet"
    data_before = data_path.stat()
    parquet = pq.ParquetFile(data_path)
    if parquet.num_row_groups <= max(groups):
        raise ValueError("F0 row-group ID is outside the producer parquet")
    expected_schema = [
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        *FEATURE_COLUMNS,
        "t0",
        "t1",
    ]
    if parquet.schema_arrow.names != expected_schema:
        raise ValueError("valid.parquet schema differs from the official producer contract")
    columns = ("seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS)
    source = manifest["sources"]["datasets/valid.parquet"]
    data_identity: dict[str, int | str] = {
        "bytes": data_before.st_size,
        "device": data_before.st_dev,
        "inode": data_before.st_ino,
        "mtime_ns": data_before.st_mtime_ns,
        "footer_sha256_from_split": str(source["footer_sha256"]),
    }
    return F0Source(
        split=split,
        manifest=manifest,
        parquet=parquet,
        data_path=data_path,
        groups=groups,
        sequence_by_group=sequence_by_group,
        feature_columns=tuple(FEATURE_COLUMNS),
        sequence_length=SEQUENCE_LENGTH,
        columns=columns,
        data_identity=data_identity,
    )
