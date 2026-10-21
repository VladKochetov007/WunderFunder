"""Validate the source-qualified F2 replay inputs without reading labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pyarrow.parquet as pq


class CampaignSplitLike(Protocol):
    """Campaign split fields consumed by source receipts and replay."""

    fold: int
    sha256: str
    evaluation_groups: tuple[int, ...]


@dataclass(frozen=True)
class F2Source:
    """Feature-only row-group reader with exact split and sequence identities."""

    split: CampaignSplitLike
    manifest: dict[str, object]
    parquet: pq.ParquetFile
    data_path: Path
    groups: tuple[int, ...]
    sequence_by_group: dict[int, int]
    feature_columns: tuple[str, ...]
    sequence_length: int
    columns: tuple[str, ...]
    data_identity: dict[str, int | str]


def validate_f2_source(repo_root: Path, split_path: Path, max_groups: int | None) -> F2Source:
    """Validate F2 membership and producer schema, excluding all label fields."""
    from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, repo_root)
    if split.fold != 2:
        raise ValueError(f"this replay requires fold 2, got fold {split.fold}")
    manifest = cast(dict[str, object], json.loads(split_path.read_text(encoding="utf-8")))
    sequences = cast(dict[str, list[dict[str, object]]], manifest["sequences"])
    sequence_by_group = {int(item["row_group"]): int(item["seq_ix"]) for item in sequences["valid"]}
    all_groups = tuple(map(int, split.evaluation_groups))
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive")
    groups = all_groups[:max_groups] if max_groups is not None else all_groups
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("source-qualified F2 groups must be nonempty and unique")

    data_path = repo_root / "datasets" / "valid.parquet"
    data_before = data_path.stat()
    parquet = pq.ParquetFile(data_path)
    if parquet.num_row_groups <= max(groups):
        raise ValueError("F2 row-group ID is outside the producer parquet")
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
        raise ValueError("valid.parquet schema differs from the producer contract")
    columns = ("seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS)
    source_index = cast(dict[str, object], manifest["sources"])
    source = cast(dict[str, object], source_index["datasets/valid.parquet"])
    data_identity: dict[str, int | str] = {
        "bytes": data_before.st_size,
        "device": data_before.st_dev,
        "inode": data_before.st_ino,
        "mtime_ns": data_before.st_mtime_ns,
        "footer_sha256_from_split": str(source["footer_sha256"]),
    }
    for row_group in groups:
        if parquet.metadata.row_group(row_group).num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"F2 row group {row_group} has an unexpected sequence length")
        if row_group not in sequence_by_group:
            raise ValueError(f"F2 row group {row_group} lacks a sequence identity")
    return F2Source(
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
