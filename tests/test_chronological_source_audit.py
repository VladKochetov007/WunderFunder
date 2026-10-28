"""Small synthetic checks for the full-source identity producer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import wnn.chronological_source_audit as source_audit
import wnn.chronological_source_audit_scan as source_audit_scan
import wnn.chronological_source_index as source_index
from wnn.chronological_source_index import SOURCE_AUDIT_FIELDS, SourcePin, SourceRowGroupReadError
from wnn.data import FEATURE_COLUMNS


def _write_source(
    root: Path,
    split: str,
    *,
    sequence_length: int = 8,
    groups: int = 2,
    nonfinite_target: tuple[int, str, int] | None = None,
    duplicate_sequence_id: bool = False,
    incorrect_step: tuple[int, int, int] | None = None,
) -> SourcePin:
    """Write a tiny typed source fixture and return its exact test-only pin."""
    rows = sequence_length * groups
    columns: dict[str, np.ndarray] = {
        name: np.zeros(rows, dtype=np.float32) for name in FEATURE_COLUMNS
    }
    sequence_ids = np.arange(groups, dtype=np.int64) + (100 if split == "valid" else 0)
    if duplicate_sequence_id and groups > 1:
        sequence_ids[1] = sequence_ids[0]
    columns["seq_ix"] = np.repeat(sequence_ids, sequence_length)
    columns["step_in_seq"] = np.tile(np.arange(sequence_length, dtype=np.int32), groups)
    if incorrect_step is not None:
        group, row, value = incorrect_step
        columns["step_in_seq"][group * sequence_length + row] = value
    columns["t0"] = np.zeros(rows, dtype=np.float32)
    columns["t1"] = np.zeros(rows, dtype=np.float32)
    if nonfinite_target is not None:
        group, target_name, row = nonfinite_target
        columns[target_name][group * sequence_length + row] = np.nan
    has_mask = split == "valid"
    if has_mask:
        columns["is_scored"] = np.arange(rows) % 2 == 0

    path = root / "datasets" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=sequence_length)
    stat = path.stat()
    return SourcePin(
        split=split,
        byte_count=stat.st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_group_count=groups,
        has_scoring_mask=has_mask,
    )


def _write_fixture(root: Path, *, nonfinite_train_target: bool = False) -> dict[str, SourcePin]:
    """Write matching train/valid parquet fixtures and return exact local test pins."""
    (root / "cache").mkdir(parents=True, exist_ok=True)
    train_pin = _write_source(
        root,
        "train",
        nonfinite_target=(1, "t1", 3) if nonfinite_train_target else None,
    )
    valid_pin = _write_source(root, "valid")
    return {"train": train_pin, "valid": valid_pin}


def test_audit_publishes_exact_root_receipt_after_sequence_checks(tmp_path: Path) -> None:
    """A valid tiny pair passes with exact v1 source-entry fields and counts."""
    pins = _write_fixture(tmp_path)
    output = tmp_path / "cache" / "source_audit.json"

    receipt = source_audit._audit_sources(
        tmp_path, output, pins=pins, sequence_length=8, progress_interval=1
    )

    assert set(receipt) == {"schema", "sources"}
    assert receipt["schema"] == "wnn.source-audit.v1"
    assert set(receipt["sources"]) == {"datasets/train.parquet", "datasets/valid.parquet"}
    assert all(set(identity) == SOURCE_AUDIT_FIELDS for identity in receipt["sources"].values())
    assert receipt["sources"]["datasets/train.parquet"]["sha256"] == pins["train"].sha256
    assert receipt["sources"]["datasets/valid.parquet"]["has_is_scored"] is True
    details = json.loads((tmp_path / "cache/source_audit_details.json").read_text())
    assert details["status"] == "complete"
    assert details["source_audit_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert details["sources"]["train"]["row_groups_scanned"] == 2
    assert details["sources"]["valid"]["scored_rows"] == 8
    loaded_audit, loaded_hash = source_index._load_source_audit(tmp_path)
    assert loaded_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    for split in ("train", "valid"):
        source_index._verify_source_audit_entry(
            tmp_path,
            split,
            loaded_audit["sources"][f"datasets/{split}.parquet"],
            sequence_length=8,
        )


def test_duplicate_sequence_id_fails_at_later_source_group(tmp_path: Path) -> None:
    """Duplicate sequence IDs within one split report the later row-group identity."""
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    train_pin = _write_source(tmp_path, "train", duplicate_sequence_id=True)
    valid_pin = _write_source(tmp_path, "valid")
    output = tmp_path / "cache" / "source_audit.json"

    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=1") as failure:
        source_audit._audit_sources(
            tmp_path,
            output,
            pins={"train": train_pin, "valid": valid_pin},
            sequence_length=8,
        )

    assert "duplicate seq_ix 0; first appeared in row_group=0" in failure.value.detail
    assert not output.exists()


def test_step_in_seq_must_match_zero_based_row_order(tmp_path: Path) -> None:
    """A nonchronological per-group step fails with its exact row and group."""
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    train_pin = _write_source(tmp_path, "train", incorrect_step=(1, 3, 0))
    valid_pin = _write_source(tmp_path, "valid")
    output = tmp_path / "cache" / "source_audit.json"

    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=1") as failure:
        source_audit._audit_sources(
            tmp_path,
            output,
            pins={"train": train_pin, "valid": valid_pin},
            sequence_length=8,
        )

    assert "step_in_seq mismatch at row=3: observed 0, expected 3" in failure.value.detail
    assert not output.exists()


def test_actual_source_hash_must_match_pinned_digest(tmp_path: Path) -> None:
    """A caller-supplied test pin cannot make a different source hash pass."""
    pins = _write_fixture(tmp_path)
    original = pins["train"]
    pins["train"] = SourcePin(
        split=original.split,
        byte_count=original.byte_count,
        sha256="0" * 64,
        row_group_count=original.row_group_count,
        has_scoring_mask=original.has_scoring_mask,
    )
    output = tmp_path / "cache" / "source_audit.json"

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        source_audit._audit_sources(tmp_path, output, pins=pins, sequence_length=8)

    assert not output.exists()
    details = json.loads((tmp_path / "cache/source_audit_details.json").read_text())
    assert details["status"] == "failed"
    assert details["sources"]["train"]["observed_sha256"] == original.sha256


def test_nonfinite_target_fails_with_exact_source_group_and_detail_receipt(tmp_path: Path) -> None:
    """A bad label stops publication and names its exact source-local group."""
    pins = _write_fixture(tmp_path, nonfinite_train_target=True)
    output = tmp_path / "cache" / "source_audit.json"

    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=1") as failure:
        source_audit._audit_sources(tmp_path, output, pins=pins, sequence_length=8)

    assert failure.value.detail == "nonfinite target t1 at row=3"
    assert not output.exists()
    details = json.loads((tmp_path / "cache/source_audit_details.json").read_text())
    assert details["status"] == "failed"
    assert details["failure"]["source"] == str(tmp_path / "datasets/train.parquet")
    assert details["failure"]["row_group"] == 1
    assert "nonfinite target t1 at row=3" in details["failure"]["error"]
    assert details["sources"]["train"]["observed_sha256"] == pins["train"].sha256


def test_mtime_change_during_label_scan_prevents_receipt_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source stat is rechecked after scanning and ignores access-time changes."""
    pins = _write_fixture(tmp_path)
    output = tmp_path / "cache" / "source_audit.json"
    original_audit_row_group = source_audit_scan._audit_row_group

    def mutate_mtime(
        parquet: pq.ParquetFile,
        path: Path,
        split: str,
        row_group: int,
        sequence_length: int,
        has_mask: bool,
        seen_ids: dict[int, int],
    ) -> int:
        """Change only the fixture's mtime after a successful group read."""
        scored_rows = original_audit_row_group(
            parquet, path, split, row_group, sequence_length, has_mask, seen_ids
        )
        stat = path.stat()
        os.utime(path, ns=(0, stat.st_mtime_ns + 10_000_000))
        return scored_rows

    monkeypatch.setattr(source_audit_scan, "_audit_row_group", mutate_mtime)
    with pytest.raises(
        ValueError, match="size or mtime changed during label and sequence-identity scan"
    ):
        source_audit._audit_sources(tmp_path, output, pins=pins, sequence_length=8)

    assert not output.exists()
    details = json.loads((tmp_path / "cache/source_audit_details.json").read_text())
    assert details["status"] == "failed"
    assert details["failure"]["source"] == str(tmp_path / "datasets/train.parquet")
