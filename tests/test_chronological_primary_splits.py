"""Bounded producer-fixture checks for mandatory chronological primary membership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wnn.chronological_manifest import build_chronological_primary_manifest
from wnn.chronological_source_index import build_source_index
from wnn.chronological_splits import _footer_identity
from wnn.splits import load_campaign_split
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_data import build_data


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    """Create tiny official-source parquet and actual chain/index producer fixtures."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "datasets").mkdir()
    train_ids = np.array([301, 101, 401, 201], dtype=np.int64)
    valid_ids = np.array([901, 701, 601, 801], dtype=np.int64)
    for split, ids in (("train", train_ids), ("valid", valid_ids)):
        table = pa.table(
            {
                "seq_ix": ids,
                "t0": np.zeros(4, dtype=np.float32),
                "t1": np.ones(4, dtype=np.float32),
                **({"is_scored": np.ones(4, dtype=bool)} if split == "valid" else {}),
            }
        )
        pq.write_table(table, tmp_path / f"datasets/{split}.parquet", row_group_size=1)
    source_identities = {}
    for split in ("train", "valid"):
        source = tmp_path / f"datasets/{split}.parquet"
        stat = source.stat()
        footer = _footer_identity(source)
        source_identities[f"datasets/{split}.parquet"] = {
            "bytes": stat.st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mtime_ns": stat.st_mtime_ns,
            "footer_bytes": footer["footer_bytes"],
            "footer_sha256": footer["footer_sha256"],
            "row_group_count": 4,
            "sequence_length": 1,
            "has_is_scored": split == "valid",
        }
    source_audit_path = tmp_path / "cache/source_audit.json"
    source_audit_path.write_text(
        json.dumps({"schema": "wnn.source-audit.v1", "sources": source_identities}, indent=2) + "\n"
    )
    for split in ("train", "valid"):
        source = tmp_path / f"datasets/{split}.parquet"
        build_source_index(
            tmp_path,
            split,
            expected_bytes=source.stat().st_size,
            expected_sha256=source_identities[f"datasets/{split}.parquet"]["sha256"],
            expected_row_groups=4,
            sequence_length=1,
        )
    index_receipts = {
        split: json.loads((tmp_path / f"cache/{split}_index.identity.json").read_text())
        for split in ("train", "valid")
    }
    np.savez(
        tmp_path / "cache/chronological_chains.npz",
        split=np.array(["train"] * 4 + ["valid"] * 4),
        row_group=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        next=np.array([4, -1, -1, -1, -1, -1, 7, 6], dtype=np.int64),
        source_audit_sha256=np.array(hashlib.sha256(source_audit_path.read_bytes()).hexdigest()),
        source_sha256=np.array(
            [index_receipts[split]["source_sha256"] for split in ("train", "valid")]
        ),
        index_map_sha256=np.array(
            [index_receipts[split]["index_map_sha256"] for split in ("train", "valid")]
        ),
        index_receipt_sha256=np.array(
            [
                hashlib.sha256(
                    (tmp_path / f"cache/{split}_index.identity.json").read_bytes()
                ).hexdigest()
                for split in ("train", "valid")
            ]
        ),
    )
    return tmp_path


def test_shuffled_sequence_ids_preserve_source_membership_and_exclude_linked_fit(
    source_root: Path,
) -> None:
    path = build_chronological_primary_manifest(source_root, source_root / "primary.json")
    split = load_campaign_split(path, source_root)
    assert split.train_groups == (1, 2, 3)
    assert split.valid_groups == ()
    assert split.evaluation_groups == (0, 1, 2, 3)
    assert split.fold == 0 and split.n_folds == 1
    component_of = {(source, group): component for source, group, component in split.components}
    assert component_of["train", 0] == component_of["valid", 0]
    assert set(split.train_groups).isdisjoint({0})
    manifest = json.loads(path.read_text())
    assert "seq_ix" not in json.dumps(manifest["sequences"])
    assert manifest["chronology_basis"].endswith("calendar timestamps unavailable")


def test_valid_targets_never_enter_fit_or_normalization_and_teacher_cache(
    source_root: Path,
) -> None:
    path = build_chronological_primary_manifest(source_root, source_root / "primary.json")
    data = build_data(
        {
            "model": {"hint_dim": 0},
            "kd_weight": 0,
            "hint_weight": 0,
            "hint_mode": "aligned",
            "seed": 7,
            "training_mask": "all_after_warmup",
            "normalization_scope": "prefix",
            "normalization_sequences": 32,
        },
        source_root,
        path,
    )
    assert data.fit.members == [("train", 1), ("train", 2), ("train", 3)]
    assert data.evaluation.members == [("valid", group) for group in range(4)]
    assert data.provenance["fit_members"] == [["train", 1], ["train", 2], ["train", 3]]
    assert set(data.provenance["normalization"]["source_fingerprints"]) == {
        "datasets/train.parquet"
    }

    cache_root = source_root / "teacher-cache"
    cache_root.mkdir()
    cache_members = data.fit.members + [("valid", 0)]
    (cache_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "sequence-teacher-cache-v1",
                "complete": True,
                "input_schema": "raw112-v1",
                "membership": "fit",
                "sequence_length": 20_000,
                "split_sha256": data.split.sha256,
                "hint_dim": 0,
                "producer": {},
                "sequences": [
                    {
                        "split": split,
                        "row_group": group,
                        "file": f"{split}-{group}.npy",
                        "sha256": "a" * 64,
                    }
                    for split, group in cache_members
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="exactly fitting membership"):
        TeacherCache(cache_root, data.split, data.fit.members)


def test_component_overlap_and_fit_valid_tampering_are_rejected(source_root: Path) -> None:
    path = build_chronological_primary_manifest(source_root, source_root / "primary.json")
    manifest = json.loads(path.read_text())
    manifest["fit"]["train"].append(0)
    manifest["fit"]["train"].sort()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership differs"):
        load_campaign_split(path, source_root)


def test_source_identity_change_is_rejected(source_root: Path) -> None:
    path = build_chronological_primary_manifest(source_root, source_root / "primary.json")
    with (source_root / "datasets/valid.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        load_campaign_split(path, source_root)
