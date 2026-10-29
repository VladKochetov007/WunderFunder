"""Bounded checks for source receipts and the source-qualified feature graph."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.sequence_chains as sequence_chains
import wnn.chronological_source_index as source_index
from wnn.chronological_source_index import (
    SourceRowGroupReadError,
    build_source_index,
    validate_source_index_receipt,
)
from wnn.chronological_splits import _footer_identity
from wnn.data import FEATURE_COLUMNS


def _write_root_audit(root: Path, groups: dict[str, int]) -> dict[str, dict[str, object]]:
    """Create tiny synthetic parquets and a root-receipt-shaped test audit."""
    (root / "datasets").mkdir()
    (root / "cache").mkdir()
    source_identities: dict[str, dict[str, object]] = {}
    for split, group_count in groups.items():
        columns: dict[str, object] = {"seq_ix": np.arange(group_count, dtype=np.int64)}
        if split == "valid":
            columns["is_scored"] = np.ones(group_count, dtype=bool)
        source = root / f"datasets/{split}.parquet"
        pq.write_table(pa.table(columns), source, row_group_size=1)
        stat = source.stat()
        footer = _footer_identity(source)
        source_identities[f"datasets/{split}.parquet"] = {
            "bytes": stat.st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mtime_ns": stat.st_mtime_ns,
            "footer_bytes": footer["footer_bytes"],
            "footer_sha256": footer["footer_sha256"],
            "row_group_count": group_count,
            "sequence_length": 1,
            "has_is_scored": split == "valid",
        }
    audit = {"schema": "wnn.source-audit.v1", "sources": source_identities}
    (root / "cache/source_audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n"
    )
    return source_identities


def _build_test_index(root: Path, split: str, identity: dict[str, object]) -> Path:
    """Run the same thin-index producer with a small test-only pin."""
    return build_source_index(
        root,
        split,
        expected_bytes=int(identity["bytes"]),
        expected_sha256=str(identity["sha256"]),
        expected_row_groups=int(identity["row_group_count"]),
        sequence_length=int(identity["sequence_length"]),
    )


def test_thin_index_covers_group_528_and_binds_root_audit(tmp_path: Path) -> None:
    """The new index path includes every source row group and stores no targets."""
    identities = _write_root_audit(tmp_path, {"train": 529, "valid": 1})
    path = _build_test_index(tmp_path, "train", identities["datasets/train.parquet"])
    with np.load(path, allow_pickle=False) as index:
        assert set(index.files) == {"row_group", "seq_ix"}
        assert index["row_group"].shape == (529,)
        assert index["row_group"][528] == 528
        assert index["seq_ix"][528] == 528
    audit_hash = hashlib.sha256((tmp_path / "cache/source_audit.json").read_bytes()).hexdigest()
    row_group, seq_ix = validate_source_index_receipt(tmp_path, "train", audit_hash)
    assert len(row_group) == len(seq_ix) == 529
    audit_path = tmp_path / "cache/source_audit.json"
    audit_path.write_bytes(audit_path.read_bytes() + b" ")
    changed_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="receipt disagrees with root audit"):
        validate_source_index_receipt(tmp_path, "train", changed_hash)


def test_unreadable_group_emits_exact_index_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index read error preserves the source and exact failing row-group id."""
    identities = _write_root_audit(tmp_path, {"train": 3, "valid": 1})

    def fail_group(path: Path, split: str, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        raise SourceRowGroupReadError(split, 2, str(path), "DecodeError: corrupt row group")

    monkeypatch.setattr(source_index, "_read_sequence_ids", fail_group)
    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=2"):
        _build_test_index(tmp_path, "train", identities["datasets/train.parquet"])
    receipt = json.loads((tmp_path / "cache/train_index.failure.json").read_text())
    assert receipt["source"] == "datasets/train.parquet"
    assert receipt["row_group"] == 2
    assert receipt["source_sha256"] == identities["datasets/train.parquet"]["sha256"]


def test_source_graph_jobs_include_historical_group_528() -> None:
    """The new graph job builder enumerates every index group, including 528."""
    root = Path("receiver")
    jobs = sequence_chains._source_group_jobs(
        root, {"train": np.arange(529), "valid": np.arange(2)}
    )
    assert ("train", 528, str(root), 20_000) in jobs
    assert len(jobs) == 531


def test_source_feature_sample_is_feature_only_and_matches_midpoint_rule(tmp_path: Path) -> None:
    """The new sampler uses the same two-row head/tail/midpoint samples as V1."""
    (tmp_path / "datasets").mkdir()
    rng = np.random.default_rng(11)
    features = rng.normal(size=(8, len(FEATURE_COLUMNS))).astype(np.float32)
    table = pa.table(
        {name: features[:, column] for column, name in enumerate(FEATURE_COLUMNS)}
        | {"seq_ix": np.full(8, 7001, dtype=np.int64)}
    )
    source = tmp_path / "datasets/train.parquet"
    pq.write_table(table, source, row_group_size=8)
    cache_key = str(source)
    sequence_chains._OFFICIAL_FILES.pop(cache_key, None)
    head, tail, mid = sequence_chains._official_edge(("train", 0, str(tmp_path), 8))
    assert np.array_equal(head, features[:2])
    assert np.array_equal(tail, features[-2:])
    assert np.array_equal(mid, features[4:6])


def test_source_feature_read_error_names_exact_parquet_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature decode error raises with the source parquet and row-group number."""

    class BrokenParquet:
        def read_row_group(
            self, row_group: int, *, columns: list[str], use_threads: bool
        ) -> pa.Table:
            raise OSError("synthetic decode failure")

    source = tmp_path / "datasets/train.parquet"
    cache_key = str(source)
    sequence_chains._OFFICIAL_FILES.pop(cache_key, None)
    monkeypatch.setattr(sequence_chains.pq, "ParquetFile", lambda _: BrokenParquet())
    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=528"):
        sequence_chains._official_edge(("train", 528, str(tmp_path), 20_000))


def test_nonfinite_feature_emits_exact_chain_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonfinite value in a tiny official-graph fixture fails with source/group evidence."""
    datasets = tmp_path / "datasets"
    cache = tmp_path / "cache"
    datasets.mkdir()
    cache.mkdir()
    train = np.zeros((20_000, len(FEATURE_COLUMNS)), dtype=np.float32)
    train[123, 7] = np.nan
    valid = np.zeros_like(train)
    for split, values in (("train", train), ("valid", valid)):
        pq.write_table(
            pa.table({name: values[:, column] for column, name in enumerate(FEATURE_COLUMNS)}),
            datasets / f"{split}.parquet",
            row_group_size=20_000,
        )

    audit_bytes = b"synthetic root source audit\n"
    (cache / "source_audit.json").write_bytes(audit_bytes)
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    for split, source_sha256 in (("train", "a" * 64), ("valid", "c" * 64)):
        (cache / f"{split}_index.identity.json").write_text(
            json.dumps(
                {
                    "source_bytes": 123,
                    "source_sha256": source_sha256,
                    "source_audit_sha256": audit_sha256,
                    "index_map_sha256": "d" * 64,
                }
            )
        )

    monkeypatch.setattr(
        sequence_chains,
        "OFFICIAL_SOURCE_PINS",
        {
            "train": types.SimpleNamespace(row_group_count=1),
            "valid": types.SimpleNamespace(row_group_count=1),
        },
    )
    monkeypatch.setattr(
        sequence_chains,
        "verify_source_index_receipt",
        lambda _root, _split: (np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)),
    )
    sequence_chains._OFFICIAL_FILES.pop(str(datasets / "train.parquet"), None)
    sequence_chains._OFFICIAL_FILES.pop(str(datasets / "valid.parquet"), None)

    with pytest.raises(SourceRowGroupReadError, match=r"train\.parquet row_group=0") as failure:
        sequence_chains.build_official_chains(tmp_path, workers=1)

    assert (
        failure.value.detail == f"nonfinite float32 feature at row=123, column={FEATURE_COLUMNS[7]}"
    )
    receipt = json.loads((cache / "chronological_chains.failure.json").read_text())
    assert receipt["source"] == "datasets/train.parquet"
    assert receipt["split"] == "train"
    assert receipt["row_group"] == 0
    assert receipt["source_sha256"] == "a" * 64
    assert receipt["source_audit_sha256"] == audit_sha256
    assert receipt["error"] == failure.value.detail


def test_successor_rule_matches_frozen_v1_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reusable rule is bit-for-bit equivalent on a tiny deterministic sample."""
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(9, 6, 112)).astype(np.float32)
    head = samples[:, 0]
    tail = samples[:, -1]
    mid = samples[:, 3:5]
    reference = np.abs(mid[:, 1, :112] - mid[:, 0, :112]).sum(1)

    class ManhattanNearestNeighbors:
        def __init__(self, n_neighbors: int, metric: str) -> None:
            assert metric == "manhattan"
            self.n_neighbors = n_neighbors

        def fit(self, values: np.ndarray) -> ManhattanNearestNeighbors:
            self.values = values
            return self

        def kneighbors(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            distances = np.abs(queries[:, None, :] - self.values[None, :, :]).sum(2)
            indexes = np.argsort(distances, axis=1, kind="stable")[:, : self.n_neighbors]
            return np.take_along_axis(distances, indexes, axis=1), indexes

    fake_neighbors = types.ModuleType("sklearn.neighbors")
    fake_neighbors.NearestNeighbors = ManhattanNearestNeighbors
    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn.neighbors = fake_neighbors
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", fake_neighbors)

    distances_to_heads = np.abs(tail[:, None, :] - head[None, :, :]).sum(2)
    indexes = np.argsort(distances_to_heads, axis=1, kind="stable")[:, :2]
    distances = np.take_along_axis(distances_to_heads, indexes, axis=1)
    distances_to_tails = np.abs(head[:, None, :] - tail[None, :, :]).sum(2)
    back = np.argsort(distances_to_tails, axis=1, kind="stable")[:, :1]
    expected = np.full(len(samples), -1)
    threshold = np.quantile(reference, 0.99)
    for row in range(len(samples)):
        candidate = indexes[row, 0] if indexes[row, 0] != row else indexes[row, 1]
        distance = distances[row, 0] if indexes[row, 0] != row else distances[row, 1]
        if back[candidate, 0] == row and distance <= threshold:
            expected[row] = candidate
    actual, _, _, _, _ = sequence_chains._successors(head, tail, mid)
    assert np.array_equal(actual, expected)


def test_chain_failure_receipt_names_corrupt_source_group(tmp_path: Path) -> None:
    """Graph sampling records the exact source identity when feature decoding fails."""
    (tmp_path / "cache").mkdir()
    identity = {
        "source_bytes": 123,
        "source_sha256": "a" * 64,
        "source_audit_sha256": "b" * 64,
    }
    (tmp_path / "cache/train_index.identity.json").write_text(json.dumps(identity))
    sequence_chains._write_failure_receipt(
        tmp_path,
        SourceRowGroupReadError(
            "train", 528, "datasets/train.parquet", "DecodeError: corrupt row group"
        ),
    )
    receipt = json.loads((tmp_path / "cache/chronological_chains.failure.json").read_text())
    assert receipt["source"] == "datasets/train.parquet"
    assert receipt["row_group"] == 528
    assert receipt["source_audit_sha256"] == "b" * 64
