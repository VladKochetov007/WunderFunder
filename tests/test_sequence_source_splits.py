"""Receiver-only tests for source coverage and component-excluded holdouts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wnn import sequence_source_inputs
from wnn.sequence_source_inputs import GRAPH, INDEX, METHOD, RECEIPT, source_identity
from wnn.sequence_source_splits import prepare_sequence_source_split
from wnn.splits import file_sha256, load_campaign_split


def _seal(root: Path) -> None:
    names = ["datasets/train.parquet", "datasets/valid.parquet", GRAPH, *INDEX.values()]
    receipt = {
        "schema": "wnn.sequence-source-inputs.v1",
        "method": METHOD,
        "sources": {name: source_identity(root / name) for name in names},
    }
    (root / RECEIPT).write_text(json.dumps(receipt))


def _root(root: Path, train_count: int = 4) -> Path:
    (root / "cache").mkdir(parents=True)
    (root / "datasets").mkdir()
    for source, count in (("train", train_count), ("valid", 3)):
        ids = np.arange(count, dtype=np.int64) + (10000 if source == "valid" else 0)
        with pq.ParquetWriter(
            root / f"datasets/{source}.parquet", pa.schema([("seq_ix", pa.int64())])
        ) as writer:
            for sequence in ids:
                writer.write_table(pa.table({"seq_ix": np.full(20_000, sequence, dtype=np.int64)}))
        np.savez(root / INDEX[source], row_group=np.arange(count), seq_ix=ids)
    next_nodes = np.full(train_count + 3, -1, dtype=np.int64)
    next_nodes[0], next_nodes[1] = train_count, 0
    np.savez(
        root / GRAPH,
        split=np.asarray(["train"] * train_count + ["valid"] * 3),
        row_group=np.asarray([*range(train_count), *range(3)]),
        next=next_nodes,
    )
    _seal(root)
    return root


def _rewrite_npz(path: Path, replacement: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    np.savez(path, **(values | replacement))


def test_source_split_is_full_valid_train_only_and_excludes_transitive_links(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path / "data")
    first = prepare_sequence_source_split(root, tmp_path / "first.json")
    second = prepare_sequence_source_split(root, tmp_path / "second.json")
    assert first.train_groups == (2, 3)
    assert first.valid_groups == ()
    assert first.evaluation_groups == (0, 1, 2)
    assert first.fold == 0 and first.n_folds == 1
    assert first.sha256 == second.sha256
    assert load_campaign_split(first.path, root) == first
    assert not (root / "cache/chains.npz").exists()
    assert not (root / "cache/train_targets.npz").exists()
    assert not (root / "cache/chain_folds.npz").exists()
    with pytest.raises(ValueError, match="outside"):
        first.assert_fit_groups((0,), ())
    with pytest.raises(FileExistsError):
        prepare_sequence_source_split(root, first.path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("fit", "train", [2]),
        ("fit", "train", [True, 2, 3]),
        ("fit", "valid", [1]),
        ("evaluation", "valid", [0, 1]),
        ("evaluation", "valid", [0, 1, 1, 2]),
        ("excluded", "train_component_overlap", []),
    ],
)
def test_source_split_recomputes_every_membership(
    tmp_path: Path, section: str, key: str, value: list[int]
) -> None:
    root = _root(tmp_path / "data")
    output = prepare_sequence_source_split(root, tmp_path / "split.json").path
    manifest = json.loads(output.read_text())
    manifest[section][key] = value
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership"):
        load_campaign_split(output, root)


def test_source_split_recomputes_component_ids_and_protocol(tmp_path: Path) -> None:
    root = _root(tmp_path / "data")
    output = prepare_sequence_source_split(root, tmp_path / "split.json").path
    original = json.loads(output.read_text())
    for field, value in (("fold", 2), ("n_folds", 4), ("version", True)):
        output.write_text(json.dumps(original | {field: value}))
        with pytest.raises(ValueError, match="membership"):
            load_campaign_split(output, root)
    original["sequences"]["train"][2]["component"] = 0
    output.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="sequences"):
        load_campaign_split(output, root)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "wrong_id"])
def test_source_split_rejects_incomplete_or_wrong_index(tmp_path: Path, change: str) -> None:
    root = _root(tmp_path / "data")
    groups, ids = np.arange(4), np.arange(4)
    if change == "missing":
        groups, ids = groups[:-1], ids[:-1]
    elif change == "duplicate":
        groups[-1] = 2
    elif change == "unknown":
        groups[-1] = 4
    else:
        ids[-1] = 99
    _rewrite_npz(root / INDEX["train"], {"row_group": groups, "seq_ix": ids})
    _seal(root)
    with pytest.raises(ValueError, match="index"):
        prepare_sequence_source_split(root, tmp_path / "split.json")


def test_source_split_requires_physical_train_group_528(tmp_path: Path) -> None:
    root = _root(tmp_path / "data", train_count=529)
    with np.load(root / GRAPH, allow_pickle=False) as graph:
        keep = (graph["split"] != "train") | (graph["row_group"] != 528)
        groups, sources = graph["row_group"][keep], graph["split"][keep]
    _rewrite_npz(
        root / GRAPH,
        {
            "split": sources,
            "row_group": groups,
            "next": np.full(len(groups), -1),
        },
    )
    _seal(root)
    with pytest.raises(ValueError, match="every physical train row group"):
        prepare_sequence_source_split(root, tmp_path / "split.json")


@pytest.mark.parametrize("change", ["duplicate", "unknown_source", "unknown_group", "bad_edge"])
def test_source_split_rejects_malformed_graph(tmp_path: Path, change: str) -> None:
    root = _root(tmp_path / "data")
    with np.load(root / GRAPH, allow_pickle=False) as graph:
        values = {name: graph[name] for name in graph.files}
    if change == "duplicate":
        values["row_group"][1] = 0
    elif change == "unknown_source":
        values["split"][1] = "test"
    elif change == "unknown_group":
        values["row_group"][1] = 999
    else:
        values["next"][1] = 999
    _rewrite_npz(root / GRAPH, values)
    _seal(root)
    with pytest.raises(ValueError):
        prepare_sequence_source_split(root, tmp_path / "split.json")


@pytest.mark.parametrize("source", ["datasets/train.parquet", GRAPH, INDEX["valid"]])
def test_source_split_rejects_drift(tmp_path: Path, source: str) -> None:
    root = _root(tmp_path / "data")
    output = prepare_sequence_source_split(root, tmp_path / "split.json").path
    path = root / source
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="source changed"):
        load_campaign_split(output, root)


def test_source_split_accepts_full_byte_verified_portable_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path / "data")
    output = prepare_sequence_source_split(root, tmp_path / "split.json").path
    manifest = json.loads(output.read_text())
    registry = tmp_path / "portable.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    name: {"original_identity": identity, "sha256": file_sha256(root / name)}
                    for name, identity in manifest["sources"].items()
                },
            }
        )
    )
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    for name in manifest["sources"]:
        path = copied / name
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", str(registry))
    assert load_campaign_split(output, copied).valid_groups == ()


def test_source_split_rejects_partial_inputs_without_repairing_legacy_caches(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path / "data")
    (root / RECEIPT).unlink()
    with pytest.raises(FileExistsError, match="partial"):
        prepare_sequence_source_split(root, tmp_path / "split.json")


@pytest.mark.parametrize("change", ["partial", "mixed", "duplicate"])
def test_source_split_validates_physical_sequence_ids(tmp_path: Path, change: str) -> None:
    root = _root(tmp_path / "data")
    path = root / "datasets/train.parquet"
    with pq.ParquetWriter(path, pa.schema([("seq_ix", pa.int64())])) as writer:
        for group in range(4):
            values = np.full(19_999 if change == "partial" else 20_000, group, dtype=np.int64)
            if group == 1 and change == "mixed":
                values[-1] = 2
            if group == 1 and change == "duplicate":
                values[:] = 0
            writer.write_table(pa.table({"seq_ix": values}))
    _seal(root)
    with pytest.raises(ValueError, match="row group|duplicate seq_ix"):
        prepare_sequence_source_split(root, tmp_path / "split.json")


def test_receiver_can_prepare_inputs_from_parquets_without_labels_or_legacy_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fresh"
    (root / "datasets").mkdir(parents=True)
    monkeypatch.setattr(sequence_source_inputs, "FEATURE_COLUMNS", ["f0"])
    for source, offset in (("train", 0), ("valid", 100)):
        schema = pa.schema([("seq_ix", pa.int64()), ("f0", pa.float32())])
        with pq.ParquetWriter(root / f"datasets/{source}.parquet", schema) as writer:
            for group in range(3):
                writer.write_table(
                    pa.table(
                        {
                            "seq_ix": np.full(20_000, offset + group, dtype=np.int64),
                            "f0": np.linspace(0, 1, 20_000, dtype=np.float32)
                            + (offset + group) * 100,
                        }
                    )
                )
    result = prepare_sequence_source_split(root, tmp_path / "split.json", workers=1)
    assert result.train_groups == (0, 1, 2)
    assert result.valid_groups == ()
    assert result.evaluation_groups == (0, 1, 2)
    assert (root / RECEIPT).is_file()
    assert not (root / "cache/chains.npz").exists()
