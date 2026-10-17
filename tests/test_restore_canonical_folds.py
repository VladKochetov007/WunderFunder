from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0,
    str(ROOT / "research/campaign_2026-10-18/priority25_execution_r4/packetcode"),
)
from restore_canonical_folds import (  # noqa: E402
    CANONICAL_FOLD_CACHE_SHA256,
    restore_cache,
)


def fixture_inputs(tmp_path: Path) -> tuple[Path, Path, list[Path], dict[int, str]]:
    """Write an eight-node graph, canonical memberships and a deliberately bad old cache."""
    cache = tmp_path / "cache"
    manifests_dir = tmp_path / "manifests"
    cache.mkdir()
    manifests_dir.mkdir()
    splits = np.asarray(["train"] * 4 + ["valid"] * 5)
    row_groups = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=np.int64)
    successors = np.asarray([4, 6, 7, 8, 5, -1, -1, -1, -1], dtype=np.int64)
    graph_path = cache / "chains.npz"
    np.savez(graph_path, split=splits, row_group=row_groups, next=successors)
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    graph_bytes = graph_path.stat().st_size
    current_folds_path = cache / "chain_folds.npz"
    old_folds = np.asarray([0, 1, 2, 3, 0, 1, 1, 2, 3], dtype=np.int8)
    np.savez(current_folds_path, split=splits, row_group=row_groups, fold=old_folds)

    components = np.asarray([0, 1, 2, 3, 0, 0, 1, 2, 3])
    nodes = {
        "train": [
            {"row_group": group, "seq_ix": 100 + group, "component": int(components[group])}
            for group in range(4)
        ],
        "valid": [
            {
                "row_group": group,
                "seq_ix": 200 + group,
                "component": int(components[4 + group]),
            }
            for group in range(5)
        ],
    }
    held_groups = {0: [0, 1], 1: [2], 2: [3], 3: [4]}
    manifest_paths = []
    manifest_hashes = {}
    for fold in range(4):
        manifest = {
            "version": 1,
            "fold": fold,
            "n_folds": 4,
            "sources": {
                "cache/chains.npz": {"bytes": graph_bytes, "sha256": graph_sha},
                "cache/chain_folds.npz": {"sha256": CANONICAL_FOLD_CACHE_SHA256},
            },
            "sequences": nodes,
            "evaluation": {"valid": held_groups[fold]},
        }
        path = manifests_dir / f"fold{fold}.json"
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
        path.write_bytes(payload)
        manifest_paths.append(path)
        manifest_hashes[fold] = hashlib.sha256(payload).hexdigest()
    return graph_path, current_folds_path, manifest_paths, manifest_hashes


def test_restores_frozen_partition_and_records_old_component_leakage(tmp_path: Path) -> None:
    graph, current, manifests, manifest_hashes = fixture_inputs(tmp_path)
    old_bytes = current.read_bytes()
    output = tmp_path / "derived/chain_folds.npz"
    receipt_path = tmp_path / "derived/restore.json"

    receipt = restore_cache(
        graph,
        current,
        manifests,
        output,
        receipt_path,
        expected_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
        expected_manifest_sha256=manifest_hashes,
    )

    with np.load(output, allow_pickle=False) as restored:
        assert set(restored.files) == {"split", "row_group", "fold", "chain"}
        valid = restored["split"] == "valid"
        assert dict(zip(restored["row_group"][valid], restored["fold"][valid], strict=True)) == {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }
        assert restored["chain"].tolist() == [0, 1, 2, 3, 0, 0, 1, 2, 3]
        assert restored["fold"].tolist() == [0, 1, 2, 3, 0, 0, 1, 2, 3]
    assert receipt["current_validation_components_crossing_folds"] == 1
    assert receipt["current_valid_rows_in_cross_fold_components"] == 2
    assert receipt["canonical_valid_rows_by_fold"] == {"0": 2, "1": 1, "2": 1, "3": 1}
    assert json.loads(receipt_path.read_text())["output_sha256"] == receipt["output_sha256"]
    assert current.read_bytes() == old_bytes


def test_rejects_canonical_manifest_identity_change_before_writing(tmp_path: Path) -> None:
    graph, current, manifests, manifest_hashes = fixture_inputs(tmp_path)
    manifests[0].write_text(manifests[0].read_text() + " ")
    output = tmp_path / "derived/chain_folds.npz"
    receipt = tmp_path / "derived/restore.json"

    with pytest.raises(ValueError, match="SHA256 differs from frozen main"):
        restore_cache(
            graph,
            current,
            manifests,
            output,
            receipt,
            expected_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            expected_manifest_sha256=manifest_hashes,
        )
    assert not output.exists()
    assert not receipt.exists()


def test_rejects_incomplete_valid_partition(tmp_path: Path) -> None:
    graph, current, manifests, _manifest_hashes = fixture_inputs(tmp_path)
    manifest = json.loads(manifests[3].read_text())
    manifest["evaluation"]["valid"] = []
    payloads = []
    expected_hashes = {}
    for fold, path in enumerate(manifests):
        value = manifest if fold == 3 else json.loads(path.read_text())
        encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
        payloads.append(encoded)
        expected_hashes[fold] = hashlib.sha256(encoded).hexdigest()
    output = tmp_path / "derived/chain_folds.npz"
    receipt = tmp_path / "derived/restore.json"

    from restore_canonical_folds import derive_fold_arrays

    with np.load(graph, allow_pickle=False) as arrays:
        graph_arrays = {name: arrays[name] for name in arrays.files}
    with np.load(current, allow_pickle=False) as arrays:
        current_arrays = {name: arrays[name] for name in arrays.files}
    with pytest.raises(ValueError, match="disjoint, complete valid partition"):
        derive_fold_arrays(
            graph_arrays,
            current_arrays,
            payloads,
            graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            graph_bytes=graph.stat().st_size,
            current_folds_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
            expected_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            expected_manifest_sha256=expected_hashes,
        )
    assert not output.exists()
    assert not receipt.exists()


def test_rejects_live_graph_sha256_outside_canonical_source_identity(tmp_path: Path) -> None:
    graph, current, manifests, manifest_hashes = fixture_inputs(tmp_path)
    output = tmp_path / "derived/chain_folds.npz"
    receipt = tmp_path / "derived/restore.json"

    with pytest.raises(ValueError, match="live component graph SHA256 differs"):
        restore_cache(
            graph,
            current,
            manifests,
            output,
            receipt,
            expected_graph_sha256="0" * 64,
            expected_manifest_sha256=manifest_hashes,
        )
    assert not output.exists()
    assert not receipt.exists()


def test_rejects_manifest_bound_to_a_different_graph(tmp_path: Path) -> None:
    graph, current, manifests, manifest_hashes = fixture_inputs(tmp_path)
    manifest = json.loads(manifests[0].read_text())
    manifest["sources"]["cache/chains.npz"]["sha256"] = "f" * 64
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifests[0].write_bytes(encoded)
    manifest_hashes[0] = hashlib.sha256(encoded).hexdigest()
    output = tmp_path / "derived/chain_folds.npz"
    receipt = tmp_path / "derived/restore.json"

    with pytest.raises(ValueError, match="names a different component graph"):
        restore_cache(
            graph,
            current,
            manifests,
            output,
            receipt,
            expected_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            expected_manifest_sha256=manifest_hashes,
        )
    assert not output.exists()
    assert not receipt.exists()


def test_rejects_canonical_component_split_across_validation_folds(tmp_path: Path) -> None:
    graph, current, manifests, _manifest_hashes = fixture_inputs(tmp_path)
    payloads = [json.loads(path.read_text()) for path in manifests]
    payloads[0]["evaluation"]["valid"].remove(1)
    payloads[1]["evaluation"]["valid"].append(1)
    encoded = [(json.dumps(value, sort_keys=True, indent=2) + "\n").encode() for value in payloads]
    expected_hashes = {
        fold: hashlib.sha256(payload).hexdigest() for fold, payload in enumerate(encoded)
    }
    from restore_canonical_folds import derive_fold_arrays

    with np.load(graph, allow_pickle=False) as arrays:
        graph_arrays = {name: arrays[name] for name in arrays.files}
    with np.load(current, allow_pickle=False) as arrays:
        current_arrays = {name: arrays[name] for name in arrays.files}
    with pytest.raises(ValueError, match="validation component crosses folds"):
        derive_fold_arrays(
            graph_arrays,
            current_arrays,
            encoded,
            graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            graph_bytes=graph.stat().st_size,
            current_folds_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
            expected_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
            expected_manifest_sha256=expected_hashes,
        )
