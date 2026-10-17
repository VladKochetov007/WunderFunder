#!/usr/bin/env python3
"""Derive a fresh fold cache from the frozen main manifests and live graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FOLD_COUNT = 4
SPLITS = ("train", "valid")
CANONICAL_GRAPH_SHA256 = "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766"
CANONICAL_FOLD_CACHE_SHA256 = "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f"
CANONICAL_MANIFEST_SHA256 = {
    0: "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    1: "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
    2: "a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9",
    3: "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343",
}
MAX_MANIFEST_BYTES = 2_000_000


def sha256_file(path: Path) -> str:
    """Hash a small cache or manifest without loading its bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    """Load numeric/string arrays only; object arrays are never enabled."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _split_values(values: np.ndarray) -> np.ndarray:
    """Normalize graph split labels while rejecting non-string identities."""
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    if array.dtype.kind != "U" or array.ndim != 1:
        raise ValueError("split labels must be a one-dimensional string array")
    return array


def _node_keys(splits: np.ndarray, row_groups: np.ndarray) -> list[tuple[str, int]]:
    """Validate and return unique source-qualified row-group identities."""
    split_values = _split_values(splits)
    groups = np.asarray(row_groups)
    if groups.ndim != 1 or groups.dtype.kind not in "iu" or len(groups) != len(split_values):
        raise ValueError("row_group must be a parallel integer array")
    keys = [(str(split), int(group)) for split, group in zip(split_values, groups, strict=True)]
    if any(split not in SPLITS or group < 0 for split, group in keys):
        raise ValueError("graph contains an invalid source-qualified node")
    if len(set(keys)) != len(keys):
        raise ValueError("graph contains duplicate source-qualified nodes")
    return keys


def _graph_components(graph: Mapping[str, np.ndarray]) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Recover components from the live graph and validate its node arrays."""
    missing = {"split", "row_group", "next"} - graph.keys()
    if missing:
        raise ValueError(f"live graph lacks required fields: {sorted(missing)}")
    keys = _node_keys(graph["split"], graph["row_group"])
    successors = np.asarray(graph["next"])
    if successors.ndim != 1 or len(successors) != len(keys):
        raise ValueError("successor graph does not align with source-qualified nodes")
    from wnn.splits import connected_components

    components = connected_components(successors)
    return keys, components


def _json_integer(value: object, description: str) -> int:
    """Require a nonnegative JSON integer without accepting boolean values."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _manifest_identity(
    payload: bytes,
    fold: int,
    expected_digest: str,
    graph_sha256: str,
    graph_bytes: int,
) -> tuple[dict[tuple[str, int], tuple[int, int]], set[int]]:
    """Validate one pinned canonical manifest and return nodes and held-out rows."""
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"canonical fold{fold} manifest SHA256 differs from frozen main")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError(f"canonical fold{fold} manifest exceeds its bounded read size")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError(f"canonical fold{fold} manifest is not a JSON object")
    if (
        manifest.get("version") != 1
        or manifest.get("fold") != fold
        or manifest.get("n_folds") != FOLD_COUNT
    ):
        raise ValueError(f"canonical fold{fold} manifest has the wrong fold identity")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"canonical fold{fold} manifest has no source identities")
    graph_source = sources.get("cache/chains.npz")
    folds_source = sources.get("cache/chain_folds.npz")
    if (
        not isinstance(graph_source, dict)
        or graph_source.get("sha256") != graph_sha256
        or graph_source.get("bytes") != graph_bytes
    ):
        raise ValueError(f"canonical fold{fold} manifest names a different component graph")
    if (
        not isinstance(folds_source, dict)
        or folds_source.get("sha256") != CANONICAL_FOLD_CACHE_SHA256
    ):
        raise ValueError(f"canonical fold{fold} manifest does not bind main's frozen fold cache")

    sequence_rows = manifest.get("sequences")
    if not isinstance(sequence_rows, dict) or set(sequence_rows) != set(SPLITS):
        raise ValueError(f"canonical fold{fold} manifest has incomplete source-qualified nodes")
    nodes: dict[tuple[str, int], tuple[int, int]] = {}
    sequence_ids: set[tuple[str, int]] = set()
    for split in SPLITS:
        rows = sequence_rows[split]
        if not isinstance(rows, list):
            raise ValueError(f"canonical fold{fold} {split} nodes are not a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"canonical fold{fold} has a malformed sequence identity")
            group = _json_integer(row.get("row_group"), "row_group")
            sequence = _json_integer(row.get("seq_ix"), "seq_ix")
            component = _json_integer(row.get("component"), "component")
            key = (split, group)
            if key in nodes or (split, sequence) in sequence_ids:
                raise ValueError(
                    f"canonical fold{fold} duplicates a source-qualified identity {key}"
                )
            nodes[key] = (sequence, component)
            sequence_ids.add((split, sequence))
    evaluation = manifest.get("evaluation")
    held_out = evaluation.get("valid") if isinstance(evaluation, dict) else None
    if not isinstance(held_out, list):
        raise ValueError(f"canonical fold{fold} manifest has no validation evaluation rows")
    held_groups = {_json_integer(group, "evaluation.valid row_group") for group in held_out}
    if len(held_groups) != len(held_out):
        raise ValueError(f"canonical fold{fold} repeats a validation evaluation row")
    if any(("valid", group) not in nodes for group in held_groups):
        raise ValueError(f"canonical fold{fold} evaluates an unknown validation row")
    return nodes, held_groups


def derive_fold_arrays(
    graph: Mapping[str, np.ndarray],
    current_folds: Mapping[str, np.ndarray],
    manifest_payloads: Sequence[bytes],
    *,
    graph_sha256: str,
    graph_bytes: int,
    current_folds_sha256: str,
    expected_graph_sha256: str = CANONICAL_GRAPH_SHA256,
    expected_manifest_sha256: Mapping[int, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Validate canonical folds against the live graph and derive a fresh cache.

    Args:
        graph: The live ``chains.npz`` arrays.
        current_folds: The current Beleriand ``chain_folds.npz`` arrays.
        manifest_payloads: Frozen main manifests in fold0 through fold3 order.
        graph_sha256: SHA256 of the live graph file.
        graph_bytes: Byte size of the live graph file.
        current_folds_sha256: SHA256 of the old fold cache, retained in the receipt.
        expected_graph_sha256: Pinned canonical graph SHA256.
        expected_manifest_sha256: Pinned manifest SHAs; injectable only for fixtures.

    Returns:
        The minimal compatible fold arrays and an audit receipt body.

    Raises:
        ValueError: A graph, source identity, node or fold partition is inconsistent.
    """
    if expected_manifest_sha256 is None:
        expected_manifest_sha256 = CANONICAL_MANIFEST_SHA256
    if graph_sha256 != expected_graph_sha256:
        raise ValueError("live component graph SHA256 differs from canonical main")
    if len(manifest_payloads) != FOLD_COUNT:
        raise ValueError("exactly four canonical fold manifests are required")
    graph_nodes, components = _graph_components(graph)
    graph_component = dict(zip(graph_nodes, components.tolist(), strict=True))
    expected_keys = set(graph_nodes)
    expected_by_fold: dict[int, str] = {}
    canonical_nodes: dict[tuple[str, int], tuple[int, int]] | None = None
    evaluation_by_fold: dict[int, set[int]] = {}
    for fold, payload in enumerate(manifest_payloads):
        expected_digest = expected_manifest_sha256.get(fold)
        if expected_digest is None:
            raise ValueError(f"no frozen SHA256 is configured for fold{fold}")
        expected_by_fold[fold] = expected_digest
        nodes, held_groups = _manifest_identity(
            payload, fold, expected_digest, expected_graph_sha256, graph_bytes
        )
        if set(nodes) != expected_keys:
            raise ValueError(
                f"canonical fold{fold} source-qualified node coverage differs from graph"
            )
        for key, (_sequence, component) in nodes.items():
            if graph_component[key] != component:
                raise ValueError(f"canonical component differs from live graph at {key}")
        if canonical_nodes is None:
            canonical_nodes = nodes
        elif canonical_nodes != nodes:
            raise ValueError("canonical manifests disagree on source-qualified node identities")
        evaluation_by_fold[fold] = held_groups

    if canonical_nodes is None:
        raise ValueError("canonical manifest set is empty")
    valid_groups = {group for split, group in expected_keys if split == "valid"}
    assigned_groups = [group for groups in evaluation_by_fold.values() for group in groups]
    if len(set(assigned_groups)) != len(assigned_groups) or set(assigned_groups) != valid_groups:
        raise ValueError("canonical evaluation folds are not a disjoint, complete valid partition")
    valid_fold = {group: fold for fold, groups in evaluation_by_fold.items() for group in groups}
    folds_by_component: dict[int, set[int]] = {}
    for (split, group), component in graph_component.items():
        if split == "valid":
            folds_by_component.setdefault(component, set()).add(valid_fold[group])
    cross_fold_components = {
        component for component, folds in folds_by_component.items() if len(folds) != 1
    }
    if cross_fold_components:
        raise ValueError("canonical validation component crosses folds")
    component_fold = {
        component: next(iter(folds)) for component, folds in folds_by_component.items()
    }

    current_keys = _node_keys(
        current_folds.get("split", np.array([])),
        current_folds.get("row_group", np.array([])),
    )
    current_fold_values = np.asarray(current_folds.get("fold", np.array([])))
    if (
        current_fold_values.ndim != 1
        or current_fold_values.dtype.kind not in "iu"
        or len(current_fold_values) != len(current_keys)
    ):
        raise ValueError("current cached fold assignments are not a parallel integer array")
    if set(current_keys) != expected_keys:
        raise ValueError(
            "current fold cache does not cover the live graph's source-qualified nodes"
        )
    if np.any(current_fold_values < 0) or np.any(current_fold_values >= FOLD_COUNT):
        raise ValueError("current cached fold assignment is outside 0..3")
    current_by_node = dict(zip(current_keys, current_fold_values.tolist(), strict=True))
    current_folds_by_component: dict[int, set[int]] = {}
    current_rows_by_component: dict[int, list[int]] = {}
    for key, component in graph_component.items():
        split, group = key
        if split == "valid":
            current_folds_by_component.setdefault(component, set()).add(current_by_node[key])
            current_rows_by_component.setdefault(component, []).append(group)
    old_crossed = {
        component for component, folds in current_folds_by_component.items() if len(folds) > 1
    }

    output_fold = np.asarray(
        [component_fold.get(graph_component[key], 0) for key in graph_nodes], dtype=np.int8
    )
    arrays = {
        "split": _split_values(graph["split"]),
        "row_group": np.asarray(graph["row_group"], dtype=np.int64),
        "fold": output_fold,
        "chain": components.astype(np.int64),
    }
    valid_counts = {str(fold): len(evaluation_by_fold[fold]) for fold in range(FOLD_COUNT)}
    receipt: dict[str, object] = {
        "status": "verified_derived_cache",
        "derivation": "frozen main validation memberships mapped to verified live graph components",
        "canonical_graph_sha256": expected_graph_sha256,
        "canonical_original_fold_cache_sha256": CANONICAL_FOLD_CACHE_SHA256,
        "current_fold_cache_sha256": current_folds_sha256,
        "canonical_manifest_sha256": {
            str(fold): expected_by_fold[fold] for fold in range(FOLD_COUNT)
        },
        "source_qualified_nodes": len(graph_nodes),
        "train_nodes": sum(split == "train" for split, _group in graph_nodes),
        "valid_nodes": len(valid_groups),
        "actual_graph_components": len(set(components.tolist())),
        "canonical_valid_rows_by_fold": valid_counts,
        "canonical_validation_components_crossing_folds": 0,
        "current_validation_components_crossing_folds": len(old_crossed),
        "current_valid_rows_in_cross_fold_components": sum(
            len(current_rows_by_component[component]) for component in old_crossed
        ),
        "train_fold_assignment": "inherit linked validation component fold; otherwise 0 (unused by canonical campaign split consumers)",
        "output_fields": ["split", "row_group", "fold", "chain"],
        "output_fold_sha256": hashlib.sha256(output_fold.tobytes()).hexdigest(),
    }
    return arrays, receipt


def restore_cache(
    graph_path: Path,
    current_folds_path: Path,
    manifest_paths: Sequence[Path],
    output_path: Path,
    receipt_path: Path,
    *,
    expected_graph_sha256: str = CANONICAL_GRAPH_SHA256,
    expected_manifest_sha256: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Write an exclusive derived cache and receipt without touching live inputs."""
    output_path, receipt_path = output_path.resolve(), receipt_path.resolve()
    if output_path == receipt_path:
        raise ValueError("derived cache and receipt paths must differ")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("derived cache and receipt paths must not already exist")
    if len(manifest_paths) != FOLD_COUNT:
        raise ValueError("exactly four canonical fold manifest paths are required")
    graph_sha256 = sha256_file(graph_path)
    current_sha256 = sha256_file(current_folds_path)
    graph_bytes = graph_path.stat().st_size
    current_folds_bytes = current_folds_path.stat().st_size
    payloads = []
    manifest_sizes = []
    for path in manifest_paths:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError(f"canonical manifest exceeds its bounded read size: {path}")
        payload = path.read_bytes()
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ValueError(f"canonical manifest exceeds its bounded read size: {path}")
        payloads.append(payload)
        manifest_sizes.append(len(payload))
    graph_arrays = _read_npz(graph_path)
    current_arrays = _read_npz(current_folds_path)
    if (
        sha256_file(graph_path) != graph_sha256
        or graph_path.stat().st_size != graph_bytes
        or sha256_file(current_folds_path) != current_sha256
        or current_folds_path.stat().st_size != current_folds_bytes
    ):
        raise ValueError("graph or current fold cache changed while being read")
    arrays, receipt = derive_fold_arrays(
        graph_arrays,
        current_arrays,
        payloads,
        graph_sha256=graph_sha256,
        graph_bytes=graph_bytes,
        current_folds_sha256=current_sha256,
        expected_graph_sha256=expected_graph_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    receipt.update(
        created_utc=datetime.now(UTC).isoformat(),
        live_graph_path=str(graph_path.resolve()),
        live_graph_bytes=graph_bytes,
        current_fold_cache_path=str(current_folds_path.resolve()),
        current_fold_cache_bytes=current_folds_bytes,
        canonical_manifest_paths=[str(path.resolve()) for path in manifest_paths],
        canonical_manifest_bytes=manifest_sizes,
        output_path=str(output_path),
        output_bytes=output_path.stat().st_size,
        output_sha256=sha256_file(output_path),
    )
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with receipt_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Restore folds from canonical manifests to two explicit new artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--current-folds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    canonical_dir = args.canonical_dir or root / "research/campaign_2026-09-29/artifacts/splits"
    graph = args.graph or root / "cache/chains.npz"
    current_folds = args.current_folds or root / "cache/chain_folds.npz"
    manifests = [canonical_dir / f"fold{fold}.json" for fold in range(FOLD_COUNT)]
    receipt = restore_cache(graph, current_folds, manifests, args.output, args.receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
