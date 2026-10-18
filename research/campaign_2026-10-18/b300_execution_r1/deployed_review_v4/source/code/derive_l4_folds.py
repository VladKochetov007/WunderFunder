#!/usr/bin/env python3
"""Derive fold assignments from the pinned graph and canonical manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from l4_source_preflight import validate_graph_source, validate_input_preflight

GRAPH_SHA256 = "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766"
GRAPH_BYTES = 650142
ORIGINAL_FOLD_SHA256 = "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f"
DERIVED_FOLD_SHA256 = "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567"
MANIFEST_SHA256 = {
    0: "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    1: "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
    2: "a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9",
    3: "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343",
}


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_components(successors: np.ndarray) -> np.ndarray:
    """Return deterministic undirected component IDs for successor edges."""
    edges = np.asarray(successors)
    if edges.ndim != 1 or edges.dtype.kind not in "iu":
        raise ValueError("graph next array must be a one-dimensional integer array")
    if np.any(edges < -1) or np.any(edges >= len(edges)):
        raise ValueError("graph has an out-of-range successor")
    parents = np.arange(len(edges))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for index, successor in enumerate(edges):
        if successor >= 0:
            left, right = root(index), root(int(successor))
            parents[max(left, right)] = min(left, right)
    roots = np.array([root(index) for index in range(len(edges))])
    return np.unique(roots, return_inverse=True)[1]


def validate_graph_identity(digest: str, byte_count: int) -> None:
    """Reject graph bytes that are not the frozen canonical graph."""
    if digest != GRAPH_SHA256 or byte_count != GRAPH_BYTES:
        raise ValueError(
            "graph identity mismatch; regenerate with pinned a266 sequence_chains.py "
            "and exact train/valid parquet hashes before deriving folds"
        )


def fold_partition(
    component_by_node: dict[tuple[str, int], int],
    evaluation_by_fold: dict[int, set[int]],
    valid_groups: set[int],
) -> dict[int, int]:
    """Map components to one fold after enforcing a complete disjoint partition."""
    assigned = [group for groups in evaluation_by_fold.values() for group in groups]
    if len(assigned) != len(set(assigned)) or set(assigned) != valid_groups:
        raise ValueError("canonical folds are not a disjoint complete valid partition")
    if set(evaluation_by_fold) != set(range(4)):
        raise ValueError("canonical folds must contain exactly folds 0 through 3")
    fold_by_group = {group: fold for fold, groups in evaluation_by_fold.items() for group in groups}
    component_folds: dict[int, set[int]] = {}
    for (split, group), component in component_by_node.items():
        if split == "valid":
            component_folds.setdefault(component, set()).add(fold_by_group[group])
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise ValueError("a canonical validation component crosses folds")
    return {component: next(iter(folds)) for component, folds in component_folds.items()}


def derive_folds(
    graph_path: Path,
    manifest_dir: Path,
    output: Path,
    receipt_path: Path,
    source_root: Path,
    executed_script: Path,
    preflight_path: Path,
) -> dict[str, object]:
    """Derive the canonical fold cache without requiring an unavailable prior cache."""
    if output.exists() or receipt_path.exists():
        raise FileExistsError("choose new fold-cache and receipt paths")
    source_identity = validate_graph_source(source_root, executed_script)
    preflight_identity = validate_input_preflight(preflight_path)
    graph_sha, graph_bytes = sha256_file(graph_path), graph_path.stat().st_size
    validate_graph_identity(graph_sha, graph_bytes)
    with np.load(graph_path, allow_pickle=False) as loaded:
        graph = {key: loaded[key] for key in loaded.files}
    required = {"split", "row_group", "next"}
    if not required <= graph.keys():
        raise ValueError(f"graph lacks fields: {sorted(required - graph.keys())}")
    splits = graph["split"]
    if splits.dtype.kind == "S":
        splits = np.char.decode(splits, "utf-8")
    groups = np.asarray(graph["row_group"])
    if (
        splits.ndim != 1
        or splits.dtype.kind != "U"
        or groups.ndim != 1
        or groups.dtype.kind not in "iu"
    ):
        raise ValueError("graph source identities have invalid dtype or shape")
    keys = list(zip(splits.tolist(), groups.astype(np.int64).tolist(), strict=True))
    if len(keys) != len(set(keys)) or any(s not in {"train", "valid"} or g < 0 for s, g in keys):
        raise ValueError("graph source-qualified identities are invalid or duplicated")
    components = graph_components(graph["next"])
    if len(components) != len(keys):
        raise ValueError("graph successor list is not aligned with source identities")
    graph_component = dict(zip(keys, components.tolist(), strict=True))
    expected_nodes = set(keys)
    canonical_nodes: dict[tuple[str, int], int] | None = None
    eval_by_fold: dict[int, set[int]] = {}
    manifest_records = []
    for fold, expected_hash in MANIFEST_SHA256.items():
        path = manifest_dir / f"fold{fold}.json"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_hash or len(payload) > 2_000_000:
            raise ValueError(f"canonical manifest fold{fold} identity mismatch")
        manifest = json.loads(payload)
        sources = manifest.get("sources", {})
        graph_source = sources.get("cache/chains.npz", {})
        folds_source = sources.get("cache/chain_folds.npz", {})
        if graph_source.get("sha256") != GRAPH_SHA256 or graph_source.get("bytes") != GRAPH_BYTES:
            raise ValueError(f"fold{fold} does not bind canonical graph identity")
        if folds_source.get("sha256") != ORIGINAL_FOLD_SHA256:
            raise ValueError(f"fold{fold} does not bind canonical original fold cache")
        if (
            manifest.get("version") != 1
            or manifest.get("fold") != fold
            or manifest.get("n_folds") != 4
        ):
            raise ValueError(f"fold{fold} manifest identity is malformed")
        nodes: dict[tuple[str, int], int] = {}
        seqs = manifest.get("sequences", {})
        if set(seqs) != {"train", "valid"}:
            raise ValueError(f"fold{fold} source-qualified nodes are incomplete")
        for split in ("train", "valid"):
            for row in seqs[split]:
                key = (split, int(row["row_group"]))
                if key in nodes:
                    raise ValueError(f"fold{fold} duplicates graph node {key}")
                nodes[key] = int(row["component"])
        if set(nodes) != expected_nodes:
            raise ValueError(f"fold{fold} graph-node coverage differs")
        for key, component in nodes.items():
            if graph_component[key] != component:
                raise ValueError(f"fold{fold} component differs at source-qualified node {key}")
        if canonical_nodes is None:
            canonical_nodes = nodes
        elif canonical_nodes != nodes:
            raise ValueError("canonical fold manifests disagree on graph nodes")
        evaluation = manifest.get("evaluation", {}).get("valid")
        if not isinstance(evaluation, list) or len(evaluation) != len(set(evaluation)):
            raise ValueError(f"fold{fold} validation membership is malformed")
        eval_by_fold[fold] = set(map(int, evaluation))
        manifest_records.append({"fold": fold, "sha256": digest, "bytes": len(payload)})
    valid_groups = {group for split, group in expected_nodes if split == "valid"}
    fold_by_component = fold_partition(graph_component, eval_by_fold, valid_groups)
    fold_values = np.asarray(
        [fold_by_component.get(graph_component[key], 0) for key in keys], dtype=np.int8
    )
    arrays = {
        "split": splits,
        "row_group": groups.astype(np.int64),
        "fold": fold_values,
        "chain": components.astype(np.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
    result: dict[str, object] = {
        "schema": "tabdata-l4-derived-fold-cache.v1",
        "status": "verified_from_canonical_main_manifests",
        "prior_fold_cache_used": False,
        "graph": {"path": str(graph_path.resolve()), "bytes": graph_bytes, "sha256": graph_sha},
        "graph_generation_source": source_identity,
        "input_preflight": preflight_identity,
        "canonical_manifests": manifest_records,
        "identity_scope": "graph_and_derived_fold_cache_artifacts_only",
        "derived_fields": list(arrays),
        "derived_fold_array_sha256": hashlib.sha256(fold_values.tobytes()).hexdigest(),
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "r4_expected_sha256": DERIVED_FOLD_SHA256,
        "identity_class": "r4_exact"
        if graph_sha == GRAPH_SHA256 and sha256_file(output) == DERIVED_FOLD_SHA256
        else "new_attempt_local_reencoded",
        "identity_mismatch_reasons": []
        if sha256_file(output) == DERIVED_FOLD_SHA256
        else ["derived fold cache archive encoding differs from the frozen R4 receipt"],
        "required_next_step_if_reencoded": "build fresh source-bound campaign split manifests and compare all memberships; do not claim R4 manifest identity",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Derive a new cache and write its source-identity receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--executed-script", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = derive_folds(
        args.graph,
        args.canonical_dir,
        args.output,
        args.receipt,
        args.source_root,
        args.executed_script,
        args.input_preflight,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
