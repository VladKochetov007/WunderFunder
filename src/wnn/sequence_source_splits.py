"""Freeze train-only fits and full-valid evaluation without calendar assumptions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from wnn.sequence_inner_splits import _json_equal, _publish_exclusive
from wnn.sequence_source_inputs import (
    GRAPH,
    INDEX,
    METHOD,
    RECEIPT,
    prepare_source_inputs,
    sequence_ids,
    source_identity,
    validate_input_receipt,
    verify_identity,
)
from wnn.splits import (
    MAX_CAMPAIGN_SPLIT_BYTES,
    SEQUENCE_SOURCE_SPLIT_SCHEMA,
    CampaignSplit,
    connected_components,
)

# Any is limited to heterogeneous untrusted JSON at the manifest boundary.
POLICY = "train-source-only/full-valid-source/component-exclusion-v1"
FOLD_ROLE = "single-source-holdout-protocol-id-not-cross-validation"


def _graph_nodes(root: Path) -> tuple[tuple[str, int, int], ...]:
    with np.load(root / GRAPH, allow_pickle=False) as graph:
        source, group, successor = graph["split"], graph["row_group"], graph["next"]
    if (
        source.ndim != 1
        or group.ndim != 1
        or successor.ndim != 1
        or source.dtype.kind != "U"
        or group.dtype.kind not in "iu"
        or len(source) != len(group)
        or len(group) != len(successor)
        or not len(group)
    ):
        raise ValueError("invalid source-qualified graph arrays")
    components = connected_components(successor)
    nodes = tuple(
        (str(split), int(row_group), int(component))
        for split, row_group, component in zip(source, group, components, strict=True)
    )
    if any(split not in INDEX or row_group < 0 for split, row_group, _ in nodes):
        raise ValueError("unknown source or row group in feature graph")
    if len({(split, row_group) for split, row_group, _ in nodes}) != len(nodes):
        raise ValueError("duplicate source-qualified graph row group")
    return nodes


def _coverage(root: Path, nodes: tuple[tuple[str, int, int], ...]) -> dict[str, np.ndarray]:
    ids = {}
    for source in INDEX:
        reader = pq.ParquetFile(root / f"datasets/{source}.parquet")
        ids[source] = sequence_ids(reader)
        with np.load(root / INDEX[source], allow_pickle=False) as index:
            groups, sequences = index["row_group"], index["seq_ix"]
        if (
            groups.ndim != 1
            or sequences.ndim != 1
            or groups.dtype.kind not in "iu"
            or sequences.dtype.kind not in "iu"
            or len(groups) != len(sequences)
            or len(set(groups.tolist())) != len(groups)
            or set(groups.tolist()) != set(range(len(ids[source])))
        ):
            raise ValueError(f"index must cover every physical {source} row group exactly once")
        order = np.argsort(groups)
        if not np.array_equal(sequences[order], ids[source]):
            raise ValueError(f"index seq_ix disagrees with physical {source} row groups")
        actual = {group for split, group, _ in nodes if split == source}
        if actual != set(range(len(ids[source]))):
            raise ValueError(f"graph must cover every physical {source} row group exactly once")
    return ids


def _snapshot(root: Path) -> dict[str, Any]:
    sources = validate_input_receipt(root)
    receipt_identity = source_identity(root / RECEIPT)
    nodes = _graph_nodes(root)
    ids = _coverage(root, nodes)
    held_components = {component for source, _, component in nodes if source == "valid"}
    training = sorted(
        group
        for source, group, component in nodes
        if source == "train" and component not in held_components
    )
    excluded = sorted(
        group
        for source, group, component in nodes
        if source == "train" and component in held_components
    )
    if not training:
        raise ValueError("source holdout leaves no eligible train sequences")
    for name, identity in sources.items():
        verify_identity(root, name, identity)
    if source_identity(root / RECEIPT) != receipt_identity:
        raise ValueError("source input receipt changed during split validation")
    return {
        "schema": SEQUENCE_SOURCE_SPLIT_SCHEMA,
        "version": 1,
        "policy": POLICY,
        "fold": 0,
        "n_folds": 1,
        "fold_role": FOLD_ROLE,
        "chronology": "source-separated; calendar ordering is unverified",
        "component_method": METHOD,
        "source_paths": {"graph": GRAPH, "indexes": INDEX, "receipt": RECEIPT},
        "sources": {**sources, RECEIPT: receipt_identity},
        "sequences": {
            source: [
                {"row_group": group, "seq_ix": int(ids[source][group]), "component": component}
                for split, group, component in sorted(nodes)
                if split == source
            ]
            for source in INDEX
        },
        "fit": {"train": training, "valid": []},
        "evaluation": {"valid": list(range(len(ids["valid"])))},
        "excluded": {"train_component_overlap": excluded},
    }


def load_source_campaign_split(
    path: Path, root: Path, raw: bytes, manifest: dict[str, Any]
) -> CampaignSplit:
    """Recompute full coverage and components; never consult old random fold caches.

    Local parquet immutability is checked with full-file SHA256, size, mtime and
    footer. Relocation uses the full-byte portable registry.
    This graph detects feature adjacency, not every possible latent dependency.
    """
    expected = _snapshot(root)
    if set(manifest) != set(expected) or not isinstance(manifest.get("sources"), dict):
        raise ValueError("invalid sequence source holdout manifest schema")
    if set(manifest["sources"]) != set(expected["sources"]):
        raise ValueError("source holdout source identities are incomplete")
    for name, identity in manifest["sources"].items():
        verify_identity(root, name, identity)
        if name != RECEIPT and not _json_equal(identity, expected["sources"][name]):
            raise ValueError("source holdout identity disagrees with sealed input receipt")
    for key in expected.keys() - {"sources"}:
        if not _json_equal(manifest[key], expected[key]):
            raise ValueError(f"source holdout {key} differs from complete graph/source membership")
    nodes = tuple(
        (source, node["row_group"], node["component"])
        for source, entries in expected["sequences"].items()
        for node in entries
    )
    result = CampaignSplit(
        path.resolve(),
        hashlib.sha256(raw).hexdigest(),
        0,
        1,
        tuple(expected["fit"]["train"]),
        (),
        tuple(expected["evaluation"]["valid"]),
        nodes,
    )
    result.assert_fit_groups(result.train_groups, result.valid_groups)
    return result


def prepare_sequence_source_split(data_root: Path, output: Path, workers: int = 2) -> CampaignSplit:
    """Prepare full feature-only caches and publish one immutable source holdout.

    Args:
        data_root: Receiver root with complete train.parquet and valid.parquet.
        output: New immutable JSON path; existing manifests are never replaced.
        workers: Feature-read worker count; defaults to two bounded shared readers.

    Returns:
        Validated source-qualified membership with empty fitting valid groups.

    Raises:
        ValueError: Source, index, graph or complete-sequence coverage disagrees.
        FileExistsError: Output exists or a previous input preparation is partial.
    """
    data_root, output = data_root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace source holdout manifest: {output}")
    prepare_source_inputs(data_root, workers)
    manifest = _snapshot(data_root)
    raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_CAMPAIGN_SPLIT_BYTES:
        raise ValueError("source holdout manifest exceeds bounded read size")
    result = load_source_campaign_split(output, data_root, raw, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_exclusive(output, raw)
    return result
