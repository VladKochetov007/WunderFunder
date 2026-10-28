"""Build immutable manifests for the official train/validation source boundary."""

from __future__ import annotations

import json
from pathlib import Path

from wnn.chronological_source_index import (
    SOURCE_AUDIT_PATH,
    _load_source_audit,
    validate_source_index_receipt,
)
from wnn.chronological_splits import (
    _CHRONOLOGY_BASIS,
    _COMPONENT_METHOD,
    _POLICY,
    _expected_membership,
    _footer_identity,
    _graph_members,
    _target_identity,
    load_chronological_primary_split,
    validate_official_source_coverage,
)
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, file_sha256


def _source_identities(root: Path) -> dict[str, dict[str, int | str]]:
    """Fingerprint official parquets, source audit, graph, and thin indexes."""
    sources = {
        name: _footer_identity(root / name)
        for name in ("datasets/train.parquet", "datasets/valid.parquet")
    }
    _, source_audit_sha256 = _load_source_audit(root)
    source_audit_path = root / SOURCE_AUDIT_PATH
    source_audit_stat = source_audit_path.stat()
    sources[SOURCE_AUDIT_PATH.as_posix()] = {
        "bytes": source_audit_stat.st_size,
        "mtime_ns": source_audit_stat.st_mtime_ns,
        "sha256": source_audit_sha256,
    }
    for split in ("train", "valid"):
        validate_source_index_receipt(root, split, source_audit_sha256)
        receipt_path = root / f"cache/{split}_index.identity.json"
        receipt_stat = receipt_path.stat()
        sources[receipt_path.relative_to(root).as_posix()] = {
            "bytes": receipt_stat.st_size,
            "mtime_ns": receipt_stat.st_mtime_ns,
            "sha256": file_sha256(receipt_path),
        }
    graph_path = root / "cache/chronological_chains.npz"
    graph_stat = graph_path.stat()
    sources["cache/chronological_chains.npz"] = {
        "bytes": graph_stat.st_size,
        "mtime_ns": graph_stat.st_mtime_ns,
        "sha256": file_sha256(graph_path),
    }
    for split in ("train", "valid"):
        sources[f"cache/{split}_index.npz"] = _target_identity(root / f"cache/{split}_index.npz")
    return sources


def build_chronological_primary_manifest(root: Path, output: Path) -> Path:
    """Freeze train-only fitting and complete official validation membership.

    The chronology basis is the supplied source boundary. Current parquet files
    do not provide usable timestamps, so this builder never sorts by ``seq_ix``.
    """
    root = root.resolve()
    sources = _source_identities(root)
    component_of, nodes = _graph_members(root)
    validate_official_source_coverage(root, component_of)
    fit_train, evaluation, excluded_train = _expected_membership(component_of)
    manifest = {
        "schema": CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
        "version": 1,
        "policy": _POLICY,
        "chronology_basis": _CHRONOLOGY_BASIS,
        "component_method": _COMPONENT_METHOD,
        "sources": sources,
        "sequences": {
            split: [
                {"row_group": group, "component": component}
                for source, group, component in nodes
                if source == split
            ]
            for split in ("train", "valid")
        },
        "fit": {"train": list(fit_train), "valid": []},
        "evaluation": {"valid": list(evaluation)},
        "excluded": {"train_component_overlap": list(excluded_train)},
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace frozen manifest: {output}")
    else:
        with output.open("xb") as stream:
            stream.write(encoded)
    load_chronological_primary_split(output, root, encoded, manifest)
    return output
