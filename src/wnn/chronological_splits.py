"""Source-bound primary validation on the supplied train/validation boundary."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    connected_components,
    file_sha256,
)

_POLICY = "official_train_fit_full_official_valid_evaluation_v1"
_COMPONENT_METHOD = "undirected connected components of feature-only successor graph"
_CHRONOLOGY_BASIS = "official train/valid source boundary; calendar timestamps unavailable"
_SOURCE_NAMES = {
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
}
_SOURCE_IDENTITY_FIELDS = {
    "datasets/train.parquet": {"bytes", "mtime_ns", "footer_bytes", "footer_sha256"},
    "datasets/valid.parquet": {"bytes", "mtime_ns", "footer_bytes", "footer_sha256"},
    "cache/chains.npz": {"bytes", "mtime_ns", "sha256"},
    "cache/train_targets.npz": {"bytes", "mtime_ns", "sha256", "index_sha256"},
    "cache/valid_targets.npz": {"bytes", "mtime_ns", "sha256", "index_sha256"},
}
_MANIFEST_FIELDS = {
    "schema",
    "version",
    "policy",
    "chronology_basis",
    "component_method",
    "sources",
    "sequences",
    "fit",
    "evaluation",
    "excluded",
}


def _footer_identity(path: Path) -> dict[str, int | str]:
    """Return the bounded identity used for a parquet source file."""
    stat = path.stat()
    with path.open("rb") as stream:
        if stat.st_size < 8:
            raise ValueError(f"invalid parquet footer: {path}")
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > stat.st_size - 8:
            raise ValueError(f"invalid parquet footer: {path}")
        stream.seek(-8 - footer_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = footer_bytes + 8
        while remaining:
            chunk = stream.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError(f"truncated parquet footer: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "footer_bytes": footer_bytes,
        "footer_sha256": digest.hexdigest(),
    }


def _target_identity(path: Path) -> dict[str, int | str]:
    """Pin a small target index and its source-qualified row-group/sequence map."""
    stat = path.stat()
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as index:
        row_groups = np.asarray(index["row_group"])
        seq_ix = np.asarray(index["seq_ix"])
    if (
        row_groups.ndim != 1
        or seq_ix.shape != row_groups.shape
        or row_groups.dtype.kind not in "iu"
        or seq_ix.dtype.kind not in "iu"
        or len(np.unique(row_groups)) != len(row_groups)
    ):
        raise ValueError(f"invalid target index: {path}")
    for values in (row_groups, seq_ix):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
        "index_sha256": digest.hexdigest(),
    }


def _verify_source(root: Path, name: str, identity: dict[str, Any]) -> None:
    """Check one source identity without reading parquet row payloads."""
    source = root / name
    stat = source.stat()
    portable_registry = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if portable_registry:
        from wnn.portability import verify_portable_source

        verify_portable_source(source, name, identity, Path(portable_registry))
    elif stat.st_size != identity.get("bytes") or stat.st_mtime_ns != identity.get("mtime_ns"):
        raise ValueError(f"chronological primary source changed: {source}")
    expected_hash = identity.get("sha256")
    if expected_hash is not None and file_sha256(source) != expected_hash:
        raise ValueError(f"chronological primary source hash changed: {source}")
    if "footer_sha256" in identity and _footer_identity(source) != identity:
        raise ValueError(f"chronological parquet identity changed: {source}")
    if "index_sha256" in identity:
        current = _target_identity(source)
        if current.get("index_sha256") != identity["index_sha256"]:
            raise ValueError(f"chronological target index changed: {source}")


def _ordered_groups(value: object, name: str) -> tuple[int, ...]:
    """Validate a sorted, unique row-group list from manifest JSON."""
    if (
        not isinstance(value, list)
        or any(type(group) is not int or group < 0 for group in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{name} must be sorted unique nonnegative row groups")
    return tuple(value)


def _graph_members(
    root: Path,
) -> tuple[dict[tuple[str, int], int], tuple[tuple[str, int, int], ...]]:
    """Recover source-qualified components from the feature-only successor graph."""
    with np.load(root / "cache/chains.npz", allow_pickle=False) as graph:
        splits = np.asarray(graph["split"])
        row_groups = np.asarray(graph["row_group"])
        next_rows = np.asarray(graph["next"])
    if (
        splits.ndim != 1
        or row_groups.shape != splits.shape
        or next_rows.shape != splits.shape
        or row_groups.dtype.kind not in "iu"
        or any(str(value) not in ("train", "valid") for value in splits.tolist())
    ):
        raise ValueError("chronological graph has an invalid source-qualified index")
    recovered = connected_components(next_rows)
    nodes = tuple(
        (str(source), int(group), int(component))
        for source, group, component in zip(splits, row_groups, recovered, strict=True)
    )
    component_of = {(source, group): component for source, group, component in nodes}
    if len(component_of) != len(nodes) or any(group < 0 for _, group, _ in nodes):
        raise ValueError("chronological graph contains duplicate or negative sequence identities")
    return component_of, nodes


def _expected_membership(
    component_of: dict[tuple[str, int], int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Keep train components disjoint from every official validation component."""
    train_groups = sorted(group for source, group in component_of if source == "train")
    valid_groups = sorted(group for source, group in component_of if source == "valid")
    if not train_groups or not valid_groups:
        raise ValueError("official train and validation sources must both be nonempty")
    valid_components = {component_of["valid", group] for group in valid_groups}
    fit_train = tuple(
        group for group in train_groups if component_of["train", group] not in valid_components
    )
    excluded_train = tuple(group for group in train_groups if group not in fit_train)
    if not fit_train:
        raise ValueError("component exclusion leaves no official training sequences")
    return fit_train, tuple(valid_groups), excluded_train


def validate_official_source_coverage(root: Path, component_of: dict[tuple[str, int], int]) -> None:
    """Require graph and target-index row groups to cover both complete parquets."""
    for split in ("train", "valid"):
        with np.load(root / f"cache/{split}_targets.npz", allow_pickle=False) as index:
            groups = np.asarray(index["row_group"])
            seq_ix = np.asarray(index["seq_ix"])
        if (
            groups.ndim != 1
            or seq_ix.shape != groups.shape
            or groups.dtype.kind not in "iu"
            or seq_ix.dtype.kind not in "iu"
            or len(np.unique(groups)) != len(groups)
        ):
            raise ValueError(f"invalid {split} target index")
        graph_groups = {group for source, group in component_of if source == split}
        parquet_groups = set(
            range(pq.ParquetFile(root / f"datasets/{split}.parquet").num_row_groups)
        )
        if set(map(int, groups)) != graph_groups or graph_groups != parquet_groups:
            raise ValueError(f"target index, parquet and feature graph disagree for {split}")


def load_chronological_primary_split(
    path: Path, root: Path, raw: bytes, manifest: dict[str, Any]
) -> CampaignSplit:
    """Validate all source identities and recompute the mandatory membership."""
    if set(manifest) != _MANIFEST_FIELDS or manifest.get("version") != 1:
        raise ValueError("unsupported chronological primary manifest fields or version")
    if manifest.get("policy") != _POLICY:
        raise ValueError("unsupported chronological primary split policy")
    if manifest.get("component_method") != _COMPONENT_METHOD:
        raise ValueError("unsupported chronological component identity method")
    if manifest.get("chronology_basis") != _CHRONOLOGY_BASIS:
        raise ValueError("chronological primary manifest makes an unsupported chronology claim")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_NAMES:
        raise ValueError("chronological primary manifest must pin all official input sources")
    root = root.resolve()
    for name, identity in sources.items():
        if not isinstance(identity, dict) or set(identity) != _SOURCE_IDENTITY_FIELDS[name]:
            raise ValueError("chronological source identities must be objects")
        _verify_source(root, name, identity)
    component_of, nodes = _graph_members(root)
    validate_official_source_coverage(root, component_of)
    sequence_data = manifest.get("sequences")
    if not isinstance(sequence_data, dict) or set(sequence_data) != {"train", "valid"}:
        raise ValueError("chronological sequence index must include train and valid")
    listed_components: dict[tuple[str, int], int] = {}
    for split in ("train", "valid"):
        entries = sequence_data[split]
        if not isinstance(entries, list):
            raise ValueError("chronological sequence membership must be lists")
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"row_group", "component"}
                or type(entry["row_group"]) is not int
                or type(entry["component"]) is not int
                or entry["row_group"] < 0
                or entry["component"] < 0
            ):
                raise ValueError("invalid source-qualified sequence record")
            member = split, entry["row_group"]
            if member in listed_components:
                raise ValueError("chronological source-qualified sequence is duplicated")
            listed_components[member] = entry["component"]
    if listed_components != component_of:
        raise ValueError("chronological sequence identities disagree with the feature graph")
    fit_train, evaluation, excluded_train = _expected_membership(component_of)
    fit = manifest.get("fit")
    evaluation_manifest = manifest.get("evaluation")
    excluded = manifest.get("excluded")
    if (
        not isinstance(fit, dict)
        or set(fit) != {"train", "valid"}
        or _ordered_groups(fit["train"], "fit train") != fit_train
        or fit["valid"] != []
        or not isinstance(evaluation_manifest, dict)
        or set(evaluation_manifest) != {"valid"}
        or _ordered_groups(evaluation_manifest["valid"], "evaluation valid") != evaluation
        or not isinstance(excluded, dict)
        or set(excluded) != {"train_component_overlap"}
        or _ordered_groups(excluded["train_component_overlap"], "excluded train") != excluded_train
    ):
        raise ValueError("chronological primary membership differs from official source coverage")
    result = CampaignSplit(
        path.resolve(),
        hashlib.sha256(raw).hexdigest(),
        0,
        1,
        fit_train,
        (),
        evaluation,
        nodes,
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )
    result.assert_fit_groups(result.train_groups, result.valid_groups)
    if set(evaluation) != {group for source, group, _ in nodes if source == "valid"}:
        raise ValueError("primary evaluation does not cover every official validation sequence")
    return result
