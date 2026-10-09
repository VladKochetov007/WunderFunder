"""Frozen evaluation splits (research/contract.yaml).

The holdout is defined by a hash of ``seq_ix`` so it does not depend on file
order or on any experiment's permutation seed. Nothing may be fitted, tuned or
probed on holdout sequences; they are read only by the grader.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow.parquet as pq

HOLDOUT_MODULUS = 5


@dataclass(frozen=True)
class VerifiedParsedCacheSources:
    """Raw input identities already validated through a pinned parsed cache."""

    split_manifest_sha256: str
    cache_manifest_sha256: str
    validation_receipt_sha256: str
    source_verification_sha256: str
    sources: dict[str, dict[str, Any]]


def is_holdout(seq_ix: int) -> bool:
    return int(hashlib.sha256(str(int(seq_ix)).encode()).hexdigest(), 16) % HOLDOUT_MODULUS == 0


def valid_row_groups(path) -> tuple[np.ndarray, np.ndarray]:
    """(dev_row_groups, holdout_row_groups) for a validation parquet file."""
    pf = pq.ParquetFile(path)
    seq = pf.read(columns=["seq_ix"]).column(0).to_numpy()[::20_000]
    hold = np.array([is_holdout(s) for s in seq])
    groups = np.arange(pf.num_row_groups)
    return groups[~hold], groups[hold]


def dev_halves(path) -> tuple[np.ndarray, np.ndarray]:
    """Split dev row groups in two by a salted seq_ix hash (for fine-tune-on-A, score-on-B)."""
    pf = pq.ParquetFile(path)
    seq = pf.read(columns=["seq_ix"]).column(0).to_numpy()[::20_000]
    dev, _ = valid_row_groups(path)
    half_a = np.array(
        [
            int(hashlib.sha256(f"devhalf:{int(seq[g])}".encode()).hexdigest(), 16) % 2 == 0
            for g in dev
        ]
    )
    return dev[half_a], dev[~half_a]


def file_sha256(path: Path) -> str:
    """Hash a source or checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connected_components(successors: np.ndarray) -> np.ndarray:
    """Return deterministic undirected components, including successor cycles."""
    successors = np.asarray(successors)
    if successors.ndim != 1 or successors.dtype.kind not in "iu":
        raise ValueError("successors must be a one-dimensional integer array")
    if np.any(successors < -1) or np.any(successors >= len(successors)):
        raise ValueError("successor index outside the graph")
    parents = np.arange(len(successors))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for index, successor in enumerate(successors):
        if successor >= 0:
            left, right = root(index), root(int(successor))
            parents[max(left, right)] = min(left, right)
    roots = np.array([root(index) for index in range(len(successors))])
    return np.unique(roots, return_inverse=True)[1]


class CampaignSplit(NamedTuple):
    """Immutable row membership frozen before any outer-fold fitting."""

    path: Path
    sha256: str
    fold: int
    n_folds: int
    train_groups: tuple[int, ...]
    valid_groups: tuple[int, ...]
    evaluation_groups: tuple[int, ...]
    components: tuple[tuple[str, int, int], ...]

    def assert_fit_groups(self, train: Iterable[int], valid: Iterable[int]) -> None:
        """Reject unknown, excluded or evaluation-linked fitting rows."""
        training, validation = set(map(int, train)), set(map(int, valid))
        if not training.issubset(self.train_groups) or not validation.issubset(self.valid_groups):
            raise ValueError("fitting rows are outside the campaign manifest")
        component_of = {(split, group): component for split, group, component in self.components}
        fitted = {component_of[("train", group)] for group in training}
        fitted.update(component_of[("valid", group)] for group in validation)
        evaluated = {component_of[("valid", group)] for group in self.evaluation_groups}
        if fitted & evaluated:
            raise ValueError("fit/evaluation component overlap")

    def assert_complete_component_subset(
        self,
        train: Iterable[int],
        valid: Iterable[int],
        selected_components: Iterable[int],
    ) -> None:
        """Require cached fit rows to include every manifest row in selected components."""
        training, validation = set(map(int, train)), set(map(int, valid))
        selected = set(map(int, selected_components))
        self.assert_fit_groups(training, validation)
        component_of = {(split, group): component for split, group, component in self.components}
        observed = {component_of[("train", group)] for group in training}
        observed.update(component_of[("valid", group)] for group in validation)
        if not selected or observed != selected:
            raise ValueError("cached fit rows disagree with their selected components")
        expected_train = {
            group for group in self.train_groups if component_of[("train", group)] in selected
        }
        expected_valid = {
            group for group in self.valid_groups if component_of[("valid", group)] in selected
        }
        if training != expected_train or validation != expected_valid:
            raise ValueError("cached fit membership is not complete by component")


def load_campaign_split(
    path: Path,
    root: Path,
    *,
    preverified_sources: VerifiedParsedCacheSources | None = None,
) -> CampaignSplit:
    """Validate frozen source identities and return component-disjoint membership.

    Dataset identities use file size, nanosecond mtime and a recorded footer hash;
    the footer hash is generated once, while fitting checks the immutable file
    metadata. Small chain caches are also hashed on every read. An explicit
    ``preverified_sources`` may skip filesystem verification only when a parsed
    cache reader has validated a pinned cache manifest, source receipt, and
    array-validation receipt bound to this exact split. Those source files are
    not consumed after the cache is opened.
    """
    path = path.resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != 1:
        raise ValueError("unsupported campaign split version")
    if preverified_sources is not None:
        pin_fields = (
            preverified_sources.split_manifest_sha256,
            preverified_sources.cache_manifest_sha256,
            preverified_sources.validation_receipt_sha256,
            preverified_sources.source_verification_sha256,
        )
        if hashlib.sha256(raw).hexdigest() != preverified_sources.split_manifest_sha256 or any(
            len(pin) != 64 or any(char not in "0123456789abcdef" for char in pin)
            for pin in pin_fields
        ):
            raise ValueError("preverified cache receipt is not pinned to this split")
        if set(preverified_sources.sources) != {
            "datasets/train.parquet",
            "datasets/valid.parquet",
        }:
            raise ValueError("preverified cache receipt must bind both dataset Parquet sources")
    auxiliary_receipt = os.environ.get("WNN_FOLD0_ASSET_VALIDATION_RECEIPT")
    auxiliary_receipt_sha256 = os.environ.get("WNN_FOLD0_ASSET_VALIDATION_SHA256")
    preverified_auxiliary_sources: frozenset[str] = frozenset()
    if auxiliary_receipt or auxiliary_receipt_sha256:
        if not auxiliary_receipt or not auxiliary_receipt_sha256 or preverified_sources is None:
            raise ValueError("fold-0 auxiliary verification requires a parsed cache and both receipt pins")
        from wnn.fold0_assets import AUXILIARY_SOURCE_PATHS, validate_preverified_fold0_assets

        portable_registry = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
        if not portable_registry:
            raise ValueError("fold-0 auxiliary verification requires the pinned portable registry")
        validate_preverified_fold0_assets(
            root,
            path,
            Path(portable_registry),
            Path(auxiliary_receipt),
            auxiliary_receipt_sha256,
        )
        preverified_auxiliary_sources = AUXILIARY_SOURCE_PATHS
    for name, identity in manifest["sources"].items():
        source = root / name
        preverified = (
            preverified_sources.sources.get(name) if preverified_sources is not None else None
        )
        if preverified is not None:
            digest = preverified.get("sha256")
            if (
                name not in {"datasets/train.parquet", "datasets/valid.parquet"}
                or preverified.get("bytes") != identity.get("bytes")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"preverified source receipt disagrees with split: {name}")
            continue
        if name not in preverified_auxiliary_sources:
            stat = source.stat()
            portable_registry = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
            if portable_registry:
                from wnn.portability import verify_portable_source

                verify_portable_source(source, name, identity, Path(portable_registry))
            elif stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
                raise ValueError(f"campaign source changed: {source}")
            if "sha256" in identity and file_sha256(source) != identity["sha256"]:
                raise ValueError(f"campaign source hash changed: {source}")
        if "index_sha256" in identity:
            digest = hashlib.sha256()
            with np.load(source, allow_pickle=False) as index:
                for field in ("row_group", "seq_ix"):
                    digest.update(np.asarray(index[field], dtype="<i8").tobytes())
            if digest.hexdigest() != identity["index_sha256"]:
                raise ValueError(f"campaign target index changed: {source}")
    nodes = tuple(
        (split, int(node["row_group"]), int(node["component"]))
        for split, sequences in manifest["sequences"].items()
        for node in sequences
    )
    if len({(split, group) for split, group, _ in nodes}) != len(nodes):
        raise ValueError("duplicate sequence in campaign manifest")
    if any(
        split not in ("train", "valid") or group < 0 or component < 0
        for split, group, component in nodes
    ):
        raise ValueError("invalid campaign sequence identity")
    groups = [
        tuple(map(int, manifest[section][split]))
        for section, split in (("fit", "train"), ("fit", "valid"), ("evaluation", "valid"))
    ]
    if any(len(set(group)) != len(group) for group in groups) or not all(groups):
        raise ValueError("campaign groups must be unique and nonempty")
    fold, n_folds = int(manifest["fold"]), int(manifest["n_folds"])
    if not 0 <= fold < n_folds:
        raise ValueError("invalid campaign fold")
    component_of = {(split, group): component for split, group, component in nodes}
    with np.load(root / "cache/chains.npz", allow_pickle=False) as graph:
        recovered = connected_components(graph["next"])
        actual_components = {
            (str(split), int(group)): int(component)
            for split, group, component in zip(
                graph["split"], graph["row_group"], recovered, strict=True
            )
        }
    if component_of != actual_components:
        raise ValueError("campaign components disagree with the feature-only graph")
    with np.load(root / "cache/chain_folds.npz", allow_pickle=False) as cached:
        expected_evaluation = {
            int(group)
            for split, group, assigned in zip(
                cached["split"], cached["row_group"], cached["fold"], strict=True
            )
            if split == "valid" and assigned == fold
        }
    if set(groups[2]) != expected_evaluation:
        raise ValueError("campaign evaluation membership disagrees with the cached fold")
    for split, selected in (("train", groups[0]), ("valid", groups[1]), ("valid", groups[2])):
        if any((split, group) not in component_of for group in selected):
            raise ValueError("campaign membership names an unknown sequence")
    result = CampaignSplit(path, hashlib.sha256(raw).hexdigest(), fold, n_folds, *groups, nodes)
    result.assert_fit_groups(result.train_groups, result.valid_groups)
    evaluated_components = {component_of[("valid", group)] for group in result.evaluation_groups}
    for split, fitting in (("train", result.train_groups), ("valid", result.valid_groups)):
        expected = {
            group
            for (source_split, group), component in component_of.items()
            if source_split == split and component not in evaluated_components
        }
        if set(fitting) != expected:
            raise ValueError("campaign fitting membership omits eligible graph sequences")
    return result
