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
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq

HOLDOUT_MODULUS = 5
MAX_CAMPAIGN_SPLIT_BYTES = 2_000_000


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


def load_campaign_split(path: Path, root: Path) -> CampaignSplit:
    """Validate frozen source identities and return component-disjoint membership.

    Dataset identities use file size, nanosecond mtime and a recorded footer hash;
    the footer hash is generated once, while fitting checks the immutable file
    metadata. Small chain caches are also hashed on every read.
    """
    path = path.resolve()
    size = path.stat().st_size
    if size <= 0 or size > MAX_CAMPAIGN_SPLIT_BYTES:
        raise ValueError(
            f"campaign split manifest exceeds its {MAX_CAMPAIGN_SPLIT_BYTES}-byte read bound"
        )
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("campaign split manifest changed while being read")
    manifest = json.loads(raw)
    if manifest.get("version") != 1:
        raise ValueError("unsupported campaign split version")
    for name, identity in manifest["sources"].items():
        source = root / name
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
