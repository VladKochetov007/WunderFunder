"""Immutable disk-backed sequence caches for online relationship readouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from wnn.splits import file_sha256

CACHE_SCHEMA = "wnn.online-relationships.sequence-cache.v1"
ROWS = 20_000
RAW = slice(0, 112)
TARGET = slice(112, 114)
WEIGHT = 114
PARENT = slice(115, 117)
WIDTH = 117


class CachedOnlineSequence(NamedTuple):
    """Four arrays consumed by the canonical online relationship fitter."""

    raw: np.ndarray
    target: np.ndarray
    row_weight: np.ndarray
    parent_prediction: np.ndarray


class SequenceCache:
    """A hash-verified mmap and its immutable source-qualified row order."""

    def __init__(
        self, data_path: Path, metadata_path: Path, metadata: Mapping[str, object]
    ) -> None:
        self.data_path = data_path
        self.sidecar_path = metadata_path
        self.metadata = dict(metadata)
        self.row_refs = tuple(dict(ref) for ref in metadata["row_refs"])
        self.array = np.load(data_path, mmap_mode="r", allow_pickle=False)
        expected = (len(self.row_refs), ROWS, WIDTH)
        if self.array.shape != expected or self.array.dtype != np.dtype("<f4"):
            raise ValueError("online sequence cache NPY shape or dtype is not canonical")

    def iter_sequences(self) -> Iterator[CachedOnlineSequence]:
        """Yield one complete sequence at a time in the sidecar's exact order."""
        if self.metadata.get("scope") == "feature_only_eval":
            for index in range(len(self.row_refs)):
                row = self.array[index]
                if np.any(row[:, TARGET]) or np.any(row[:, WEIGHT]):
                    raise ValueError("evaluation cache contains target or weight values")
                yield _sequence_views(row)
            return
        for row in self.array:
            yield _sequence_views(row)

    def iter_with_refs(self) -> Iterator[tuple[dict[str, object], CachedOnlineSequence]]:
        """Yield each immutable source reference beside its sequence arrays."""
        yield from zip(self.row_refs, self.iter_sequences(), strict=True)


def load_sequence_cache(
    data_path: Path,
    metadata_path: Path,
    *,
    expected_scope: str,
    expected_recipe_sha256: str,
    expected_fold: int,
    verify_payload: bool = True,
) -> SequenceCache:
    """Validate the sidecar, NPY header, source identity and optionally payload hash."""
    metadata = _read_json(metadata_path)
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported online relationship sequence-cache schema")
    if metadata.get("scope") != expected_scope:
        raise ValueError("online sequence-cache scope does not match the consumer")
    if metadata.get("recipe_sha256") != expected_recipe_sha256:
        raise ValueError("online sequence cache belongs to a different recipe")
    if metadata.get("fold") != expected_fold:
        raise ValueError("online sequence cache belongs to a different fold")
    if metadata.get("data_file") != data_path.name:
        raise ValueError("online sequence-cache filename differs from its sidecar")
    refs = metadata.get("row_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("online sequence cache has no ordered row references")
    if metadata.get("membership_sha256") != row_refs_sha256(refs):
        raise ValueError("online sequence-cache membership digest is invalid")
    read_contract = metadata.get("read_contract")
    if expected_scope == "feature_only_eval":
        if read_contract != {"targets": False, "scored_mask": False}:
            raise ValueError("evaluation cache does not prove target-free feature reads")
        if metadata.get("zero_columns") != [112, 113, 114]:
            raise ValueError("evaluation cache must declare target and weight columns as zero")
    elif expected_scope == "in_sample_fit" and read_contract != {
        "targets": True,
        "scored_mask": True,
    }:
        raise ValueError("fit cache did not read its canonical labels and score masks")
    if metadata.get("data_sha256") is None or not data_path.is_file():
        raise FileNotFoundError("online sequence cache payload is missing")
    if verify_payload and file_sha256(data_path) != metadata["data_sha256"]:
        raise ValueError("online sequence-cache payload hash changed")
    cache = SequenceCache(data_path, metadata_path, metadata)
    if metadata.get("shape") != list(cache.array.shape) or metadata.get("dtype") != "<f4":
        raise ValueError("online sequence-cache header differs from the sidecar")
    return cache


def row_refs_sha256(row_refs: list[Mapping[str, object]]) -> str:
    """Hash an exact ordered list of source-qualified row groups."""
    normalized = [
        {"split": str(ref["split"]), "row_group": int(ref["row_group"])} for ref in row_refs
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sequence_cache_paths(directory: Path, scope: str) -> tuple[Path, Path]:
    """Return the stable payload and sidecar names for one cache scope."""
    if scope == "in_sample_fit":
        stem = "fit_sequences"
    elif scope == "feature_only_eval":
        stem = "evaluation_sequences"
    else:
        raise ValueError(f"unknown online sequence-cache scope: {scope}")
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def _sequence_views(row: np.ndarray) -> CachedOnlineSequence:
    return CachedOnlineSequence(row[:, RAW], row[:, TARGET], row[:, WEIGHT], row[:, PARENT])


def _read_json(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    if size <= 0 or size > 5_000_000:
        raise ValueError(f"online sequence-cache sidecar size is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"online sequence-cache sidecar must be a JSON object: {path}")
    return value
