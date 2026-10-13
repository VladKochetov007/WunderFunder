"""Validation, hashing, and small metric helpers for the member-bank grader."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from scripts.alias_grade_prediction import MAX_NPY_HEADER_BYTES, _npy_header

EXPECTED_ROW_GROUPS = 1873
N_OUTPUTS = 2
MAX_JSON_BYTES = 16 * 1_048_576
REQUIRED_DEPENDENCIES = frozenset(
    {"exporter", "exporter_helper", "wnn.data", "wnn.features", "wnn.nn"}
)
TARGET_COLUMNS = ("t0", "t1")


class HashState(Protocol):
    """The incremental interface shared by hashlib digest objects."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None: ...

    def hexdigest(self) -> str: ...


class CovarianceAccumulator:
    """Merge unweighted centered cross-products without retaining scored rows."""

    def __init__(self, width: int) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.cross = np.zeros((width, width), dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        """Add one row block to the centered cross-product matrix."""
        block = np.asarray(values, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != self.mean.size:
            raise ValueError("correlation block has an unexpected shape")
        if not np.isfinite(block).all():
            raise ValueError("correlation block contains non-finite values")
        count = int(block.shape[0])
        if count == 0:
            return
        block_mean = block.mean(axis=0)
        centered = block - block_mean
        block_cross = centered.T @ centered
        if self.count == 0:
            self.count = count
            self.mean = block_mean
            self.cross = block_cross
            return
        total = self.count + count
        delta = block_mean - self.mean
        self.cross += block_cross + np.outer(delta, delta) * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def correlation(self) -> np.ndarray:
        """Return Pearson correlations; constant columns receive zero."""
        variance = np.maximum(np.diag(self.cross), 0.0)
        denominator = np.sqrt(np.outer(variance, variance))
        result = np.divide(
            self.cross,
            denominator,
            out=np.zeros_like(self.cross),
            where=denominator > 0.0,
        )
        return np.clip(result, -1.0, 1.0)


def blend_predictions(
    predictions: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    """Blend raw FP32 member predictions before official metric clipping."""
    if not weights:
        raise ValueError("mixture weights cannot be empty")
    first_id = next(iter(weights))
    if first_id not in predictions:
        raise ValueError(f"mixture references missing member {first_id}")
    first = np.asarray(predictions[first_id])
    if first.dtype != np.float32 or first.ndim != 2 or first.shape[1] != N_OUTPUTS:
        raise ValueError("member predictions must be float32 (rows, 2)")
    blend = np.zeros(first.shape, dtype=np.float32)
    for member_id, coefficient in weights.items():
        if member_id not in predictions:
            raise ValueError(f"mixture references missing member {member_id}")
        value = np.asarray(predictions[member_id])
        if value.dtype != np.float32 or value.shape != first.shape:
            raise ValueError("mixture members must share one float32 prediction shape")
        blend += np.float32(coefficient) * value
    if not np.isfinite(blend).all():
        raise ValueError("fixed mixture produced non-finite predictions")
    return blend


def _as_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _as_sha256(value: object, field: str) -> str:
    digest = _as_string(value, field)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def digest_file(path: Path) -> str:
    """Hash a file with bounded memory and release its clean cache pages."""
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value


def stable_file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    """Compare file identity and content-version metadata without atime."""
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def read_bounded_bytes(path: Path, max_bytes: int = MAX_JSON_BYTES) -> tuple[bytes, os.stat_result]:
    """Read a bounded metadata file and reject changes during the read."""
    with path.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if opened_stat.st_size > max_bytes:
            raise ValueError(f"input exceeds the {max_bytes}-byte limit: {path}")
        data = stream.read(max_bytes + 1)
        if len(data) > max_bytes or len(data) != opened_stat.st_size:
            raise ValueError(f"input changed size or exceeds the {max_bytes}-byte limit: {path}")
    if stable_file_identity(path.stat()) != stable_file_identity(opened_stat):
        raise ValueError(f"input changed while it was read: {path}")
    return data, opened_stat


def _read_json(path: Path) -> dict[str, object]:
    try:
        data, _ = read_bounded_bytes(path)
        return _as_object(json.loads(data), str(path))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc


def _plan_records(
    plan: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    raw_members = plan.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 14:
        raise ValueError("frozen member plan must contain 14 unique exports")
    members = [_as_object(item, "plan member") for item in raw_members]
    identifiers = [_as_string(item.get("id"), "member.id") for item in members]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("member plan has duplicate IDs")
    weight_hashes = [_as_sha256(item.get("sha256"), "member.sha256") for item in members]
    if len(set(weight_hashes)) != len(weight_hashes):
        raise ValueError("member plan is not deduplicated by original NPZ hash")

    raw_mixtures = _as_object(plan.get("mixtures"), "plan.mixtures")
    if len(raw_mixtures) != 9:
        raise ValueError("frozen member plan must contain nine fixed mixtures")
    known_members = set(identifiers)
    mixtures: dict[str, dict[str, float]] = {}
    for name, raw_weights in raw_mixtures.items():
        if not name or name in known_members:
            raise ValueError("mixture names must be non-empty and distinct from member IDs")
        weight_object = _as_object(raw_weights, f"mixture {name}")
        if not weight_object or not set(weight_object).issubset(known_members):
            raise ValueError(f"mixture {name} is empty or references an unknown member")
        weights: dict[str, float] = {}
        for member_id, raw_value in weight_object.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"mixture {name} has a non-numeric coefficient")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"mixture {name} has an invalid coefficient")
            weights[member_id] = value
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"mixture {name} coefficients must sum to one")
        mixtures[name] = weights
    return members, mixtures


def _load_prediction_map(path: Path, shape: tuple[int, int, int]) -> tuple[np.memmap, HashState]:
    """Bound and validate the NPY prefix before mapping its prediction payload."""
    source_identity = stable_file_identity(path.stat())
    header_shape, header_dtype, fortran_order, offset = _npy_header(path)
    if stable_file_identity(path.stat()) != source_identity:
        raise ValueError(f"prediction file changed while its header was read: {path}")
    if header_shape != shape or header_dtype != np.dtype(np.float32):
        raise ValueError(f"prediction NPY shape/dtype mismatch: {path}")
    if fortran_order:
        raise ValueError(f"prediction NPY must use C order: {path}")
    if offset < 1 or offset > MAX_NPY_HEADER_BYTES + 16:
        raise ValueError(f"prediction NPY header offset is outside the bound: {path}")
    payload_nbytes = math.prod(shape) * np.dtype(np.float32).itemsize
    stat = path.stat()
    if stable_file_identity(stat) != source_identity:
        raise ValueError(f"prediction file changed before mapping: {path}")
    if stat.st_size != offset + payload_nbytes:
        raise ValueError(f"prediction NPY has truncation or trailing bytes: {path}")
    value = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=shape, order="C")
    if not value.flags.c_contiguous or value.flags.writeable:
        _close_prediction_map(value)
        raise ValueError(f"prediction NPY must be read-only C-order: {path}")
    if stable_file_identity(path.stat()) != source_identity:
        _close_prediction_map(value)
        raise ValueError(f"prediction file changed while it was mapped: {path}")
    with path.open("rb") as stream:
        header = stream.read(offset)
    if len(header) != offset:
        _close_prediction_map(value)
        raise ValueError(f"prediction NPY header is truncated: {path}")
    if stable_file_identity(path.stat()) != source_identity:
        _close_prediction_map(value)
        raise ValueError(f"prediction file changed while its header digest was read: {path}")
    return value, hashlib.sha256(header)


def _close_prediction_map(value: np.memmap) -> None:
    """Close a newly opened mapping when its bounded header fails validation."""
    mapped = getattr(value, "_mmap", None)
    if mapped is not None:
        mapped.close()


def _validate_dependencies(result: dict[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    raw = _as_object(result.get("source_dependencies"), "result.source_dependencies")
    if set(raw) != REQUIRED_DEPENDENCIES:
        raise ValueError("export result has an incomplete source dependency manifest")
    digests: dict[str, str] = {}
    paths: dict[str, str] = {}
    for name in sorted(REQUIRED_DEPENDENCIES):
        dependency = _as_object(raw[name], f"source dependency {name}")
        digest = _as_sha256(dependency.get("sha256"), f"{name}.sha256")
        path = _as_string(dependency.get("path"), f"{name}.path")
        digests[name] = digest
        paths[name] = path
    exporter_digest = _as_sha256(result.get("exporter_sha256"), "exporter_sha256")
    if digests["exporter"] != exporter_digest:
        raise ValueError("exporter_sha256 disagrees with the source dependency manifest")
    return digests, paths
