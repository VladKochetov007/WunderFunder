from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# One canonical row group is the fixed upper bound; members are never stacked.
NORMALIZATION_CHUNK_ROWS = 20_000
NORMALIZATION_STD_FLOOR = 0.001
MAX_CACHE_BYTES = 64_000
ALGORITHM = {
    "name": "float64_population_moments",
    "version": 1,
    "accumulation": "sequence_order_sum_and_square",
    "chunk_rows": NORMALIZATION_CHUNK_ROWS,
    "std_floor": NORMALIZATION_STD_FLOOR,
}


@dataclass(frozen=True)
class NormalizationMoments:
    """Resolved FP32 model moments plus the identities used to produce them."""

    mean: np.ndarray
    std: np.ndarray
    identity_sha256: str
    moments_sha256: str
    sequence_count: int


def _canonical_json(value: object) -> bytes:
    """Serialize cache identity data with a stable JSON representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def normalization_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Hash the exact source, split, scope, and member identity."""
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def moments_sha256(mean: np.ndarray, std: np.ndarray) -> str:
    """Hash canonical little-endian FP32 means and scales."""
    mean32 = np.ascontiguousarray(mean, dtype="<f4")
    std32 = np.ascontiguousarray(std, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(mean32.tobytes(order="C"))
    digest.update(std32.tobytes(order="C"))
    return digest.hexdigest()


def compute_population_moments(
    members: Sequence[tuple[str, int]],
    reader: Callable[[str, int], np.ndarray],
    *,
    feature_count: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate population moments from one source-qualified sequence at a time."""
    if not members or feature_count < 1:
        raise ValueError("normalization requires fitting members and a positive feature width")
    total = np.zeros(feature_count, dtype=np.float64)
    square = np.zeros(feature_count, dtype=np.float64)
    count = 0
    seen: set[tuple[str, int]] = set()
    for source, group in members:
        member = (source, int(group))
        if not isinstance(source, str) or not source or member[1] < 0 or member in seen:
            raise ValueError("normalization members must be unique source-qualified groups")
        seen.add(member)
        features = np.asarray(reader(source, member[1]))
        if features.ndim != 2 or features.shape[1] != feature_count or features.shape[0] == 0:
            raise ValueError("normalization input has an unexpected feature shape")
        for start in range(0, features.shape[0], NORMALIZATION_CHUNK_ROWS):
            chunk = np.asarray(
                features[start : start + NORMALIZATION_CHUNK_ROWS], dtype=np.float64
            )
            if not np.isfinite(chunk).all():
                raise ValueError(f"nonfinite normalization input: {source}/{group}")
            total += chunk.sum(axis=0)
            square += np.square(chunk).sum(axis=0)
            count += len(chunk)
    if count == 0:
        raise ValueError("normalization has no input rows")
    mean64 = total / count
    variance64 = np.maximum(square / count - np.square(mean64), 0.0)
    std64 = np.sqrt(variance64)
    std64 = np.maximum(std64, NORMALIZATION_STD_FLOOR)
    mean = np.ascontiguousarray(mean64.astype(np.float32))
    std = np.ascontiguousarray(std64.astype(np.float32))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization moments must be finite FP32 with positive scales")
    return mean, std


def _members_identity(members: Sequence[tuple[str, int]]) -> list[list[object]]:
    """Validate and encode ordered source-qualified row-group membership."""
    result: list[list[object]] = []
    seen: set[tuple[str, int]] = set()
    for source, group in members:
        member = (source, int(group))
        if not isinstance(source, str) or not source or member[1] < 0 or member in seen:
            raise ValueError("normalization members must be unique source-qualified groups")
        seen.add(member)
        result.append([member[0], member[1]])
    if not result:
        raise ValueError("full-fit normalization requires at least one fitting member")
    return result


def build_normalization_identity(
    source_fingerprints: Mapping[str, Any],
    split_sha256: str,
    members: Sequence[tuple[str, int]],
    scope: str,
    feature_count: int,
) -> dict[str, Any]:
    """Build source-bound identity without model size or distributed-world fields."""
    if scope not in ("prefix", "all_fit"):
        raise ValueError("unsupported normalization scope")
    valid_digest = len(split_sha256) == 64 and all(
        char in "0123456789abcdef" for char in split_sha256
    )
    if not valid_digest:
        raise ValueError("split_sha256 must be a lowercase SHA-256 digest")
    if feature_count < 1:
        raise ValueError("normalization feature width must be positive")
    if not source_fingerprints or any(not isinstance(name, str) for name in source_fingerprints):
        raise ValueError("normalization cache requires source fingerprints")
    fingerprints = dict(sorted(source_fingerprints.items()))
    encoded_members = _members_identity(members)
    required_sources = {f"datasets/{member[0]}.parquet" for member in encoded_members}
    if not required_sources.issubset(fingerprints):
        raise ValueError("normalization cache is missing a fitting-source fingerprint")
    return {
        "algorithm": ALGORITHM,
        "feature_count": feature_count,
        "normalization_scope": scope,
        "normalization_members": encoded_members,
        "normalization_sequence_count": len(encoded_members),
        "source_fingerprints": fingerprints,
        "split_sha256": split_sha256,
    }


def _decode_cache(
    path: Path, identity: dict[str, Any], identity_sha256: str
) -> NormalizationMoments:
    """Read and verify one bounded immutable cache record."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_CACHE_BYTES:
        raise ValueError("normalization cache record exceeds its read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("normalization cache changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("normalization cache is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "identity",
        "identity_sha256",
        "mean",
        "std",
        "moments_sha256",
    }:
        raise ValueError("normalization cache schema is invalid")
    if payload["identity"] != identity or payload["identity_sha256"] != identity_sha256:
        raise ValueError("normalization cache identity does not match this fitting set")
    try:
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("normalization cache moments are not numeric") from error
    feature_count = int(identity["feature_count"])
    if mean.shape != (feature_count,) or std.shape != (feature_count,):
        raise ValueError("normalization cache moment shapes do not match the feature schema")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization cache moments must be finite with positive scales")
    digest = moments_sha256(mean, std)
    if payload["moments_sha256"] != digest:
        raise ValueError("normalization cache moments SHA-256 mismatch")
    return NormalizationMoments(
        np.ascontiguousarray(mean),
        np.ascontiguousarray(std),
        identity_sha256,
        digest,
        int(identity["normalization_sequence_count"]),
    )


def _write_cache_no_overwrite(path: Path, payload: bytes) -> bool:
    """Atomically publish a complete cache file without replacing an existing one."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".normalization-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_or_compute_full_fit_moments(
    cache_dir: Path,
    source_fingerprints: Mapping[str, Any],
    split_sha256: str,
    members: Sequence[tuple[str, int]],
    reader: Callable[[str, int], np.ndarray],
    *,
    feature_count: int = 112,
) -> NormalizationMoments:
    """Reuse or atomically create fitting-only all-fit moments bound to exact ancestry."""
    identity = build_normalization_identity(
        source_fingerprints, split_sha256, members, "all_fit", feature_count
    )
    identity_sha = normalization_identity_sha256(identity)
    cache_path = cache_dir / f"{identity_sha}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return _decode_cache(cache_path, identity, identity_sha)
    mean, std = compute_population_moments(members, reader, feature_count=feature_count)
    digest = moments_sha256(mean, std)
    payload = {
        "identity": identity,
        "identity_sha256": identity_sha,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "moments_sha256": digest,
    }
    encoded = _canonical_json(payload)
    _write_cache_no_overwrite(cache_path, encoded)
    return _decode_cache(cache_path, identity, identity_sha)
