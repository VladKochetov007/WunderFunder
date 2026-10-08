"""Read-only consumer for the cohort parsed-sequence cache."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, Sequence
from wnn.parsed_sequence_cache import (
    CACHE_SCHEMA,
    FORMAT_VERSION,
    SOURCE_RECEIPT_SCHEMA,
    _validate_membership,
    canonical_json_bytes,
    sha256_bytes,
)
from wnn.splits import VerifiedParsedCacheSources

ArrayRole = Literal["fit_train", "fit_valid", "evaluation_valid"]
SourceName = Literal["train", "valid"]
_EXPECTED_ARRAYS = {
    f"{split}_{name}": (split, name)
    for split in ("train", "valid")
    for name in ("features", "targets", "scored")
}
_DTYPES = {"features": np.dtype("<f4"), "targets": np.dtype("<f4"), "scored": np.dtype("|b1")}
_HASH_BLOCK_BYTES = 8 << 20


class ParsedCacheError(ValueError):
    """A parsed cache or its validation receipt is stale or incompatible."""


def _stat_identity(path: Path) -> dict[str, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ParsedCacheError(f"cache artifact is not a regular file: {path.name}")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _child_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ParsedCacheError("cache paths must be normalized relative paths")
    path = root.joinpath(*candidate.parts)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ParsedCacheError("cache artifact path escapes the cache directory") from exc
    _stat_identity(path)
    return path


def _read_pinned_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ParsedCacheError(f"{label} SHA-256 differs from its pin")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ParsedCacheError(f"{label} must be a JSON object")
    return value, raw


def _phase_io() -> dict[str, int] | None:
    try:
        values = {}
        for line in Path("/proc/self/io").read_text().splitlines():
            key, _, value = line.partition(":")
            if value:
                values[key] = int(value.strip())
        return values
    except (OSError, ValueError):
        return None


def _cgroup_snapshot() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    candidates = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _, relative = line.partition("::")
            if hierarchy == "0" and relative.startswith("/"):
                parts = Path(relative.lstrip("/")).parts
                if ".." not in parts:
                    candidates.insert(0, Path("/sys/fs/cgroup").joinpath(*parts))
                break
    except OSError:
        pass
    for candidate in candidates:
        try:
            for field, filename in (
                ("memory_current_bytes", "memory.current"),
                ("memory_max_bytes", "memory.max"),
            ):
                value = (candidate / filename).read_text().strip()
                if value != "max":
                    result[field] = int(value)
            for line in (candidate / "memory.stat").read_text().splitlines():
                key, value = line.split()
                if key in {
                    "anon",
                    "file",
                    "file_dirty",
                    "file_writeback",
                    "inactive_file",
                    "active_file",
                }:
                    result[f"memory_{key}_bytes"] = int(value)
            if result:
                result["cgroup_path"] = str(candidate)
                return result
        except (OSError, ValueError):
            continue
    return result


def _hash_array(path: Path) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    offset = 0
    advised_bytes = 0
    try:
        before = os.fstat(fd)
        while True:
            block = os.read(fd, _HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(fd, offset, len(block), os.POSIX_FADV_DONTNEED)
                    advised_bytes += len(block)
                except OSError:
                    pass
            offset += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ParsedCacheError(f"array changed while hashing: {path.name}")
    return digest.hexdigest(), {"bytes_hashed": offset, "fadvise_dontneed_bytes": advised_bytes}


def _load_manifest_bundle(
    manifest_path: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
    manifest_raw = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_raw)
    if manifest_sha256 != expected_manifest_sha256:
        raise ParsedCacheError("cache manifest SHA-256 differs from its pin")
    manifest = json.loads(manifest_raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != CACHE_SCHEMA
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("scope") != "cohort"
    ):
        raise ParsedCacheError("unsupported parsed-cache manifest")
    root = manifest_path.parent.resolve(strict=True)
    member_meta = manifest.get("membership_artifact", {})
    membership_path = _child_path(root, member_meta.get("path", ""))
    membership, _ = _read_pinned_json(membership_path, member_meta.get("sha256", ""), "membership")
    _validate_membership(membership)
    if (
        membership.get("fold") != manifest.get("fold")
        or membership.get("split_manifest_sha256") != manifest.get("split_manifest_sha256")
        or membership.get("portable_registry_sha256") != manifest.get("portable_registry_sha256")
        or membership.get("sources") != manifest.get("sources")
        or membership.get("member_counts") != manifest.get("member_counts")
    ):
        raise ParsedCacheError("manifest and membership bindings disagree")
    receipt_meta = manifest.get("source_verification_artifact", {})
    source_path = _child_path(root, receipt_meta.get("path", ""))
    source_receipt, _ = _read_pinned_json(
        source_path, receipt_meta.get("sha256", ""), "source receipt"
    )
    if (
        source_receipt.get("schema") != SOURCE_RECEIPT_SCHEMA
        or source_receipt.get("format_version") != FORMAT_VERSION
        or source_receipt.get("portable_registry_sha256")
        != manifest.get("portable_registry_sha256")
        or source_receipt.get("split_manifest_sha256") != manifest.get("split_manifest_sha256")
    ):
        raise ParsedCacheError("source verification receipt binds different inputs")
    for source in manifest["sources"].values():
        observed = source_receipt.get("sources", {}).get(source["logical_name"])
        if not isinstance(observed, dict) or any(
            observed.get(key) != source.get(key) for key in ("bytes", "sha256")
        ):
            raise ParsedCacheError("source verification receipt differs from the cache source pin")

    parser_identity = manifest.get("parser_identity")
    key_payload = {
        "schema": CACHE_SCHEMA,
        "format_version": FORMAT_VERSION,
        "fold": manifest["fold"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "portable_registry_sha256": manifest["portable_registry_sha256"],
        "membership_sha256": member_meta["sha256"],
        "sources": manifest["sources"],
        "feature_layout": {"sequence_length": SEQUENCE_LENGTH, "features": N_FEATURES},
        "array_dtype": {name: dtype.str for name, dtype in _DTYPES.items()},
        "parser_identity": parser_identity,
    }
    expected_cache_key = sha256_bytes(canonical_json_bytes(key_payload))
    if manifest.get("cache_key") != expected_cache_key:
        raise ParsedCacheError("cache key does not bind its source, membership, or parser identity")
    return manifest, membership, manifest_sha256, root


def verify_parsed_cache(
    manifest_path: Path, expected_manifest_sha256: str, receipt_path: Path
) -> tuple[dict[str, Any], str]:
    """Hash and header-check all six mmap arrays once; write a reusable receipt."""
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ParsedCacheError("validation receipt already exists; refusing to overwrite")
    manifest, membership, manifest_sha256, root = _load_manifest_bundle(
        manifest_path, expected_manifest_sha256
    )
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(_EXPECTED_ARRAYS):
        raise ParsedCacheError(
            "cache must contain the exact train/valid feature, target and mask arrays"
        )
    before_io, before_memory = _phase_io(), _cgroup_snapshot()
    verified: dict[str, Any] = {}
    expected_shapes = membership["array_shapes"]
    for key, (split, role) in _EXPECTED_ARRAYS.items():
        entry = arrays[key]
        path = _child_path(root, entry.get("path", ""))
        count = int(expected_shapes[split][0])
        shape = (
            [count, SEQUENCE_LENGTH, N_FEATURES]
            if role == "features"
            else [count, SEQUENCE_LENGTH, 2]
            if role == "targets"
            else [count, SEQUENCE_LENGTH]
        )
        if entry.get("shape") != shape or entry.get("dtype") != _DTYPES[role].str:
            raise ParsedCacheError(f"array schema mismatch: {key}")
        stat_before = _stat_identity(path)
        if stat_before["bytes"] != entry.get("size_bytes"):
            raise ParsedCacheError(f"array size mismatch: {key}")
        observed_sha, hash_telemetry = _hash_array(path)
        if observed_sha != entry.get("sha256"):
            raise ParsedCacheError(f"array SHA-256 mismatch: {key}")
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if (
                list(mapped.shape) != shape
                or mapped.dtype.str != _DTYPES[role].str
                or mapped.flags.writeable
            ):
                raise ParsedCacheError(f"array NPY header or read-only mapping mismatch: {key}")
        finally:
            mapped._mmap.close()
        stat_after = _stat_identity(path)
        if stat_before != stat_after:
            raise ParsedCacheError(f"array changed during validation: {key}")
        verified[key] = {
            "path": entry["path"],
            "bytes": entry["size_bytes"],
            "sha256": observed_sha,
            "stat": stat_after,
            "hash_telemetry": hash_telemetry,
        }
    receipt = {
        "schema": "wnn.parsed-sequence-cache-validation",
        "format_version": 1,
        "manifest_sha256": manifest_sha256,
        "cache_key": manifest["cache_key"],
        "membership_sha256": manifest["membership_artifact"]["sha256"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "fold": manifest["fold"],
        "arrays": verified,
        "verification": {
            "method": "streamed_sha256_and_readonly_npy_header_v1",
            "before_process_io": before_io,
            "after_process_io": _phase_io(),
            "memory_before": before_memory,
            "memory_after": _cgroup_snapshot(),
        },
    }
    raw = canonical_json_bytes(receipt) + b"\n"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt, sha256_bytes(raw)


class ParsedSequenceCache:
    """Validated immutable NPY mappings addressed by source-qualified row group."""

    def __init__(
        self,
        manifest: dict[str, Any],
        membership: dict[str, Any],
        arrays: dict[str, np.ndarray],
        manifest_sha256: str,
        validation_receipt_sha256: str,
    ) -> None:
        self.manifest = manifest
        self.membership = membership
        self.arrays = arrays
        self.preverified_sources = VerifiedParsedCacheSources(
            split_manifest_sha256=manifest["split_manifest_sha256"],
            cache_manifest_sha256=manifest_sha256,
            validation_receipt_sha256=validation_receipt_sha256,
            source_verification_sha256=manifest["source_verification_artifact"]["sha256"],
            sources={
                source["logical_name"]: {
                    "bytes": int(source["bytes"]),
                    "sha256": source["sha256"],
                }
                for source in manifest["sources"].values()
            },
        )
        self._members: dict[tuple[str, int], tuple[ArrayRole, dict[str, Any]]] = {}
        for role in ("fit_train", "fit_valid", "evaluation_valid"):
            for row in membership["memberships"][role]:
                source = "train" if row["source"] == "datasets/train.parquet" else "valid"
                self._members[(source, int(row["row_group"]))] = (role, row)

    @classmethod
    def open(
        cls,
        manifest_path: Path,
        *,
        expected_manifest_sha256: str,
        validation_receipt_path: Path,
        expected_validation_receipt_sha256: str,
        expected_cache_key: str,
        expected_split_manifest_sha256: str,
        expected_fold: int,
    ) -> ParsedSequenceCache:
        """Open only after the cohort-wide checksum pass and verify unchanged files."""
        manifest, membership, manifest_sha256, root = _load_manifest_bundle(
            manifest_path, expected_manifest_sha256
        )
        if (
            manifest["cache_key"] != expected_cache_key
            or manifest["split_manifest_sha256"] != expected_split_manifest_sha256
            or manifest["fold"] != expected_fold
        ):
            raise ParsedCacheError("cache manifest is not the requested fold/source cohort")
        receipt, receipt_sha256 = _read_pinned_json(
            validation_receipt_path,
            expected_validation_receipt_sha256,
            "cache validation receipt",
        )
        if (
            receipt.get("schema") != "wnn.parsed-sequence-cache-validation"
            or receipt.get("format_version") != 1
            or receipt.get("manifest_sha256") != manifest_sha256
            or receipt.get("cache_key") != manifest["cache_key"]
            or receipt.get("membership_sha256") != manifest["membership_artifact"]["sha256"]
            or receipt.get("split_manifest_sha256") != expected_split_manifest_sha256
            or receipt.get("fold") != expected_fold
        ):
            raise ParsedCacheError("validation receipt binds a different cache")
        verified_arrays = receipt.get("arrays")
        if not isinstance(verified_arrays, dict) or set(verified_arrays) != set(_EXPECTED_ARRAYS):
            raise ParsedCacheError("validation receipt does not cover every cache array")
        arrays = {}
        for key, (_split, role) in _EXPECTED_ARRAYS.items():
            entry = manifest["arrays"][key]
            verified = verified_arrays[key]
            path = _child_path(root, entry["path"])
            if (
                verified.get("path") != entry["path"]
                or verified.get("bytes") != entry["size_bytes"]
                or verified.get("sha256") != entry["sha256"]
                or verified.get("stat") != _stat_identity(path)
            ):
                raise ParsedCacheError(f"cache array changed after cohort validation: {key}")
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            if mapped.dtype.str != _DTYPES[role].str:
                mapped._mmap.close()
                raise ParsedCacheError(f"cache mapping dtype changed: {key}")
            arrays[key] = mapped
        return cls(manifest, membership, arrays, manifest_sha256, receipt_sha256)

    def groups(self, role: ArrayRole) -> list[int]:
        """Return the sealed row-group order for one membership role."""
        return [int(row["row_group"]) for row in self.membership["memberships"][role]]

    def read(self, split: SourceName, row_group: int, role: ArrayRole) -> Sequence:
        """Return one writable owned sequence, matching ``SequenceFile.read``."""
        member = self._members.get((split, int(row_group)))
        if member is None or member[0] != role:
            raise ParsedCacheError(f"{split} row group {row_group} is not in role {role}")
        row = member[1]
        array_index = int(row["array_index"])
        features = np.array(
            self.arrays[f"{split}_features"][array_index], dtype=np.float32, copy=True
        )
        targets = np.array(
            self.arrays[f"{split}_targets"][array_index], dtype=np.float32, copy=True
        )
        scored = np.array(self.arrays[f"{split}_scored"][array_index], dtype=bool, copy=True)
        return Sequence(int(row["seq_ix"]), features, targets, scored)

    def read_features(self, split: SourceName, row_group: int, role: ArrayRole) -> np.ndarray:
        """Return one owned feature sequence without exposing cached targets."""
        member = self._members.get((split, int(row_group)))
        if member is None or member[0] != role:
            raise ParsedCacheError(f"{split} row group {row_group} is not in role {role}")
        return np.array(
            self.arrays[f"{split}_features"][int(member[1]["array_index"])],
            dtype=np.float32,
            copy=True,
        )

    def close(self) -> None:
        """Close the six read-only mappings."""
        for array in self.arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
