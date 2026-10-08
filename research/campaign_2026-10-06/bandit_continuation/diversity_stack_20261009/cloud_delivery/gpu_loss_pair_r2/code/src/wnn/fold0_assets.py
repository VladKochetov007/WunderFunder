"""One-pass validation and path binding for the immutable fold-0 auxiliary assets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"

_AUXILIARY_FILES = {
    "cache/chain_folds.npz": {
        "filename": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn",
        "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
        "size_bytes": 550072,
    },
    "cache/chains.npz": {
        "filename": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn",
        "sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
        "size_bytes": 650142,
    },
    "cache/train_targets.npz": {
        "filename": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn",
        "sha256": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
        "size_bytes": 1909335790,
    },
    "cache/valid_targets.npz": {
        "filename": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn",
        "sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
        "size_bytes": 337186198,
    },
}
AUXILIARY_SOURCE_PATHS = frozenset(_AUXILIARY_FILES)
_HASH_BLOCK_BYTES = 8 << 20


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _process_io() -> dict[str, int] | None:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if value:
                values[key] = int(value.strip())
        return values
    except (OSError, ValueError):
        return None


def _memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["process_rss_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        result["cgroup_current_bytes"] = int(
            Path("/sys/fs/cgroup/memory.current").read_text(encoding="ascii").strip()
        )
        keys = {"anon", "file", "active_file", "inactive_file", "file_dirty", "file_writeback"}
        memory_stat = {}
        for line in Path("/sys/fs/cgroup/memory.stat").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(" ")
            if key in keys and value:
                memory_stat[key] = int(value)
        result["cgroup_memory_stat_bytes"] = memory_stat
    except (OSError, ValueError):
        result["cgroup_memory_stat_bytes"] = None
    return result


def _stat_identity(path: Path) -> dict[str, int]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"fold-0 asset is not a regular file: {path}")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _hash_file(path: Path) -> tuple[str, int, int, int]:
    digest = hashlib.sha256()
    byte_count = 0
    advised_bytes = 0
    advised_calls = 0
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(_HASH_BLOCK_BYTES):
            digest.update(block)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(
                        stream.fileno(), byte_count, len(block), os.POSIX_FADV_DONTNEED
                    )
                    advised_bytes += len(block)
                    advised_calls += 1
                except OSError:
                    pass
            byte_count += len(block)
    return digest.hexdigest(), byte_count, advised_bytes, advised_calls


def _read_pinned_json(path: Path, expected_sha256: str, name: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"{name} SHA-256 differs from its pin")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value, raw


def verify_and_link_fold0_assets(
    asset_manifest_path: Path,
    portable_registry_path: Path,
    split_manifest_path: Path,
    cache_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Hash the four pinned source files once and symlink them into the code tree."""
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"refusing to replace fold-0 asset receipt: {receipt_path}")
    manifest, manifest_raw = _read_pinned_json(
        asset_manifest_path, ASSET_MANIFEST_SHA256, "fold-0 asset manifest"
    )
    registry, registry_raw = _read_pinned_json(
        portable_registry_path, REGISTRY_SHA256, "portable input registry"
    )
    split, split_raw = _read_pinned_json(split_manifest_path, SPLIT_SHA256, "fold-0 split")
    if manifest.get("dataset") != ASSET_DATASET or manifest.get("private") is not True:
        raise ValueError("fold-0 asset manifest dataset identity or privacy flag differs")
    if manifest.get("version") != 1:
        raise ValueError("unsupported fold-0 asset manifest version")
    if registry.get("version") != 1 or registry.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ValueError("portable registry is not bound to the pinned fold-0 split")
    manifest_rows = manifest.get("files")
    if not isinstance(manifest_rows, list):
        raise ValueError("fold-0 asset manifest file list is malformed")
    by_path = {row.get("path"): row for row in manifest_rows if isinstance(row, dict)}
    if len(by_path) != len(manifest_rows):
        raise ValueError("fold-0 asset manifest has duplicate or malformed paths")
    if not set(_AUXILIARY_FILES).issubset(by_path):
        raise ValueError("fold-0 dataset omits a required auxiliary cache file")
    split_sources = split.get("sources", {})
    registry_sources = registry.get("sources", {})
    for logical_name, expected in _AUXILIARY_FILES.items():
        row = by_path[logical_name]
        split_identity = split_sources.get(logical_name)
        registry_entry = registry_sources.get(logical_name)
        if (
            row.get("filename") != expected["filename"]
            or row.get("sha256") != expected["sha256"]
            or row.get("size") != expected["size_bytes"]
            or split_identity is None
            or registry_entry is None
            or registry_entry.get("original_identity") != split_identity
            or registry_entry.get("sha256") != expected["sha256"]
            or split_identity.get("bytes") != expected["size_bytes"]
        ):
            raise ValueError(f"fold-0 asset identity disagrees with split/registry: {logical_name}")

    cache_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    total_bytes = 0
    advised_bytes = 0
    advised_calls = 0
    io_before, memory_before = _process_io(), _memory_snapshot()
    started = time.monotonic()
    for logical_name, expected in _AUXILIARY_FILES.items():
        source = asset_manifest_path.parent / expected["filename"]
        source_resolved = source.resolve(strict=True)
        before = _stat_identity(source)
        if before["size_bytes"] != expected["size_bytes"]:
            raise ValueError(f"fold-0 asset size differs from its pin: {logical_name}")
        digest, bytes_hashed, file_advised_bytes, file_advised_calls = _hash_file(source)
        after = _stat_identity(source)
        if before != after or bytes_hashed != expected["size_bytes"] or digest != expected["sha256"]:
            raise ValueError(f"fold-0 asset changed or hash differs: {logical_name}")
        destination = cache_root / Path(logical_name).name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace cache entry: {destination}")
        destination.symlink_to(source_resolved)
        if not destination.is_symlink() or destination.resolve(strict=True) != source_resolved:
            raise ValueError(f"fold-0 asset symlink did not preserve its target: {logical_name}")
        link_stat = _stat_identity(destination)
        if link_stat != after:
            raise ValueError(f"fold-0 asset changed while its cache link was created: {logical_name}")
        total_bytes += bytes_hashed
        advised_bytes += file_advised_bytes
        advised_calls += file_advised_calls
        files[logical_name] = {
            "filename": expected["filename"],
            "sha256": digest,
            "size_bytes": bytes_hashed,
            "portable_registry_sha256": registry_sources[logical_name]["sha256"],
            "split_identity": split_sources[logical_name],
            "observed_stat": after,
            "resolved_target": str(source_resolved),
            "cache_link": str(destination),
            "cache_link_is_symlink": destination.is_symlink(),
            "fadvise_dontneed_advised_bytes": file_advised_bytes,
            "fadvise_dontneed_advised_calls": file_advised_calls,
        }

    receipt = {
        "schema": "wnn.kaggle-fold0-auxiliary-assets-validation.v1",
        "format_version": 1,
        "state": "verified_and_linked",
        "dataset": ASSET_DATASET,
        "asset_manifest_sha256": _sha256_bytes(manifest_raw),
        "portable_registry_sha256": _sha256_bytes(registry_raw),
        "split_manifest_sha256": _sha256_bytes(split_raw),
        "files": files,
        "telemetry": {
            "started_monotonic_seconds": started,
            "elapsed_seconds": time.monotonic() - started,
            "total_bytes_hashed": total_bytes,
            "fadvise_dontneed_advised_bytes": advised_bytes,
            "fadvise_dontneed_advised_calls": advised_calls,
            "process_io_before": io_before,
            "process_io_after": _process_io(),
            "memory_before": memory_before,
            "memory_after": _memory_snapshot(),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def validate_preverified_fold0_assets(
    root: Path,
    split_manifest_path: Path,
    portable_registry_path: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> None:
    """Check one-pass receipt and current path/stat identity without rehashing payloads."""
    if len(expected_receipt_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_receipt_sha256
    ):
        raise ValueError("fold-0 auxiliary receipt SHA-256 pin is malformed")
    receipt_raw = receipt_path.read_bytes()
    if _sha256_bytes(receipt_raw) != expected_receipt_sha256:
        raise ValueError("fold-0 auxiliary asset receipt changed after startup validation")
    receipt = json.loads(receipt_raw)
    registry, registry_raw = _read_pinned_json(
        portable_registry_path, REGISTRY_SHA256, "portable input registry"
    )
    split, split_raw = _read_pinned_json(split_manifest_path, SPLIT_SHA256, "fold-0 split")
    if (
        receipt.get("schema") != "wnn.kaggle-fold0-auxiliary-assets-validation.v1"
        or receipt.get("format_version") != 1
        or receipt.get("state") != "verified_and_linked"
        or receipt.get("dataset") != ASSET_DATASET
        or receipt.get("asset_manifest_sha256") != ASSET_MANIFEST_SHA256
        or receipt.get("portable_registry_sha256") != _sha256_bytes(registry_raw)
        or receipt.get("split_manifest_sha256") != _sha256_bytes(split_raw)
        or set(receipt.get("files", {})) != set(_AUXILIARY_FILES)
        or receipt.get("telemetry", {}).get("total_bytes_hashed")
        != sum(row["size_bytes"] for row in _AUXILIARY_FILES.values())
    ):
        raise ValueError("fold-0 auxiliary receipt does not bind this split/registry/source")
    for logical_name, expected in _AUXILIARY_FILES.items():
        row = receipt["files"][logical_name]
        split_identity = split["sources"].get(logical_name)
        registry_entry = registry["sources"].get(logical_name)
        source = root / logical_name
        if (
            split_identity is None
            or registry_entry is None
            or registry_entry.get("original_identity") != split_identity
            or row.get("filename") != expected["filename"]
            or row.get("sha256") != expected["sha256"]
            or row.get("portable_registry_sha256") != registry_entry.get("sha256")
            or row.get("size_bytes") != expected["size_bytes"]
            or row.get("split_identity") != split_identity
            or split_identity.get("bytes") != expected["size_bytes"]
            or not source.is_symlink()
            or row.get("cache_link") != str(source)
            or row.get("cache_link_is_symlink") is not True
        ):
            raise ValueError(f"fold-0 preverified asset receipt is incomplete: {logical_name}")
        resolved = source.resolve(strict=True)
        if str(resolved) != row.get("resolved_target"):
            raise ValueError(f"fold-0 asset mount path changed: {logical_name}")
        if _stat_identity(source) != row.get("observed_stat"):
            raise ValueError(f"fold-0 asset stat identity changed after verification: {logical_name}")
