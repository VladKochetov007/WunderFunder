#!/usr/bin/env python3
"""Exact W18 metadata input contracts and source identity checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

# Any is necessary for heterogeneous manifest and restricted checkpoint mappings.
SCHEMA = "quality_gap.tl3_w18_checkpoint_metadata.v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ADMISSION_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_DEPTH = 8
MAX_ITEMS = 4096
TIME_LIMIT_SECONDS = 120
RESOURCE_CONTRACT = {
    "cpu_quota_cores": 0.5,
    "memory_max_bytes": 1_500_000_000,
    "swap_max_bytes": 0,
    "runtime_max_seconds": 120,
}

# These eight logical input paths and sizes are pinned by the W18 trace and fresh
# stat-only preflight. The root-bound manifest records the exact resolved path chain.
ALLOWLIST: dict[str, tuple[str, int]] = {
    "ffull_s3_ep1": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/ffull_s3/model_ep1.pt",
        1_053_479,
    ),
    "ffull_s4_ep1": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/ffull_s4/model_ep1.pt",
        1_053_479,
    ),
    "frawW_s3_ep3": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/frawW_s3/model_ep3.pt",
        1_054_943,
    ),
    "frawW_s4_ep3": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/frawW_s4/model_ep3.pt",
        1_054_943,
    ),
    "frawW_s3ft_ep1": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/frawW_s3ft/model_ep1.pt",
        1_053_607,
    ),
    "frawW_s4ft_ep1": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/frawW_s4ft/model_ep1.pt",
        1_053_607,
    ),
    "fmu128_s3_ep3": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fmu128_s3/model_ep3.pt",
        1_054_815,
    ),
    "fmu128_s4_ep3": (
        "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fmu128_s4/model_ep3.pt",
        1_054_815,
    ),
}
SOURCE_OBSERVATION = {
    "remote_run_evidence_sha256": "08a171c8718a393b9b91b79d6c05fcf2bec6b172fe234b693f9af128daff8609",
    "six_target_template_sha256": "615ee43acb0afecd96a95f0ed1393e982c69210a564bf715a22c8fd1eb1136d4",
    "root_path_preflight_sha256": "b555f6e282e45129be74d7614da8a77fc42b2a684da06c6c7aa6d83ae8acc5ba",
    "root_path_preflight_observed_utc": "2026-10-18T18:19:20.195765+00:00",
    "historical_observed_utc": "2026-10-18T17:49:55.242392+00:00",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_RE = re.compile(r"secret|password|token|credential|api.?key|access.?key", re.IGNORECASE)
_OMIT_KEYS = {
    "model",
    "optimizer",
    "optimizers",
    "schedulers",
    "scaler",
    "torch_rng",
    "cuda_rng",
    "numpy_rng",
    "training_state",
}
_GENERATION_KEYS = {"device", "inode", "size", "mtime_ns", "ctime_ns", "mode"}


class MetadataError(ValueError):
    """Fail-closed contract or source-integrity error."""


def require(condition: bool, message: str) -> None:
    """Raise a concise fail-closed error when a contract condition is false."""
    if not condition:
        raise MetadataError(message)


def _stat_signature(info: os.stat_result) -> dict[str, int]:
    """Return fields that distinguish a checkpoint file generation."""
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
        "mode": int(info.st_mode),
    }


def _signature(path: Path) -> dict[str, int]:
    """Return stable identity for a regular non-symlink metadata file."""
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    return _stat_signature(info)


def _sha256_file(path: Path, max_bytes: int) -> tuple[str, bytes]:
    """Hash a bounded binding file through one no-follow descriptor."""
    before_path = _signature(path)
    require(before_path["size"] <= max_bytes, f"file exceeds {max_bytes}-byte bound: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    require(nofollow != 0, "O_NOFOLLOW is required for binding files")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    fd = os.open(path, flags)
    try:
        before_fd = os.fstat(fd)
        require(stat.S_ISREG(before_fd.st_mode), f"not a regular file: {path}")
        fd_signature = _stat_signature(before_fd)
        require(fd_signature == before_path, f"file changed before open: {path}")
        require(before_fd.st_size <= max_bytes, f"file exceeds {max_bytes}-byte bound: {path}")
        expected_size = before_fd.st_size
        total = 0
        offset = 0
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.pread(fd, min(64 * 1024, max_bytes - total + 1), offset)
            if not block:
                break
            total += len(block)
            require(total <= max_bytes, f"file exceeds {max_bytes}-byte bound: {path}")
            digest.update(block)
            chunks.append(block)
            offset += len(block)
        require(total == expected_size, f"file size changed while read: {path}")
        after_fd = os.fstat(fd)
        after_signature = _stat_signature(after_fd)
        require(after_signature == fd_signature, f"file changed during read: {path}")
        require(_signature(path) == before_path, f"file path changed during read: {path}")
        return digest.hexdigest(), b"".join(chunks)
    finally:
        os.close(fd)


def _path_resolution(path: Path) -> dict[str, Any]:
    """Record an absolute path's resolved target and every symlink component."""
    require(path.is_absolute(), "logical checkpoint path must be absolute")
    current = Path(path.anchor)
    symlinks: list[dict[str, str]] = []
    for part in path.parts[1:]:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            symlinks.append(
                {
                    "link_path": str(current),
                    "link_text": os.readlink(current),
                    "resolved_target": str(current.resolve(strict=True)),
                }
            )
    return {"resolved_path": str(path.resolve(strict=True)), "symlinks": symlinks}


def _fd_sha256(fd: int) -> str:
    """Hash an already opened file descriptor without changing its open-file cursor."""
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(fd, 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def validate_manifest(manifest: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate an exact eight-checkpoint root-bound manifest."""
    require(isinstance(manifest, dict), "manifest must be an object")
    require(
        set(manifest)
        == {
            "schema",
            "status",
            "resource_contract",
            "source_observation",
            "root_admission_receipt",
            "inputs",
        },
        "manifest fields differ",
    )
    require(manifest.get("schema") == SCHEMA, "manifest schema mismatch")
    require(manifest.get("status") == "ROOT_HASH_BOUND", "fresh root-bound manifest required")
    require(manifest.get("resource_contract") == RESOURCE_CONTRACT, "resource contract mismatch")
    require(manifest.get("source_observation") == SOURCE_OBSERVATION, "source observation mismatch")
    admission = manifest.get("root_admission_receipt")
    require(isinstance(admission, dict), "root admission receipt binding missing")
    require(set(admission) == {"path", "sha256"}, "root admission binding fields differ")
    admission_path = admission.get("path")
    require(
        isinstance(admission_path, str) and Path(admission_path).is_absolute(),
        "admission path must be absolute",
    )
    require(
        _SHA_RE.fullmatch(str(admission.get("sha256", ""))) is not None, "admission SHA-256 missing"
    )
    inputs = manifest.get("inputs")
    require(
        isinstance(inputs, list) and len(inputs) == len(ALLOWLIST), "exact eight inputs required"
    )
    total_bytes = 0
    for item, (logical_id, expected) in zip(inputs, ALLOWLIST.items(), strict=True):
        require(isinstance(item, dict), "input entry must be an object")
        require(
            set(item)
            == {
                "id",
                "logical_path",
                "source_resolution",
                "expected_bytes",
                "expected_sha256",
                "expected_generation",
            },
            "input entry fields differ",
        )
        logical_path, expected_bytes = expected
        require(item.get("id") == logical_id, "input order or id differs from exact allowlist")
        require(
            item.get("logical_path") == logical_path, f"unexpected logical path for {logical_id}"
        )
        require(
            item.get("expected_bytes") == expected_bytes,
            f"historical size pin differs for {logical_id}",
        )
        require(0 < expected_bytes <= MAX_FILE_BYTES, f"per-file bound for {logical_id}")
        require(
            _SHA_RE.fullmatch(str(item.get("expected_sha256", ""))) is not None,
            f"fresh SHA missing for {logical_id}",
        )
        generation = item.get("expected_generation")
        require(
            isinstance(generation, dict) and set(generation) == _GENERATION_KEYS,
            f"generation pin missing for {logical_id}",
        )
        require(
            all(type(value) is int for value in generation.values()),
            f"invalid generation fields for {logical_id}",
        )
        require(generation["size"] == expected_bytes, f"generation size differs for {logical_id}")
        resolution = item.get("source_resolution")
        require(
            isinstance(resolution, dict) and set(resolution) == {"resolved_path", "symlinks"},
            f"path resolution missing for {logical_id}",
        )
        require(
            resolution == {"resolved_path": logical_path, "symlinks": []},
            f"resolved source path differs from the pinned no-symlink preflight for {logical_id}",
        )
        total_bytes += expected_bytes
    require(total_bytes <= MAX_TOTAL_BYTES, "aggregate checkpoint bound")
    return inputs, admission


def _read_json_bound(path: Path, max_bytes: int) -> tuple[dict[str, Any], str]:
    """Read JSON from a stable regular non-symlink file and retain its exact digest."""
    digest, raw = _sha256_file(path, max_bytes)
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), "bound JSON file must contain an object")
    return value, digest


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Read and hash the exact input manifest used for this extraction."""
    return _read_json_bound(path, MAX_MANIFEST_BYTES)


def _verify_admission(admission: dict[str, str]) -> None:
    """Require the root admission receipt file to match its manifest hash pin."""
    path = Path(admission["path"])
    digest, _ = _sha256_file(path, MAX_ADMISSION_BYTES)
    require(digest == admission["sha256"], "root admission receipt SHA mismatch")
