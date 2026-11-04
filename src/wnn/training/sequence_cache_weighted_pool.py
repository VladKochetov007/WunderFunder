"""Build a source-bound weighted cache from heterogeneous auxiliary-fit teachers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import (
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)
from wnn.training.sequence_cache import MAX_CACHE_MANIFEST_BYTES, TeacherCache
from wnn.training.sequence_cache_weighted_pool_identity import (
    auxiliary_members,
    build_weighted_pool_producer,
)

MAX_POOL_PARENTS = 8
DEFAULT_MINIMUM_FREE_BYTES = 80 << 30


def _values(cache: TeacherCache, member: tuple[str, int]) -> np.ndarray:
    """Read one complete finite raw prediction sequence without a full-cache copy."""
    entry = cache.entries[member]
    values = np.load(cache._entry_path(entry), mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2):
        raise ValueError("heterogeneous teacher arrays must be float32 20000x2")
    if not np.isfinite(values).all():
        raise ValueError("heterogeneous teacher cache contains nonfinite predictions")
    return values


def _weighted_sequence(
    caches: list[TeacherCache],
    expected_parents: list[Mapping[str, object]],
    member: tuple[str, int],
) -> np.ndarray:
    """Apply frozen weights in parent order with no clipping before or after the sum."""
    result = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for cache, parent in zip(caches, expected_parents, strict=True):
            contribution = np.multiply(
                _values(cache, member), np.float32(parent["weight"]), dtype=np.float32
            )
            np.add(result, contribution, out=result)
    if not np.isfinite(result).all():
        raise ValueError("weighted teacher target overflowed float32")
    return result


def _write_sequence(root: Path, member: tuple[str, int], values: np.ndarray) -> dict[str, object]:
    """Write one complete fit member and bind its bytes in the output manifest."""
    source, group = member
    name = f"{source}-{group}.npy"
    path = root / name
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"split": source, "row_group": group, "file": name, "sha256": file_sha256(path)}


def _write_manifest(root: Path, value: dict[str, object]) -> None:
    """Publish the cache manifest after every array checksum has been written."""
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_CACHE_MANIFEST_BYTES:
        raise ValueError("weighted teacher manifest exceeds the bounded cache metadata size")
    pending = root / "manifest.json.partial"
    with pending.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(root / "manifest.json")


def merge_weighted_teacher_caches(
    input_roots: list[Path],
    expected_parents: list[Mapping[str, object]],
    data_root: Path,
    split_manifest: Path,
    output: Path,
    *,
    pool_id: str,
    request_sha256: str,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
) -> Path:
    """Merge pinned heterogeneous source exports over the exact auxiliary fit members."""
    if not 2 <= len(input_roots) <= MAX_POOL_PARENTS or len(expected_parents) != len(input_roots):
        raise ValueError("weighted teacher pool requires two to eight paired cache/receipt inputs")
    if not pool_id or len(pool_id) > 128 or len(request_sha256) != 64:
        raise ValueError("weighted teacher pool identity is malformed")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be a nonnegative integer")
    data_root, split_manifest = data_root.resolve(strict=True), split_manifest.resolve(strict=True)
    split = load_campaign_split(split_manifest, data_root)
    fit_members, _control_members = auxiliary_members(split)
    caches = [TeacherCache(root.resolve(strict=True), split, fit_members) for root in input_roots]
    if any(cache.metadata.get("hint_dim") != 0 for cache in caches):
        raise ValueError("heterogeneous target pools currently require hint_dim=0")
    producer, fit_members, _control_members = build_weighted_pool_producer(
        caches, expected_parents, split, pool_id, request_sha256
    )
    target = output.parent.resolve(strict=True) / output.name
    if target.name in ("", ".", "..") or target.is_symlink():
        raise ValueError("output must name a new regular cache directory")
    inputs = [root.resolve(strict=True) for root in input_roots]
    if any(target == root or root in target.parents or target in root.parents for root in inputs):
        raise ValueError("weighted cache output must be disjoint from all input caches")
    if not target.parent.is_dir() or target.exists():
        raise FileExistsError(
            f"weighted cache output parent must exist and target be new: {target}"
        )
    estimate = len(fit_members) * (SEQUENCE_LENGTH * 2 * np.dtype(np.float32).itemsize + 512)
    if shutil.disk_usage(target.parent).free < estimate + minimum_free_bytes:
        raise OSError(
            "insufficient disk space for the weighted cache and required free-space floor"
        )
    lock = target.with_name(f".{target.name}.merge.lock")
    stage: Path | None = None
    lock_created = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("weighted teacher pool merge in progress\n")
        lock_created = True
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"weighted cache output appeared during merge: {target}")
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
        entries = [
            _write_sequence(stage, member, _weighted_sequence(caches, expected_parents, member))
            for member in fit_members
        ]
        for cache in caches:
            cache.verify_unchanged()
        _verify_split_unchanged(split, split_manifest, data_root)
        manifest = {
            "schema": "sequence-teacher-cache-v1",
            "complete": True,
            "input_schema": "raw112-v1",
            "membership": "fit",
            "split_sha256": split.sha256,
            "hint_dim": 0,
            "sequence_length": SEQUENCE_LENGTH,
            "producer": producer,
            "sequences": entries,
        }
        _write_manifest(stage, manifest)
        TeacherCache(stage, split, fit_members)
        for cache in caches:
            cache.verify_unchanged()
        _verify_split_unchanged(split, split_manifest, data_root)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"weighted cache output appeared during merge: {target}")
        stage.rename(target)
        stage = None
        return target
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if lock_created:
            lock.unlink(missing_ok=True)


def _verify_split_unchanged(expected: CampaignSplit, manifest: Path, data_root: Path) -> None:
    """Reload raw source identities and component membership after the merge."""
    current = load_campaign_split(manifest, data_root)
    if current.sha256 != expected.sha256 or current != expected:
        raise ValueError("auxiliary split or source identities changed during cache merge")
