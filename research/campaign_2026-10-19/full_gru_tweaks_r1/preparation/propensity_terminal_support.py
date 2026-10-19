"""Verify a completed strict propensity run without exporting model payloads."""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import os
import pickle
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPECTED_BINDING_SHA256 = "babfd92939eb5580933929a591872dec9056aaacd5002979a26ca2de8ed6ff00"
EXPECTED_SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
EXPECTED_COMMIT = "235db75ab91e24c69f21e637a83a48512dc674fe"
EXPECTED_INVOCATION = "85b87a83f96945f4b78908f124435a2b"
EXPECTED_RUN_NAME = "fg19_propensity_clean_f0_seed0_e4"
EXPECTED_EPOCHS = 4
EXPECTED_UPDATES_PER_EPOCH = 880
MAX_JSON_BYTES = 16_000_000
MAX_TORCH_BYTES = 1_000_000_000


class _UnapprovedGlobal:
    """Serialization probe used only by the restricted-loader self-test."""

    def __init__(self) -> None:
        self.value = 1


def _stable_file(path: Path, limit: int) -> tuple[bytes, str, dict[str, int]]:
    """Read/hash one regular non-symlink file and verify it did not change."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path.name}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"file changed during read: {path.name}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"path changed during read: {path.name}")
        if total != before.st_size:
            raise ValueError(f"short read: {path.name}")
        return (
            b"".join(chunks),
            digest.hexdigest(),
            {
                "bytes": total,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mtime_ns": before.st_mtime_ns,
            },
        )
    finally:
        os.close(descriptor)


def _hash_file(path: Path, limit: int) -> tuple[str, dict[str, int]]:
    """Hash a regular file in bounded memory and guard its identity."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path.name}")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"file changed during hash: {path.name}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"path changed during hash: {path.name}")
        if total != before.st_size:
            raise ValueError(f"short read: {path.name}")
        return digest.hexdigest(), {
            "bytes": total,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def _load_torch_file(
    path: Path, torch: object, *, allowlist: list[object] | None = None
) -> tuple[object, str, dict[str, int]]:
    """Hash before restricted CPU load and verify file identity after deserialization."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_TORCH_BYTES:
            raise ValueError(f"not a bounded checkpoint file: {path.name}")
        digest = hashlib.sha256()
        total = 0
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
            total += len(chunk)
        if total != before.st_size:
            raise ValueError(f"short checkpoint read: {path.name}")
        stream.seek(0)
        if allowlist is None:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        else:
            with torch.serialization.safe_globals(allowlist):
                payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"checkpoint changed during load: {path.name}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"checkpoint path changed during load: {path.name}")
        return (
            payload,
            digest.hexdigest(),
            {
                "bytes": total,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mtime_ns": before.st_mtime_ns,
            },
        )


def _json_file(path: Path) -> tuple[dict[str, object], str]:
    raw, digest, _ = _stable_file(path, MAX_JSON_BYTES)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return value, digest


def _safe_relative(root: Path, relative: str) -> Path:
    """Resolve a manifest path without permitting traversal or external targets."""
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts or not child.parts:
        raise ValueError(f"unsafe source path: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / child).resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"source path escaped its root: {relative}")
    return resolved


def _verify_binding(binding: Mapping[str, object], source_root: Path) -> dict[str, object]:
    """Hash all source pins and bind the narrower strict-fit source subset."""
    sources = binding.get("source_files")
    profile = binding.get("profile_source_hashes")
    file_hashes = binding.get("file_hashes")
    if not isinstance(sources, dict) or len(sources) != 153:
        raise ValueError("binding must contain all 153 frozen source files")
    if not isinstance(profile, dict) or len(profile) != 25:
        raise ValueError("binding must contain the 25 strict propensity source pins")
    if not isinstance(file_hashes, dict):
        raise ValueError("binding file hash map is missing")
    if binding.get("source_commit") != EXPECTED_COMMIT:
        raise ValueError("binding source commit differs")
    combined = hashlib.sha256()
    for relative, expected in sorted(sources.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("invalid source file pin")
        actual = _hash_file(_safe_relative(source_root, relative), MAX_TORCH_BYTES)[0]
        if actual != expected:
            raise ValueError(f"source file hash differs: {relative}")
        if file_hashes.get(f"source/{relative}") != expected:
            raise ValueError(f"binding file hash map disagrees: {relative}")
        combined.update(relative.encode("utf-8") + b"\0" + actual.encode("ascii") + b"\n")
    for relative, expected in profile.items():
        if sources.get(relative) != expected:
            raise ValueError(f"profile source is not bound by the full source set: {relative}")
    return {
        "source_commit": EXPECTED_COMMIT,
        "source_files_verified": len(sources),
        "source_set_sha256": combined.hexdigest(),
        "strict_profile_files": len(profile),
        "strict_profile_sha256": hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _numpy_safe_globals(np: object) -> list[object]:
    """Return only globals required by the producer's NumPy RNG state tuple."""
    core = getattr(np, "_core", None) or getattr(np, "core", None)
    multiarray = getattr(core, "multiarray", None)
    reconstruct = getattr(multiarray, "_reconstruct", None)
    if reconstruct is None:
        raise RuntimeError("NumPy multiarray._reconstruct is unavailable")
    return [reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32")), codecs.encode]


def run_loader_self_test(torch: object, np: object) -> list[str]:
    """Prove the narrow NumPy RNG allowlist works and still rejects unknown globals."""
    allowlist = _numpy_safe_globals(np)
    before = tuple(torch.serialization.get_safe_globals())
    valid = io.BytesIO()
    torch.save({"rng": np.random.RandomState(17).get_state()}, valid)
    valid.seek(0)
    with torch.serialization.safe_globals(allowlist):
        restored = torch.load(valid, map_location="cpu", weights_only=True)
    if restored["rng"][0] != "MT19937" or restored["rng"][1].dtype != np.dtype("uint32"):
        raise RuntimeError("restricted RNG fixture did not round-trip")
    blocked = io.BytesIO()
    torch.save({"unsupported": _UnapprovedGlobal()}, blocked)
    blocked.seek(0)
    try:
        with torch.serialization.safe_globals(allowlist):
            torch.load(blocked, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if "_UnapprovedGlobal" not in str(error) and "Unsupported global" not in str(error):
            raise RuntimeError(
                "unknown-global rejection failed for an unexpected reason"
            ) from error
    else:
        raise RuntimeError("restricted loader accepted an unapproved global")
    if tuple(torch.serialization.get_safe_globals()) != before:
        raise RuntimeError("restricted safe globals leaked beyond their context")
    return [f"{item.__module__}.{item.__qualname__}" for item in allowlist]


def _tensor_summary(tensor: object, torch: object) -> dict[str, object]:
    """Hash tensor bytes and report metadata only; never return tensor values."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("checkpoint state contains a non-tensor model value")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    finite = bool(torch.isfinite(value).all().item())
    if not finite:
        raise ValueError("checkpoint contains a non-finite tensor")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": len(raw),
        "finite": finite,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _state_tensor_summary(tree: object, torch: object, np: object) -> dict[str, object]:
    """Summarize nested tensor/array leaves without exposing numeric contents."""
    if isinstance(tree, torch.Tensor):
        return {"kind": "tensor", **_tensor_summary(tree, torch)}
    if isinstance(tree, np.ndarray):
        if not np.isfinite(tree).all():
            raise ValueError("checkpoint contains a non-finite NumPy array")
        raw = np.ascontiguousarray(tree).tobytes()
        return {
            "kind": "ndarray",
            "dtype": str(tree.dtype),
            "shape": list(tree.shape),
            "bytes": len(raw),
            "finite": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(tree, dict):
        return {
            str(key): _state_tensor_summary(value, torch, np)
            for key, value in sorted(tree.items(), key=lambda item: str(item[0]))
            if isinstance(value, (dict, list, tuple, torch.Tensor, np.ndarray))
        }
    if isinstance(tree, (list, tuple)):
        return {
            "kind": type(tree).__name__,
            "length": len(tree),
            "items": [
                _state_tensor_summary(value, torch, np)
                for index, value in enumerate(tree)
                if isinstance(value, (dict, list, tuple, torch.Tensor, np.ndarray))
            ],
        }
    return {"kind": type(tree).__name__}


def _group_identity(groups: object, np: object) -> dict[str, object]:
    values = np.asarray(groups, dtype="<i8")
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError("row-group membership is not a unique vector")
    return {
        "count": int(len(values)),
        "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
