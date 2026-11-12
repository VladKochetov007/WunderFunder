#!/usr/bin/env python3
"""Bounded, hash-first input and runtime helpers for the CPU receiver."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (  # JSON and package objects are heterogeneous until boundary validation.
    Any,
)

import numpy as np
import pyarrow.parquet as pq

MAX_ZIP_BYTES = 20_000_000
MAX_ZIP_MEMBERS = 256
MAX_UNCOMPRESSED_BYTES = 512_000_000
MAX_MEMBER_BYTES = 256_000_000
VALID_BYTES = 5_615_509_128
VALID_FOOTER_SHA256 = "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b"
VALID_ROWS = 37_460_000
VALID_SCORED_ROWS = 3_528_273
VALID_SEQUENCES = 1_873
SEQUENCE_LENGTH = 20_000


@dataclass(frozen=True)
class SequenceData:
    """One validated complete validation sequence in competition column order."""

    row_group: int
    sequence_id: int
    features: np.ndarray
    need_prediction: np.ndarray
    is_scored: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class SequenceReplay:
    """Retained outputs and measured official callback time for one sequence."""

    base: np.ndarray
    lstm: np.ndarray
    blended: np.ndarray
    scored_steps: np.ndarray
    callback_ns: int


def read_sequence(
    parquet: pq.ParquetFile,
    row_group: int,
    feature_columns: list[str],
    target_columns: tuple[str, ...],
    validate_fn: Callable[[Any], tuple[int, np.ndarray]],
) -> SequenceData:
    """Read and validate one full row group using the supplied official contract."""
    columns = [
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        *feature_columns,
        *target_columns,
    ]
    table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
    sequence_id, need = validate_fn(table)
    features = np.column_stack([table[name].to_numpy() for name in feature_columns]).astype(
        np.float32, copy=False
    )
    targets = np.column_stack([table[name].to_numpy() for name in target_columns]).astype(
        np.float32, copy=False
    )
    scored = table["is_scored"].to_numpy().astype(bool, copy=False)
    if features.shape != (SEQUENCE_LENGTH, len(feature_columns)):
        raise ValueError(f"unexpected raw feature shape {features.shape}")
    if targets.shape != (SEQUENCE_LENGTH, len(target_columns)):
        raise ValueError(f"unexpected target shape {targets.shape}")
    if not np.isfinite(features).all() or not np.isfinite(targets[scored & need]).all():
        raise ValueError(f"non-finite feature or selected target in row group {row_group}")
    if np.any(scored & ~need):
        raise ValueError(f"scored warm-up row in row group {row_group}")
    return SequenceData(row_group, sequence_id, features, need, scored, targets)


def replay_sequence(
    model: Any,
    data: SequenceData,
    datapoint_type: type,
    baseline_weight: float,
    lstm_weight: float,
    *,
    retain_unscored_components: bool = True,
    verify_blend: bool = True,
) -> SequenceReplay:
    """Call every raw row, retaining full diagnostics or only selected scored rows.

    Profile/state gates retain every component and verify exact blend arithmetic.
    Full validation retains only scored outputs while checking the actual callback
    on every requested row and finite components across all requested rows.
    """
    scored_steps = np.flatnonzero(data.is_scored & data.need_prediction)
    retained_rows = SEQUENCE_LENGTH if retain_unscored_components else len(scored_steps)
    base = np.empty((retained_rows, 2), dtype=np.float32)
    lstm = np.empty_like(base)
    blended = np.zeros_like(base)
    requested_rows = int(data.need_prediction.sum())
    base_requested = None
    lstm_requested = None
    if not retain_unscored_components:
        base_requested = np.empty((requested_rows, 2), dtype=np.float32)
        lstm_requested = np.empty_like(base_requested)
    callback_ns = 0
    scored_position = 0
    requested_position = 0
    for step, features in enumerate(data.features):
        started = time.perf_counter_ns()
        value = model.predict(
            datapoint_type(data.sequence_id, step, bool(data.need_prediction[step]), features)
        )
        callback_ns += time.perf_counter_ns() - started
        components = getattr(model, "last_components", None)
        if not isinstance(components, tuple) or len(components) != 2:
            raise TypeError("PredictionModel.last_components must be a (base, lstm) tuple")
        base_value, lstm_value = components
        if not isinstance(base_value, np.ndarray) or not isinstance(lstm_value, np.ndarray):
            raise TypeError("components must be NumPy arrays")
        if (
            base_value.dtype != np.float32
            or base_value.shape != (2,)
            or lstm_value.dtype != np.float32
            or lstm_value.shape != (2,)
        ):
            raise TypeError(
                f"components must be float32 vectors, got {base_value.dtype}/{lstm_value.dtype}"
            )
        mixed = None
        if retain_unscored_components or verify_blend:
            mixed = (
                baseline_weight * base_value.astype(np.float64)
                + lstm_weight * lstm_value.astype(np.float64)
            ).astype(np.float32)
            if retain_unscored_components:
                blended[step] = mixed
        if data.need_prediction[step]:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != (2,)
            ):
                raise TypeError(
                    "predict must return a float32 ndarray with shape (2,) on requested rows"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite callback output at requested row {step}")
            if retain_unscored_components:
                base[step], lstm[step] = base_value, lstm_value
            else:
                assert base_requested is not None and lstm_requested is not None
                base_requested[requested_position] = base_value
                lstm_requested[requested_position] = lstm_value
                requested_position += 1
                if data.is_scored[step]:
                    base[scored_position] = base_value
                    lstm[scored_position] = lstm_value
                    blended[scored_position] = value
                    scored_position += 1
            if verify_blend and mixed is not None and not np.array_equal(value, mixed):
                raise ValueError(f"callback output does not reproduce its components at row {step}")
        elif value is not None:
            raise ValueError("predict must return None when need_prediction is false")
        elif retain_unscored_components:
            base[step], lstm[step] = base_value, lstm_value
    if retain_unscored_components:
        if (
            not np.isfinite(base).all()
            or not np.isfinite(lstm).all()
            or not np.isfinite(blended).all()
        ):
            raise ValueError("non-finite component during full-sequence diagnostics")
    elif (
        requested_position != requested_rows
        or scored_position != len(scored_steps)
        or not np.isfinite(base_requested).all()
        or not np.isfinite(lstm_requested).all()
    ):
        raise ValueError("invalid or non-finite requested component in compact replay")
    return SequenceReplay(base, lstm, blended, scored_steps, callback_ns)


def sha256_file(path: Path) -> str:
    """Hash a file with a bounded 1 MiB read buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 hex digest of an in-memory value."""
    return hashlib.sha256(value).hexdigest()


def load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    """Read JSON only after checking the externally supplied digest."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parquet_footer_identity(path: Path) -> dict[str, int | str]:
    """Hash the Parquet metadata footer and trailer without reading data pages."""
    size = path.stat().st_size
    if size < 12:
        raise ValueError("Parquet file is too small to contain a footer")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > size - 8:
            raise ValueError("invalid Parquet footer trailer")
        stream.seek(-8 - footer_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = footer_bytes + 8
        while remaining:
            chunk = stream.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("truncated Parquet footer")
            digest.update(chunk)
            remaining -= len(chunk)
    return {"bytes": size, "footer_bytes": footer_bytes, "footer_sha256": digest.hexdigest()}


def safe_extract(archive_path: Path, destination: Path) -> dict[str, str]:
    """Verify a small ZIP and extract regular relative members without traversal."""
    if archive_path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"submission ZIP exceeds {MAX_ZIP_BYTES} bytes")
    destination.mkdir(parents=True, exist_ok=False)
    member_hashes: dict[str, str] = {}
    total_uncompressed = 0
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ZIP_MEMBERS:
            raise ValueError("submission ZIP has an empty or excessive member list")
        for info in infos:
            raw_name = info.filename
            if not raw_name or "\\" in raw_name or "\x00" in raw_name:
                raise ValueError(f"unsafe ZIP member name: {raw_name!r}")
            name = PurePosixPath(raw_name)
            if name.is_absolute() or any(part in {"", ".", ".."} for part in name.parts):
                raise ValueError(f"unsafe ZIP member path: {raw_name!r}")
            if re.match(r"^[A-Za-z]:", raw_name):
                raise ValueError(f"drive-qualified ZIP member path: {raw_name!r}")
            normalized = name.as_posix().rstrip("/")
            if normalized in seen:
                raise ValueError(f"duplicate ZIP member: {normalized!r}")
            seen.add(normalized)
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member is not allowed: {raw_name!r}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise ValueError(f"non-regular ZIP member is not allowed: {raw_name!r}")
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"ZIP member exceeds the extraction bound: {raw_name!r}")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("submission ZIP exceeds the total uncompressed-size bound")
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > MAX_MEMBER_BYTES:
                        raise ValueError(
                            f"ZIP member expanded beyond its declaration: {raw_name!r}"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if written != info.file_size:
                raise ValueError(f"ZIP member size changed while extracting: {raw_name!r}")
            member_hashes[normalized] = digest.hexdigest()
    if "solution.py" not in member_hashes:
        raise ValueError("submission ZIP must contain root-level solution.py")
    if "utils.py" in member_hashes:
        raise ValueError("submission ZIP may not shadow the official utils.py ABI")
    return member_hashes


def decode_mount_path(value: str) -> str:
    """Decode Linux mountinfo octal escapes for path comparison."""
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def output_storage(output_dir: Path) -> dict[str, str]:
    """Require a mounted, persistent output path instead of a container temp layer."""
    resolved = output_dir.resolve(strict=True)
    mounts: list[tuple[int, str, str, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_path = decode_mount_path(fields[1])
        try:
            common = os.path.commonpath((str(resolved), mount_path))
        except ValueError:
            continue
        if common == mount_path:
            mounts.append((len(mount_path), mount_path, fields[2], fields[0]))
    if not mounts:
        raise RuntimeError("cannot identify the output filesystem mount")
    _length, mount_path, fs_type, source = max(mounts)
    if fs_type.lower() in {"tmpfs", "ramfs", "devtmpfs", "overlay", "aufs"}:
        raise RuntimeError(f"output path is on non-persistent filesystem {fs_type!r}")
    if mount_path == "/":
        raise RuntimeError("output directory is not covered by a dedicated persistent mount")
    return {"mount_path": mount_path, "filesystem": fs_type, "source": source}


def runtime_envelope() -> dict[str, Any]:
    """Fail closed unless the Linux receiver has one CPU, 16 GiB RAM, no GPU or route."""
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux":
        raise RuntimeError("receiver requires CPython 3.11 on Linux")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError(f"receiver requires x86_64, found {platform.machine()}")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(f"receiver must have exactly one allowed CPU, found {affinity}")
    cgroup_v2 = Path("/sys/fs/cgroup")
    cpu_max = cgroup_v2 / "cpu.max"
    memory_max = cgroup_v2 / "memory.max"
    if not cpu_max.is_file() or not memory_max.is_file():
        raise RuntimeError("receiver requires readable cgroup v2 CPU and memory limits")
    quota, period = cpu_max.read_text().split()
    memory_limit = memory_max.read_text().strip()
    if quota == "max" or int(quota) != int(period):
        raise RuntimeError("receiver CPU quota must equal one CPU")
    if memory_limit == "max" or int(memory_limit) != 16_000_000_000:
        raise RuntimeError("receiver memory limit must equal 16,000,000,000 bytes")
    routes = Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]
    if any(fields[1] == "00000000" for line in routes if len(fields := line.split()) >= 4):
        raise RuntimeError("receiver has a default IPv4 network route")
    ipv6_routes = Path("/proc/net/ipv6_route")
    if ipv6_routes.is_file():
        for line in ipv6_routes.read_text(encoding="ascii").splitlines():
            fields = line.split()
            is_default = len(fields) >= 10 and fields[0] == "0" * 32 and fields[1] == "00"
            is_loopback = len(fields) >= 10 and fields[9] == "lo"
            if is_default and not is_loopback:
                raise RuntimeError("receiver has a default IPv6 network route")
    gpu_devices = [
        str(path)
        for pattern in ("/dev/nvidia*", "/dev/dri/renderD*")
        for path in Path("/").glob(pattern.lstrip("/"))
    ]
    if gpu_devices or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError(f"receiver must be CPU-only, found GPU devices {gpu_devices}")
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "python_abi": sys.implementation.cache_tag,
        "cpu_affinity": affinity,
        "cpu_quota": {"quota": int(quota), "period": int(period)},
        "memory_limit_bytes": int(memory_limit),
        "network_default_route": False,
        "gpu_devices": gpu_devices,
    }


def write_json_once(path: Path, value: dict[str, Any]) -> str:
    """Persist one JSON receipt without replacing earlier evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()
