"""Create-only JSON, archive, physical-output, and exact-cgroup guards for R14."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from pathlib import Path

import numpy as np

HIGH_BYTES = 1_400_000_000
RESERVE_BYTES = 256 * 1024**2
SERVICE = "computer-vision-homework-eda19-r14-v1.service"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    """Create one finite JSON record and return its exact-byte SHA256."""
    encoded = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def append_json_line(path: Path, value: object) -> None:
    """Append one compact JSONL event without rewriting prior progress."""
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _own_cgroup_current() -> int:
    """Read only this exact R14 service's unified cgroup counter."""
    memberships = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[:2] != ["0", ""]:
            continue
        relative = Path(fields[2].lstrip("/"))
        if not relative.parts or relative == Path(".") or ".." in relative.parts:
            raise RuntimeError("R14 self-cgroup path is empty, root, or invalid")
        if "wnn-campaign.slice" not in relative.parts or relative.name != SERVICE:
            raise RuntimeError("R14 process is outside the exact reviewed service")
        memberships.append(Path("/sys/fs/cgroup") / relative / "memory.current")
    if len(memberships) != 1:
        raise RuntimeError("expected one exact unified R14 cgroup membership")
    return int(memberships[0].read_text().strip())


def memory_guard(stage: str) -> dict[str, int | str]:
    """Require the fixed 256 MiB reserve under R14's 1.4 GB memory high."""
    current = _own_cgroup_current()
    if current > HIGH_BYTES - RESERVE_BYTES:
        raise MemoryError(f"R14 memory reserve unavailable at {stage}: {current} bytes")
    rss = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
            break
    if rss is None:
        raise RuntimeError("cannot read process RSS")
    return {
        "stage": stage,
        "own_cgroup_memory_current_bytes": current,
        "process_rss_bytes": rss,
        "process_peak_rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "high_bytes": HIGH_BYTES,
        "reserve_bytes": RESERVE_BYTES,
    }


def physical_output(output: Path, scratch: Path) -> None:
    """Require output to be a new direct child of physical research scratch."""
    if output.parent != scratch or output.exists():
        raise ValueError("R14 output must be a new direct child of B/research_tmp")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mountpoint = fields[1].replace("\\040", " ")
        if output.is_relative_to(Path(mountpoint)):
            mounts.append((len(mountpoint), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R14 output filesystem must be physical B scratch")


def archive_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Create one B-only numeric archive and return its digest and metadata."""
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path_name": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(arrays.items())
        },
        "contains_predictions": False,
    }
