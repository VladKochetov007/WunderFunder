"""Combo-specific resource checks and create-only receipts."""

from __future__ import annotations

import hashlib
import resource
from pathlib import Path

import numpy as np
from idea_screens_r15.runtime_support import (
    append_json_line,
    archive_arrays,
    physical_output,
    write_json,
)

HIGH_BYTES = 1_400_000_000
RESERVE_BYTES = 256 * 1024**2
SERVICE = "computer-vision-homework-idea-combo-r1-v1.service"


def sha256_file(path: Path) -> str:
    """Hash a file in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def own_cgroup_current_bytes() -> int:
    """Read memory.current only for the exact reviewed unit cgroup."""
    paths: list[Path] = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[:2] != ["0", ""]:
            continue
        relative = Path(fields[2].lstrip("/"))
        if not relative.parts or ".." in relative.parts or relative == Path("."):
            raise RuntimeError("combo process has an invalid or root cgroup path")
        if "wnn-campaign.slice" not in relative.parts or relative.name != SERVICE:
            raise RuntimeError("combo process is outside its exact reviewed service cgroup")
        paths.append(Path("/sys/fs/cgroup") / relative / "memory.current")
    if len(paths) != 1:
        raise RuntimeError("combo process needs one exact unified cgroup membership")
    return int(paths[0].read_text().strip())


def memory_guard(stage: str) -> dict[str, int | str]:
    """Enforce the fixed 256 MiB reserve below the 1.4 GB memory high."""
    current = own_cgroup_current_bytes()
    if current > HIGH_BYTES - RESERVE_BYTES:
        raise MemoryError(f"combo memory reserve unavailable at {stage}: {current} bytes")
    rss = next(
        (
            int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        ),
        None,
    )
    if rss is None:
        raise RuntimeError("cannot read combo process RSS")
    return {
        "stage": stage,
        "own_cgroup_memory_current_bytes": current,
        "process_rss_bytes": rss,
        "process_peak_rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_high_bytes": HIGH_BYTES,
        "reserve_bytes": RESERVE_BYTES,
    }


def append_durable_line(path: Path, value: object) -> None:
    """Delegate JSONL serialization to the pinned R15 writer."""
    append_json_line(path, value)


def write_durable_json(path: Path, value: object) -> str:
    """Delegate finite JSON serialization to the pinned R15 writer."""
    return write_json(path, value)


def create_model_archive(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Persist candidate parameters only and return bounded metadata."""
    return archive_arrays(path, arrays)


__all__ = [
    "append_durable_line",
    "create_model_archive",
    "memory_guard",
    "physical_output",
    "sha256_file",
    "write_durable_json",
]
