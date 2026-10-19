"""Small create-only receipts and strict B-side runtime resource guards."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from pathlib import Path

HIGH_BYTES = 1_400_000_000
RESERVE_BYTES = 256 * 1024**2
SERVICE = "computer-vision-homework-eda19-r16-v1.service"


def sha256_file(path: Path) -> str:
    """Hash one file in bounded memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    """Create a durable finite JSON receipt and return its exact-byte SHA-256."""
    payload = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def initialize_progress(path: Path) -> None:
    """Create an empty JSONL stream exactly once before appending events."""
    with path.open("xb"):
        pass


def append_json_line(path: Path, value: object) -> None:
    """Append one durable compact JSONL event."""
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _own_cgroup_memory_current() -> int:
    """Read only this exact R16 service's unified cgroup memory counter."""
    matches: list[Path] = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[:2] != ["0", ""]:
            continue
        relative = Path(fields[2].lstrip("/"))
        if not relative.parts or ".." in relative.parts or relative == Path("."):
            raise RuntimeError("R16 process has an invalid or root cgroup path")
        if "wnn-campaign.slice" not in relative.parts or relative.name != SERVICE:
            raise RuntimeError("R16 process is outside its exact reviewed cgroup")
        matches.append(Path("/sys/fs/cgroup") / relative / "memory.current")
    if len(matches) != 1:
        raise RuntimeError("R16 requires exactly one unified cgroup membership")
    return int(matches[0].read_text().strip())


def memory_guard(stage: str) -> dict[str, int | str]:
    """Require the fixed 256 MiB reserve and report process/cgroup memory."""
    current = _own_cgroup_memory_current()
    if current > HIGH_BYTES - RESERVE_BYTES:
        raise MemoryError(f"R16 memory reserve unavailable at {stage}: {current} bytes")
    rss = next(
        (
            int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        ),
        None,
    )
    if rss is None:
        raise RuntimeError("cannot read R16 process RSS")
    return {
        "stage": stage,
        "own_cgroup_memory_current_bytes": current,
        "process_rss_bytes": rss,
        "process_peak_rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_high_bytes": HIGH_BYTES,
        "reserve_bytes": RESERVE_BYTES,
    }


def physical_output(output: Path, scratch: Path) -> None:
    """Require a new direct child of physical B research scratch."""
    output, scratch = output.resolve(), scratch.resolve()
    if output.parent != scratch or output.exists():
        raise ValueError("R16 output must be a new direct child of B/research_tmp")
    mounts: list[tuple[int, str]] = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        point = fields[1].replace("\\040", " ")
        if output.is_relative_to(Path(point)):
            mounts.append((len(point), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R16 output must be on physical research scratch")


def archive_parameters(path: Path, arrays: dict[str, object]) -> dict[str, object]:
    """Create a private numeric NPZ and return metadata without array values."""
    import numpy as np

    numeric = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype == object or not np.isfinite(value).all() for value in numeric.values()):
        raise ValueError("R16 parameter archive contains invalid or nonfinite arrays")
    with path.open("xb") as stream:
        np.savez(stream, **numeric)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path_name": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(numeric.items())
        },
        "contains_predictions": False,
    }
