"""Create-only runtime guards and private model-archive writing for R7."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from .model_math import MeanVarianceMLP

HIGH_BYTES = 1_400_000_000
MEMORY_RESERVE_BYTES = 256 * 1024**2


def initialize_output(
    output: Path,
    scratch: Path,
    sample_path: Path,
    identities: dict[str, str],
    preparation_sha256: str,
) -> tuple[Path, float, float]:
    """Create a physical output directory with initial START and JSONL evidence."""
    physical_output(output, scratch)
    output.mkdir()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    write_json(
        output / "START.json",
        {
            "schema": "nf19.r7_mean_variance.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "sample_path": str(sample_path),
            "output_path": str(output),
        },
    )
    progress_path = output / "PROGRESS.jsonl"
    with progress_path.open("xb"):
        pass
    append_json_line(
        progress_path,
        {"event": "initialized", "elapsed_wall_s": 0.0, "elapsed_cpu_s": 0.0},
    )
    return progress_path, started_wall, started_cpu


def write_json(path: Path, value: object) -> str:
    """Create one finite JSON evidence file and return its SHA256."""
    encoded = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def append_json_line(path: Path, value: object) -> None:
    """Append one compact, durable progress record."""
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_cgroup_memory(root: Path, membership_text: str) -> int:
    """Read only the memory counter for the exact unified cgroup membership."""
    memberships = []
    for line in membership_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and not fields[1]:
            relative = Path(fields[2].lstrip("/"))
            if ".." in relative.parts:
                raise ValueError("invalid unified cgroup membership path")
            memberships.append(root / relative / "memory.current")
    if len(memberships) != 1:
        raise RuntimeError("expected one unified self-cgroup membership")
    return int(memberships[0].read_text().strip())


def _cgroup_memory_current() -> int:
    """Read the current process's exact unified cgroup memory counter."""
    membership = Path("/proc/self/cgroup").read_text()
    return _read_cgroup_memory(Path("/sys/fs/cgroup"), membership)


def _process_rss() -> int:
    """Read current process RSS from procfs in bytes."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read process RSS from /proc/self/status")


def memory_guard(stage: str) -> dict[str, int | str]:
    """Keep 256 MiB under the fixed 1.4 GB high memory bound."""
    current = _cgroup_memory_current()
    if current > HIGH_BYTES - MEMORY_RESERVE_BYTES:
        raise MemoryError(f"R7 memory reserve unavailable before {stage}: {current} bytes")
    return {
        "stage": stage,
        "cgroup_memory_current_bytes": current,
        "process_rss_bytes": _process_rss(),
        "process_peak_rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "high_bytes": HIGH_BYTES,
        "reserved_bytes": MEMORY_RESERVE_BYTES,
    }


def physical_output(output: Path, scratch: Path) -> None:
    """Require a new direct child of B's physical research scratch directory."""
    if output.parent != scratch or output.exists():
        raise ValueError("R7 output must be a new direct child of B/research_tmp")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mountpoint = fields[1].replace("\\040", " ")
        if output.is_relative_to(Path(mountpoint)):
            mounts.append((len(mountpoint), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R7 output filesystem must be physical B scratch")


def archive_models(
    path: Path,
    control: MeanVarianceMLP,
    candidate: MeanVarianceMLP,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Write final model states and fit-only normalization into one private archive."""
    arrays: dict[str, np.ndarray] = {
        "normalizer_mean": mean,
        "normalizer_scale": scale,
    }
    for arm, model in (("control", control), ("heteroskedastic", candidate)):
        for name, tensor in model.state_dict().items():
            arrays[f"{arm}__{name.replace('.', '__')}"] = tensor.detach().cpu().numpy()
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path_name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "array_names": sorted(arrays),
        "predictions_in_archive": False,
    }
