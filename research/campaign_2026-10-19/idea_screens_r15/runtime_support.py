"""Create-only receipts, bounded source hashing, and R15 resource guards."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from pathlib import Path

import numpy as np

HIGH_BYTES = 1_400_000_000
RESERVE_BYTES = 256 * 1024**2
SERVICE = "computer-vision-homework-eda19-r15-v1.service"


def _cpu_runtime_state() -> dict[str, object]:
    """Read the configured Torch CPU, precision and device state."""
    import torch

    return {
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "autocast_cuda_enabled": torch.is_autocast_enabled(),
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu"),
        "tf32_matmul_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_enabled": bool(torch.backends.cudnn.allow_tf32),
    }


def assert_cpu_runtime() -> dict[str, object]:
    """Fail closed unless the fresh process has the exact CPU-only settings."""
    state = _cpu_runtime_state()
    if (
        state["torch_threads"] != 1
        or state["torch_interop_threads"] != 1
        or state["cuda_initialized"]
        or state["autocast_cuda_enabled"]
        or state["autocast_cpu_enabled"]
        or state["tf32_matmul_enabled"]
        or state["tf32_cudnn_enabled"]
    ):
        raise RuntimeError(f"R15 Torch runtime settings differ from the CPU contract: {state}")
    return state


def configure_cpu_runtime() -> dict[str, object]:
    """Set deterministic CPU thread/TF32 settings and verify fail-closed flags."""
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("R15 requires CUDA to remain uninitialized")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return assert_cpu_runtime()


def sha256_file(path: Path) -> str:
    """Hash one file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    """Create a finite JSON file and return the exact-byte digest."""
    payload = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def append_json_line(path: Path, value: object) -> None:
    """Append one durable compact JSONL progress event."""
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _own_cgroup_current() -> int:
    """Read only this exact systemd unit's unified cgroup memory counter."""
    matches: list[Path] = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[:2] != ["0", ""]:
            continue
        relative = Path(fields[2].lstrip("/"))
        if not relative.parts or ".." in relative.parts or relative == Path("."):
            raise RuntimeError("R15 process has an invalid or root cgroup path")
        if "wnn-campaign.slice" not in relative.parts or relative.name != SERVICE:
            raise RuntimeError("R15 process is outside its exact reviewed cgroup")
        matches.append(Path("/sys/fs/cgroup") / relative / "memory.current")
    if len(matches) != 1:
        raise RuntimeError("R15 requires exactly one unified cgroup membership")
    return int(matches[0].read_text().strip())


def memory_guard(stage: str) -> dict[str, int | str]:
    """Enforce the 256 MiB reserve under the fixed 1.4 GB memory high."""
    current = _own_cgroup_current()
    if current > HIGH_BYTES - RESERVE_BYTES:
        raise MemoryError(f"R15 memory reserve unavailable at {stage}: {current} bytes")
    rss = next(
        (
            int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        ),
        None,
    )
    if rss is None:
        raise RuntimeError("cannot read R15 process RSS")
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
    if output.parent != scratch or output.exists():
        raise ValueError("R15 output must be a new direct child of B/research_tmp")
    mounts: list[tuple[int, str]] = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        point = fields[1].replace("\\040", " ")
        if output.is_relative_to(Path(point)):
            mounts.append((len(point), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R15 output must be on physical research scratch")


def archive_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Create one private numeric archive and return only its metadata."""
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path_name": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "arrays": {
            key: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for key, v in sorted(arrays.items())
        },
        "contains_predictions": False,
    }
