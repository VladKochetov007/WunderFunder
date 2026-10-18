"""Container, GPU, runtime, and actor-envelope admission checks."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from policy import (
    EXPECTED_CUDA,
    EXPECTED_PIDS,
    EXPECTED_RAM,
    EXPECTED_TORCH,
    GPU_FREE_MIN,
    parse_gpu_query,
    validate_gpu_inventory,
)


def cgroup_value(name: str) -> str:
    relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    path = Path("/sys/fs/cgroup") / relative.lstrip("/") / name
    if not path.is_file():
        raise RuntimeError(f"required cgroup-v2 control is missing: {path}")
    return path.read_text(encoding="ascii").strip()


def cgroup_snapshot() -> dict[str, Any]:
    """Read the controller's container-wide cgroup v2 limits and event counters."""
    memory_max = int(cgroup_value("memory.max"))
    swap_max = int(cgroup_value("memory.swap.max"))
    pids_max = int(cgroup_value("pids.max"))
    relative = next(line.split(":",2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    own = Path("/sys/fs/cgroup") / relative.lstrip("/")
    quota_path = own / "cpu.max"
    cpu_quota, cpu_period = quota_path.read_text().split() if quota_path.exists() else ("max", "100000")
    ancestors = []
    node = own
    while node.is_relative_to(Path("/sys/fs/cgroup")):
        ancestors.append({"path":str(node), **{name:(node/name).read_text().strip() if (node/name).is_file() else None for name in ("cpu.max","memory.max","memory.swap.max","pids.max")}})
        if node == Path("/sys/fs/cgroup"):
            break
        node = node.parent
    events = {}
    for line in cgroup_value("memory.events").splitlines():
        key, value = line.split()
        events[key] = int(value)
    return {
        "memory_max": memory_max,
        "memory_current": int(cgroup_value("memory.current")),
        "memory_peak": int(cgroup_value("memory.peak")),
        "swap_max": swap_max,
        "pids_max": pids_max,
        "cpu_quota": None if cpu_quota == "max" else int(cpu_quota),
        "cpu_period": int(cpu_period),
        "cpu_count": len(os.sched_getaffinity(0)) if cpu_quota == "max" else min(len(os.sched_getaffinity(0)),int(cpu_quota)/int(cpu_period)),
        "cpu_enforcement": "kernel_affinity_only" if cpu_quota == "max" else "cgroup_quota_and_affinity",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_path": str(own),
        "ancestors": ancestors,
        "memory_events": events,
        "cpu_stat": {
            key: int(value)
            for key, value in (line.split() for line in ((own/"cpu.stat").read_text().splitlines() if (own/"cpu.stat").exists() else []))
        },
    }


def validate_container(snapshot: dict[str, Any]) -> None:
    """Require the externally provisioned finite CPU/RAM/swap/pid envelope."""
    if snapshot["memory_max"] != EXPECTED_RAM:
        raise RuntimeError("container memory.max differs from the exact 192-GiB envelope")
    if snapshot["swap_max"] != 0 or snapshot["pids_max"] != EXPECTED_PIDS:
        raise RuntimeError("container swap or pids.max differs from the authorized envelope")
    if snapshot["cpu_count"] != 16:
        raise RuntimeError("container CPU quota is not exactly 16 CPUs")
    affinity = os.sched_getaffinity(0)
    if set(range(16)) - affinity:
        raise RuntimeError("controller cannot access the frozen CPU affinity IDs 0 through 15")


def gpu_inventory() -> dict[str, dict[str, int | str]]:
    """Read all visible GPU UUIDs and free memory without touching other processes."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return parse_gpu_query(result.stdout)


def require_gpus(jobs: list[dict[str, Any]], only_uuid: str | None = None) -> dict[str, Any]:
    inventory = gpu_inventory()
    if only_uuid is None:
        validate_gpu_inventory(inventory, jobs)
    else:
        if only_uuid not in inventory:
            raise RuntimeError(f"assigned GPU {only_uuid} is not visible")
        row = inventory[only_uuid]
        if "B300" not in str(row["name"]) or int(row["free_mib"]) < GPU_FREE_MIN:
            raise RuntimeError(f"assigned L4 {only_uuid} has less than 8 GiB free")
    return inventory


def validate_python_runtime() -> dict[str, Any]:
    """Pin Python and CUDA torch versions in the controller's own venv."""
    import torch

    version = {
        "python": "{}.{}.{}".format(*__import__("sys").version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_device_count": torch.cuda.device_count(),
    }
    if version["python"] != "3.12.14":
        raise RuntimeError(f"Python runtime mismatch: {version['python']}")
    if version["torch"] != EXPECTED_TORCH or version["cuda"] != EXPECTED_CUDA:
        raise RuntimeError(f"Torch/CUDA runtime mismatch: {version}")
    if torch.cuda.device_count() != 8:
        raise RuntimeError("controller process must see all eight assigned L4s")
    if any("B300" not in torch.cuda.get_device_name(index) for index in range(8)):
        raise RuntimeError("one or more visible CUDA devices are not L4s")
    return version


def _gpu_process_memory() -> dict[tuple[str, int], int]:
    """Sample NVML-backed nvidia-smi process usage for controller-owned actor PIDs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    usage = {}
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"malformed nvidia-smi compute-process row: {line!r}")
        usage[(fields[0], int(fields[1]))] = int(fields[2])
    return usage


def parse_process_envelope(path: Path, uuid: str, cpus: list[int]) -> dict[str, Any]:
    """Validate the actor wrapper's exact runtime and per-process resource receipt."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "tabdata-b300-process-envelope.v1",
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_CUDA,
        "visible_devices": uuid,
        "cpu_affinity": sorted(cpus),
        "allocator_budget_bytes": 6 * 1024**3,
        "allocator_limit_is_not_a_device_memory_reservation": True,
        "status": "actor_returned",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"process resource receipt {key} differs: {receipt.get(key)!r}")
    if "B300" not in receipt.get("gpu", ""):
        raise RuntimeError("process receipt GPU is not an L4")
    for key in ("elapsed_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = receipt.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(f"process receipt {key} is missing or invalid")
    return receipt
