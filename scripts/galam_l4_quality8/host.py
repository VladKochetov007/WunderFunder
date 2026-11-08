from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wnn.training.sequence_compile import COMPILER_RECOMPILE_LIMIT
from wnn.training.sequence_precision import precision_identity

from .common import LaunchError, parse_cpu_list
from .contract import CellPlan

GPU_QUERY = "index,name,uuid,memory.total,memory.free"
CUDA_QUERY_SCRIPT = r'''
import json
import torch
from wnn.training.sequence_monitor_identity import gpu_identifier

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("expected eight CUDA-visible devices")
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    if isinstance(raw_uuid, bytes):
        raw_uuid = raw_uuid.decode("utf-8")
    if raw_uuid is None:
        raise SystemExit("CUDA device UUID is unavailable")
    physical_uuid = gpu_identifier(torch.device("cuda", index))
    if physical_uuid is None:
        raise SystemExit("CUDA physical UUID could not be normalized")
    devices.append({
        "name": torch.cuda.get_device_name(index),
        "uuid": str(raw_uuid),
        "physical_uuid": physical_uuid,
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
    })
print(json.dumps(devices, sort_keys=True))
'''
EXPECTED_TORCH = "2.8.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class HostState:
    """Receiver facts needed to admit every profile and assign its isolated process."""

    gpus: dict[int, dict[str, Any]]
    cuda_devices: dict[str, dict[str, Any]]
    driver: str
    available_cpus: frozenset[int]
    numa_cpus: dict[int, frozenset[int]]


def probe_host() -> HostState:
    """Read GPU, driver, CPU-affinity, and NUMA facts from the Linux receiver."""
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise LaunchError("the fixed-eight launcher must run on the Linux receiving host")
    gpu_result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    gpus: dict[int, dict[str, Any]] = {}
    for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            raise LaunchError("nvidia-smi returned an unexpected GPU inventory row")
        index, name, uuid, total_mib, free_mib = row
        gpu_id = int(index.strip())
        if gpu_id in gpus:
            raise LaunchError("nvidia-smi returned duplicate GPU indices")
        gpus[gpu_id] = {
            "name": name.strip(),
            "uuid": uuid.strip(),
            "total_mib": int(total_mib.strip()),
            "free_mib": int(free_mib.strip()),
        }
    driver_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    drivers = {line.strip() for line in driver_result.stdout.splitlines() if line.strip()}
    if len(gpus) != 8 or set(gpus) != set(range(8)) or len(drivers) != 1:
        raise LaunchError("receiver must expose eight indexed GPUs with one driver version")
    if any(gpu["name"] != "NVIDIA L4" or not gpu["uuid"].startswith("GPU-") for gpu in gpus.values()):
        raise LaunchError("all eight receiver GPUs must be NVIDIA L4 devices with UUIDs")
    cuda_devices = _query_cuda_devices()
    if set(cuda_devices) != {gpu["uuid"] for gpu in gpus.values()}:
        raise LaunchError("CUDA UUID inventory differs from the eight physical L4 UUIDs")

    numa_cpus: dict[int, frozenset[int]] = {}
    for node in (0, 1):
        path = Path(f"/sys/devices/system/node/node{node}/cpulist")
        try:
            numa_cpus[node] = frozenset(parse_cpu_list(path.read_text(encoding="utf-8").strip()))
        except OSError as error:
            raise LaunchError(f"NUMA node {node} is unavailable on the receiver") from error
    return HostState(
        gpus=gpus,
        cuda_devices=cuda_devices,
        driver=next(iter(drivers)),
        available_cpus=frozenset(os.sched_getaffinity(0)),
        numa_cpus=numa_cpus,
    )


def validate_host_profiles(cells: list[CellPlan], host: HostState) -> dict[str, dict[str, Any]]:
    """Require each approved profile to match this Torch 2.8 CUDA 12.8 L4 host."""
    if str(torch.__version__) != EXPECTED_TORCH or str(torch.version.cuda) != EXPECTED_CUDA:
        raise LaunchError("isolated receiver interpreter must be Torch 2.8.0+cu128")
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not set(cell.cpus).issubset(host.available_cpus):
            raise LaunchError(f"{cell.cell_id}: panel CPUs are outside the process affinity mask")
        if not set(cell.cpus).issubset(host.numa_cpus[cell.numa_node]):
            raise LaunchError(f"{cell.cell_id}: panel CPU set disagrees with its NUMA node")
        gpu = host.gpus[cell.gpu]
        cuda_device = host.cuda_devices.get(gpu["uuid"])
        if cuda_device is None:
            raise LaunchError(f"{cell.cell_id}: assigned GPU UUID is not CUDA-visible")
        profile_runtime = cell.profile["identity"].get("runtime")
        if not isinstance(profile_runtime, dict):
            raise LaunchError(f"{cell.cell_id}: profile runtime identity is missing")
        if profile_runtime.get("nvidia_driver") != host.driver:
            raise LaunchError(f"{cell.cell_id}: profile NVIDIA driver differs from receiver")
        devices = profile_runtime.get("devices")
        if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict):
            raise LaunchError(f"{cell.cell_id}: fixed fits require a one-device profile")
        profile_device = devices[0]
        expected_memory = profile_device.get("total_memory_bytes")
        if (
            profile_device.get("rank") != 0
            or profile_device.get("local_rank") != 0
            or profile_device.get("uuid") != cuda_device["uuid"]
            or cuda_device.get("physical_uuid") != gpu["uuid"]
            or profile_device.get("name") != cuda_device["name"]
            or profile_device.get("capability") != cuda_device["capability"]
            or not isinstance(expected_memory, int)
            or expected_memory != cuda_device["total_memory_bytes"]
        ):
            raise LaunchError(f"{cell.cell_id}: profile GPU identity differs from assigned L4 UUID")
        if (
            profile_runtime.get("python") != platform.python_version()
            or profile_runtime.get("platform") != platform.platform()
            or profile_runtime.get("torch") != str(torch.__version__)
            or profile_runtime.get("cuda_runtime") != str(torch.version.cuda)
            or profile_runtime.get("cudnn") != torch.backends.cudnn.version()
            or profile_runtime.get("visible_devices") != 1
            or profile_runtime.get("world_size") != 1
            or profile_runtime.get("matmul_precision") != "highest"
            or profile_runtime.get("matmul_tf32") is not False
            or profile_runtime.get("cudnn_tf32") is not False
            or profile_runtime.get("autocast") is not False
            or profile_runtime.get("amp_scaler") is not False
            or profile_runtime.get("cublas_workspace_config") != EXPECTED_CUBLAS_WORKSPACE
        ):
            raise LaunchError(f"{cell.cell_id}: profile software or strict-FP32 runtime differs")
        raw_runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_driver": host.driver,
            "visible_devices": 1,
            "world_size": 1,
            "matmul_precision": "highest",
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "autocast": False,
            "amp_scaler": False,
            "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE,
            "devices": [
                {
                    "rank": 0,
                    "local_rank": 0,
                    "name": cuda_device["name"],
                    "capability": cuda_device["capability"],
                    "uuid": cuda_device["uuid"],
                    "total_memory_bytes": expected_memory,
                }
            ],
        }
        runtime = _runtime_precision(cell, raw_runtime, profile_runtime)
        if runtime != profile_runtime:
            raise LaunchError(f"{cell.cell_id}: reconstructed runtime does not equal its profile")
        result[cell.cell_id] = runtime
    return result


def _query_cuda_devices() -> dict[str, dict[str, Any]]:
    """Query exact CUDA UUID and allocatable-memory identities in a short-lived process."""
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("RANK", None)
    env.pop("WORLD_SIZE", None)
    env.pop("LOCAL_RANK", None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", CUDA_QUERY_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    try:
        devices = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise LaunchError("CUDA inventory process returned invalid JSON") from error
    if not isinstance(devices, list) or len(devices) != 8:
        raise LaunchError("CUDA inventory process did not report exactly eight devices")
    result_by_uuid: dict[str, dict[str, Any]] = {}
    seen_raw_uuids: set[str] = set()
    for item in devices:
        if not isinstance(item, dict):
            raise LaunchError("CUDA inventory contains a malformed device record")
        uuid = item.get("uuid")
        physical_uuid = item.get("physical_uuid")
        if (
            not isinstance(uuid, str)
            or uuid in seen_raw_uuids
            or not isinstance(physical_uuid, str)
            or physical_uuid in result_by_uuid
        ):
            raise LaunchError("CUDA inventory contains a missing or duplicate UUID")
        if (
            item.get("name") != "NVIDIA L4"
            or item.get("capability") != [8, 9]
            or type(item.get("total_memory_bytes")) is not int
            or item["total_memory_bytes"] <= 0
        ):
            raise LaunchError("CUDA inventory contains a non-L4 or invalid memory identity")
        if not physical_uuid.startswith("GPU-"):
            raise LaunchError("CUDA inventory has an invalid physical UUID representation")
        seen_raw_uuids.add(uuid)
        result_by_uuid[physical_uuid] = item
    return result_by_uuid


def _runtime_precision(
    cell: CellPlan, raw_runtime: dict[str, Any], profile_runtime: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild the recipe precision/compiler portion of the recorded runtime identity."""
    expected_precision = profile_runtime.get("precision")
    if not isinstance(expected_precision, dict):
        raise LaunchError(f"{cell.cell_id}: profile precision identity is missing")
    emulation = expected_precision.get("cublas_emulate_single_precision")
    if emulation not in (None, "0"):
        raise LaunchError(f"{cell.cell_id}: strict FP32 profile has a cuBLAS emulation flag")
    backend = str(cell.config["model"].get("attention_backend", "math"))
    precision = precision_identity(
        str(cell.config["precision_mode"]),
        backend,
        environ={"CUBLAS_EMULATE_SINGLE_PRECISION": emulation}
        if emulation is not None
        else {},
    )
    compiled = bool(cell.config["model"].get("compile_blocks", False))
    compiler = {
        "enabled": compiled,
        "backend": "inductor" if compiled else None,
        "mode": "default" if compiled else None,
        "fullgraph": False,
        "dynamic": None,
        "recompile_limit": COMPILER_RECOMPILE_LIMIT if compiled else None,
    }
    return raw_runtime | {"attention_backend": backend, "precision": precision, "compiler": compiler}
