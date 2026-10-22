from __future__ import annotations

# Any represents the schema-versioned telemetry record consumed by receiver audits.
import csv
import subprocess
from typing import Any

import torch


def capture_execution_runtime(
    device: torch.device, precision: str | None, stage: str
) -> dict[str, Any]:
    """Capture source-process CUDA identity, driver, strict-FP32 flags and memory."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("execution runtime receipt requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    name = torch.cuda.get_device_name(index)
    if "b300" not in name.lower():
        raise RuntimeError(f"execution runtime receipt requires B300, got {name!r}")
    torch_uuid = getattr(properties, "uuid", None)
    torch_uuid_text = "" if torch_uuid is None else str(torch_uuid)
    if not torch_uuid_text.strip():
        raise RuntimeError("PyTorch does not expose the active CUDA device UUID")

    def normalize_uuid(value: str) -> str:
        """Normalize PyTorch and nvidia-smi UUID spellings for comparison."""
        return value.strip().casefold().removeprefix("gpu-")

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    records = list(csv.reader(query.stdout.splitlines(), skipinitialspace=True))
    selected = next(
        (
            row
            for row in records
            if len(row) == 5 and normalize_uuid(row[2]) == normalize_uuid(torch_uuid_text)
        ),
        None,
    )
    if selected is None:
        raise RuntimeError("nvidia-smi UUID does not match PyTorch's active CUDA device UUID")
    gpu_index, driver_name, uuid, driver, memory_mib = (value.strip() for value in selected)
    if "b300" not in driver_name.lower() or not uuid or not driver:
        raise RuntimeError("nvidia-smi did not return a B300 UUID and driver version")
    total_bytes = int(properties.total_memory)
    driver_total_bytes = int(memory_mib) * (1 << 20)
    free_bytes, runtime_total_bytes = torch.cuda.mem_get_info(index)
    if min(total_bytes, driver_total_bytes, int(runtime_total_bytes)) <= 0:
        raise RuntimeError("CUDA and nvidia-smi must report positive device memory totals")
    matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    try:
        autocast_active = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_active = bool(torch.is_autocast_enabled())
    if precision == "fp32" and (
        matmul_tf32 or cudnn_tf32 or autocast_active or torch.get_default_dtype() != torch.float32
    ):
        raise RuntimeError("strict-FP32 execution runtime has a reduced-precision path enabled")
    return {
        "schema_version": 1,
        "capture_stage": stage,
        "device": name,
        "cuda_device_index": int(index),
        "nvidia_smi_index": gpu_index,
        "device_uuid": normalize_uuid(torch_uuid_text),
        "nvidia_smi_device_uuid": normalize_uuid(uuid),
        "driver_name": driver_name,
        "driver_version": driver,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_total_memory_bytes": total_bytes,
        "nvidia_smi_total_memory_bytes": driver_total_bytes,
        "cuda_free_memory_bytes_at_capture": int(free_bytes),
        "cuda_runtime_total_memory_bytes": int(runtime_total_bytes),
        "precision": precision,
        "autocast_policy": "disabled" if precision == "fp32" else None,
        "autocast_active_at_capture": autocast_active,
        "matmul_tf32": matmul_tf32,
        "cudnn_tf32": cudnn_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
    }
