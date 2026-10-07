"""Physical NVIDIA device inventory and selected-device identity checks."""

from __future__ import annotations

import hashlib
import os
import platform
import time
from typing import Any

from support.gpu_memory_domains import (
    memory_domain_snapshot,
)

GPU_QUERY_RAW_MAX_BYTES = 65_536
GPU_QUERY_FIELDS = (
    "index,name,driver_version,uuid,memory.used,memory.total,utilization.gpu,utilization.memory"
)


def _process_output_receipt(value: str | bytes | None) -> dict[str, Any]:
    """Retain bounded query text and its exact digest/truncation metadata."""
    raw = b"" if value is None else value if isinstance(value, bytes) else value.encode("utf-8")
    return {
        "text": raw[:GPU_QUERY_RAW_MAX_BYTES].decode("utf-8", errors="replace"),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > GPU_QUERY_RAW_MAX_BYTES,
    }


def _optional_gpu_integer(value: str) -> int | None:
    """Parse integer metrics while representing NVIDIA N/A values as null."""
    normalized = value.strip()
    if not normalized or normalized.lower() in {"n/a", "na", "not supported"}:
        return None
    try:
        return int(normalized.split(maxsplit=1)[0])
    except ValueError:
        return None


def parse_nvidia_smi_gpu_rows(output: str) -> list[dict[str, Any]]:
    """Parse all physical GPU rows and preserve unavailable metrics as null."""
    import csv
    import io

    rows: list[dict[str, Any]] = []
    for fields in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != 8:
            raise ValueError(f"expected 8 nvidia-smi columns, got {len(fields)}")
        rows.append(
            {
                "index": _optional_gpu_integer(fields[0]),
                "name": fields[1].strip() or None,
                "driver_version": fields[2].strip() or None,
                "uuid": fields[3].strip() or None,
                "memory_used_mib": _optional_gpu_integer(fields[4]),
                "memory_total_mib": _optional_gpu_integer(fields[5]),
                "utilization_gpu_percent": _optional_gpu_integer(fields[6]),
                "utilization_memory_percent": _optional_gpu_integer(fields[7]),
            }
        )
    return rows


def nvidia_smi_gpu_sample(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Collect a timestamped bounded physical-device inventory/utilization sample."""
    import subprocess

    empty = _process_output_receipt(None)
    sample: dict[str, Any] = {
        "sampled_at_unix": time.time(),
        "query_fields": GPU_QUERY_FIELDS,
        "status": "unavailable",
        "rows": [],
        "return_code": None,
        "raw_output": empty["text"],
        "raw_output_byte_length": empty["byte_length"],
        "raw_output_sha256": empty["sha256"],
        "raw_output_truncated": empty["truncated"],
        "raw_stderr": empty["text"],
        "raw_stderr_byte_length": empty["byte_length"],
        "raw_stderr_sha256": empty["sha256"],
        "raw_stderr_truncated": empty["truncated"],
    }
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        stdout = _process_output_receipt(getattr(error, "stdout", None))
        stderr = _process_output_receipt(getattr(error, "stderr", None))
        sample.update(
            {
                "status": "query_error",
                "diagnostic": f"{type(error).__name__}: {error}",
                "raw_output": stdout["text"],
                "raw_output_byte_length": stdout["byte_length"],
                "raw_output_sha256": stdout["sha256"],
                "raw_output_truncated": stdout["truncated"],
                "raw_stderr": stderr["text"],
                "raw_stderr_byte_length": stderr["byte_length"],
                "raw_stderr_sha256": stderr["sha256"],
                "raw_stderr_truncated": stderr["truncated"],
                "return_code": getattr(error, "returncode", None),
            }
        )
        return sample

    stdout = _process_output_receipt(completed.stdout)
    stderr = _process_output_receipt(completed.stderr)
    sample.update(
        {
            "return_code": completed.returncode,
            "raw_output": stdout["text"],
            "raw_output_byte_length": stdout["byte_length"],
            "raw_output_sha256": stdout["sha256"],
            "raw_output_truncated": stdout["truncated"],
            "raw_stderr": stderr["text"],
            "raw_stderr_byte_length": stderr["byte_length"],
            "raw_stderr_sha256": stderr["sha256"],
            "raw_stderr_truncated": stderr["truncated"],
        }
    )
    if completed.returncode != 0:
        sample.update(
            {
                "status": "query_error",
                "diagnostic": f"nvidia-smi exited with return code {completed.returncode}",
            }
        )
        return sample
    try:
        rows = parse_nvidia_smi_gpu_rows(completed.stdout)
    except ValueError as error:
        sample.update({"status": "malformed_output", "diagnostic": str(error)})
        return sample
    sample["rows"] = rows
    sample["status"] = "ok" if rows else "unavailable"
    if not rows:
        sample["diagnostic"] = "nvidia-smi returned no physical GPU rows"
    return sample


def runtime_identity(torch_module: Any) -> dict[str, Any]:
    """Capture actual CUDA/software identity and the full physical GPU inventory."""
    properties = torch_module.cuda.get_device_properties(0)
    cuda_free_bytes, cuda_memory_info_total_bytes = torch_module.cuda.mem_get_info(0)
    inventory = nvidia_smi_gpu_sample()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    token = visible_devices.split(",", maxsplit=1)[0].strip() if visible_devices else "0"
    rows = inventory.get("rows", [])
    if token.isdigit():
        selected_row = next((row for row in rows if row.get("index") == int(token)), None)
    else:
        selected_row = next((row for row in rows if row.get("uuid") == token), None)
    return {
        "visible_gpu_count": int(torch_module.cuda.device_count()),
        "cuda_visible_devices": visible_devices,
        "selected_gpu_name": str(torch_module.cuda.get_device_name(0)),
        "selected_gpu_total_bytes": int(properties.total_memory),
        "selected_cuda_free_bytes": int(cuda_free_bytes),
        "selected_cuda_memory_info_total_bytes": int(cuda_memory_info_total_bytes),
        "selected_physical_gpu_uuid": selected_row.get("uuid") if selected_row else None,
        "selected_physical_gpu_memory_used_mib": (
            selected_row.get("memory_used_mib") if selected_row else None
        ),
        "selected_physical_gpu_memory_total_mib": (
            selected_row.get("memory_total_mib") if selected_row else None
        ),
        "selected_gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch_module.__version__),
        "cuda_runtime_version": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "physical_gpu_inventory_and_utilization": inventory,
    }


def require_physical_gpu_inventory(identity: dict[str, Any]) -> None:
    """Require the exact selected UUID and coherent CUDA/NVML memory domains."""
    inventory = identity.get("physical_gpu_inventory_and_utilization", {})
    rows = inventory.get("rows", [])
    if inventory.get("status") != "ok" or not rows:
        raise ValueError("physical GPU inventory query is unavailable or malformed")
    if any(row.get("uuid") is None or row.get("memory_total_mib") is None for row in rows):
        raise ValueError("physical GPU UUID/memory inventory is incomplete")
    selected_uuid = identity.get("selected_physical_gpu_uuid")
    selected_total_mib = identity.get("selected_physical_gpu_memory_total_mib")
    selected_used_mib = identity.get("selected_physical_gpu_memory_used_mib")
    selected_total_bytes = identity.get("selected_gpu_total_bytes")
    if selected_total_mib is None or selected_used_mib is None:
        raise ValueError("selected CUDA device UUID/memory could not be confirmed")
    try:
        memory_domain_snapshot(
            cuda_property_total_bytes=selected_total_bytes,
            cuda_memory_info_total_bytes=identity.get("selected_cuda_memory_info_total_bytes"),
            cuda_free_bytes=identity.get("selected_cuda_free_bytes"),
            nvml_physical_total_mib=selected_total_mib,
            nvml_memory_used_mib=selected_used_mib,
            observed_uuid=selected_uuid,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
