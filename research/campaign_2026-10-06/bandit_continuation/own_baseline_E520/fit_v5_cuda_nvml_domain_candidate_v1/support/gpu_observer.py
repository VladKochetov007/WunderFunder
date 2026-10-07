"""Fail-closed CUDA and nvidia-smi observations for the E520 fit worker."""

from __future__ import annotations

import hashlib
from typing import Any

from support.gpu_inventory import nvidia_smi_gpu_sample
from support.gpu_memory_domains import (
    EXPECTED_PHYSICAL_GPU_UUID,
    memory_domain_snapshot,
    validate_selected_uuid,
)

GPU_QUERY_RAW_MAX_BYTES = 65_536
GPU_DOMAIN_FIELDS = (
    "gpu_process_bytes",
    "gpu_process_query_status",
    "gpu_process_query_raw",
    "gpu_process_query_raw_byte_length",
    "gpu_process_query_raw_sha256",
    "gpu_process_query_raw_truncated",
    "gpu_process_query_stderr",
    "gpu_process_query_stderr_byte_length",
    "gpu_process_query_stderr_sha256",
    "gpu_process_query_stderr_truncated",
    "gpu_process_query_return_code",
    "gpu_process_query_diagnostic",
    "gpu_total_bytes",
    "gpu_memory_info_total_bytes",
    "gpu_memory_totals_match",
    "gpu_cuda_usable_total_bytes",
    "gpu_cuda_free_bytes",
    "gpu_nvml_physical_total_bytes",
    "gpu_nvml_memory_used_bytes",
    "gpu_nvml_free_bytes",
    "gpu_selected_physical_gpu_uuid",
    "gpu_physical_inventory_status",
    "gpu_physical_inventory_raw_sha256",
    "gpu_device_used_upper_bound_bytes",
    "gpu_allocation_measurement_basis",
    "gpu_resource_query_status",
    "gpu_resource_query_diagnostic",
    "gpu_free_bytes",
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


def parse_nvidia_process_bytes(pid: int, output: str) -> int | None:
    """Return matched process bytes; an empty table remains unavailable, never zero."""
    import csv
    import io

    matches: list[int] = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if (
            len(row) >= 2
            and row[0].strip().isdigit()
            and int(row[0]) == pid
            and row[1].strip().isdigit()
        ):
            matches.append(int(row[1]) * 1024 * 1024)
    return sum(matches) if matches else None


def nvidia_process_measurement(pid: int) -> dict[str, Any]:
    """Query this PID while preserving bounded stdout/stderr on success or failure."""
    import subprocess

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        stdout = _process_output_receipt(getattr(error, "stdout", None))
        stderr = _process_output_receipt(getattr(error, "stderr", None))
        return {
            "bytes": None,
            "status": "query_error",
            "raw": stdout["text"],
            "raw_byte_length": stdout["byte_length"],
            "raw_sha256": stdout["sha256"],
            "raw_truncated": stdout["truncated"],
            "stderr": stderr["text"],
            "stderr_byte_length": stderr["byte_length"],
            "stderr_sha256": stderr["sha256"],
            "stderr_truncated": stderr["truncated"],
            "return_code": getattr(error, "returncode", None),
            "diagnostic": f"{type(error).__name__}: {error}",
        }

    stdout = _process_output_receipt(completed.stdout)
    stderr = _process_output_receipt(completed.stderr)
    if completed.returncode != 0:
        status = "query_error"
        used = None
        diagnostic = f"nvidia-smi exited with return code {completed.returncode}"
    else:
        used = parse_nvidia_process_bytes(pid, stdout["text"])
        status = "matched_pid" if used is not None else "pid_not_reported"
        diagnostic = None
    return {
        "bytes": used,
        "status": status,
        "raw": stdout["text"],
        "raw_byte_length": stdout["byte_length"],
        "raw_sha256": stdout["sha256"],
        "raw_truncated": stdout["truncated"],
        "stderr": stderr["text"],
        "stderr_byte_length": stderr["byte_length"],
        "stderr_sha256": stderr["sha256"],
        "stderr_truncated": stderr["truncated"],
        "return_code": completed.returncode,
        "diagnostic": diagnostic,
    }


def cuda_resource_snapshot(torch_module: Any | None, pid: int) -> dict[str, Any]:
    """Capture allocator and device-wide bounds without inventing PID usage."""
    result: dict[str, Any] = {
        "cuda_initialized": False,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_allocated_peak_bytes": None,
        "cuda_reserved_peak_bytes": None,
        "gpu_process_bytes": None,
        "gpu_process_query_status": "pre_cuda_idle",
        "gpu_process_query_raw": None,
        "gpu_process_query_raw_byte_length": None,
        "gpu_process_query_raw_sha256": None,
        "gpu_process_query_raw_truncated": None,
        "gpu_process_query_stderr": None,
        "gpu_process_query_stderr_byte_length": None,
        "gpu_process_query_stderr_sha256": None,
        "gpu_process_query_stderr_truncated": None,
        "gpu_process_query_return_code": None,
        "gpu_process_query_diagnostic": None,
        "gpu_total_bytes": None,
        "gpu_memory_info_total_bytes": None,
        "gpu_memory_totals_match": None,
        "gpu_cuda_usable_total_bytes": None,
        "gpu_cuda_free_bytes": None,
        "gpu_nvml_physical_total_bytes": None,
        "gpu_nvml_memory_used_bytes": None,
        "gpu_nvml_free_bytes": None,
        "gpu_selected_physical_gpu_uuid": None,
        "gpu_physical_inventory_status": "pre_cuda_idle",
        "gpu_physical_inventory_raw_sha256": None,
        "gpu_device_used_upper_bound_bytes": None,
        "gpu_allocation_measurement_basis": "pre_cuda_idle",
        "gpu_resource_query_status": "pre_cuda_idle",
        "gpu_resource_query_diagnostic": None,
        "gpu_free_bytes": None,
    }
    if torch_module is None:
        return result
    if not torch_module.cuda.is_available():
        result.update(
            {
                "gpu_process_query_status": "cuda_unavailable",
                "gpu_allocation_measurement_basis": "unavailable",
                "gpu_resource_query_status": "query_error",
                "gpu_resource_query_diagnostic": "Torch reports CUDA unavailable",
            }
        )
        return result

    process = nvidia_process_measurement(pid)
    result.update(
        {
            "cuda_initialized": bool(torch_module.cuda.is_initialized()),
            "gpu_process_bytes": process["bytes"],
            "gpu_process_query_status": process["status"],
            "gpu_process_query_raw": process["raw"],
            "gpu_process_query_raw_byte_length": process["raw_byte_length"],
            "gpu_process_query_raw_sha256": process["raw_sha256"],
            "gpu_process_query_raw_truncated": process["raw_truncated"],
            "gpu_process_query_stderr": process["stderr"],
            "gpu_process_query_stderr_byte_length": process["stderr_byte_length"],
            "gpu_process_query_stderr_sha256": process["stderr_sha256"],
            "gpu_process_query_stderr_truncated": process["stderr_truncated"],
            "gpu_process_query_return_code": process["return_code"],
            "gpu_process_query_diagnostic": process["diagnostic"],
            "gpu_allocation_measurement_basis": "process_table_pid"
            if process["bytes"] is not None
            else "device_total_minus_free_upper_bound",
        }
    )
    try:
        properties = torch_module.cuda.get_device_properties(0)
        physical_inventory = nvidia_smi_gpu_sample(timeout_seconds=2)
        result["gpu_physical_inventory_status"] = physical_inventory.get("status")
        result["gpu_physical_inventory_raw_sha256"] = physical_inventory.get("raw_output_sha256")
        selected_row = next(
            (
                row
                for row in physical_inventory.get("rows", [])
                if row.get("uuid") == EXPECTED_PHYSICAL_GPU_UUID
            ),
            None,
        )
        free_bytes, memory_info_total = torch_module.cuda.mem_get_info(0)
        total_bytes = int(properties.total_memory)
        free_bytes = int(free_bytes)
        memory_info_total = int(memory_info_total)
        totals_match = total_bytes == memory_info_total
        domain = memory_domain_snapshot(
            cuda_property_total_bytes=total_bytes,
            cuda_memory_info_total_bytes=memory_info_total,
            cuda_free_bytes=free_bytes,
            nvml_physical_total_mib=(
                selected_row.get("memory_total_mib") if selected_row else None
            ),
            nvml_memory_used_mib=(selected_row.get("memory_used_mib") if selected_row else None),
            observed_uuid=selected_row.get("uuid") if selected_row else None,
        )
        result.update(
            {
                "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
                "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
                "cuda_allocated_peak_bytes": int(torch_module.cuda.max_memory_allocated(0)),
                "cuda_reserved_peak_bytes": int(torch_module.cuda.max_memory_reserved(0)),
                "gpu_total_bytes": total_bytes,
                "gpu_memory_info_total_bytes": memory_info_total,
                "gpu_memory_totals_match": totals_match,
                "gpu_cuda_usable_total_bytes": domain["cuda_usable_total_bytes"],
                "gpu_cuda_free_bytes": domain["cuda_free_bytes"],
                "gpu_nvml_physical_total_bytes": domain["nvml_physical_total_bytes"],
                "gpu_nvml_memory_used_bytes": domain["nvml_memory_used_bytes"],
                "gpu_nvml_free_bytes": domain["nvml_free_bytes"],
                "gpu_selected_physical_gpu_uuid": domain["selected_physical_gpu_uuid"],
                "gpu_device_used_upper_bound_bytes": domain["device_used_upper_bound_bytes"],
                "gpu_allocation_measurement_basis": "process_table_pid"
                if process["bytes"] is not None
                else "device_total_minus_free_upper_bound"
                if totals_match
                else "unavailable",
                "gpu_resource_query_status": "ok",
                "gpu_resource_query_diagnostic": None,
                "gpu_free_bytes": domain["conservative_free_bytes"],
            }
        )
    except ValueError as error:
        result.update(
            {
                "gpu_resource_query_status": "invalid_memory_domains",
                "gpu_resource_query_diagnostic": f"{type(error).__name__}: {error}",
            }
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        result.update(
            {
                "gpu_resource_query_status": "query_error",
                "gpu_allocation_measurement_basis": "unavailable",
                "gpu_resource_query_diagnostic": f"{type(error).__name__}: {error}",
            }
        )
    return result


def refresh_cuda_batch_boundary(
    snapshot: dict[str, Any], prior_snapshot: dict[str, Any], torch_module: Any
) -> dict[str, Any]:
    """Refresh fast CUDA counters while reusing the recent validated NVML sample."""
    for field in GPU_DOMAIN_FIELDS:
        snapshot[field] = prior_snapshot.get(field)
    properties = torch_module.cuda.get_device_properties(0)
    free_bytes, memory_info_total = torch_module.cuda.mem_get_info(0)
    total_bytes = int(properties.total_memory)
    free_bytes = int(free_bytes)
    memory_info_total = int(memory_info_total)
    physical_total = snapshot.get("gpu_nvml_physical_total_bytes")
    physical_free = snapshot.get("gpu_nvml_free_bytes")
    physical_used = snapshot.get("gpu_nvml_memory_used_bytes")
    selected_uuid = snapshot.get("gpu_selected_physical_gpu_uuid")
    try:
        domain = memory_domain_snapshot(
            cuda_property_total_bytes=total_bytes,
            cuda_memory_info_total_bytes=memory_info_total,
            cuda_free_bytes=free_bytes,
            nvml_physical_total_mib=(
                physical_total // 1024**2 if isinstance(physical_total, int) else None
            ),
            nvml_memory_used_mib=(
                physical_used // 1024**2 if isinstance(physical_used, int) else None
            ),
            observed_uuid=selected_uuid,
        )
    except ValueError as error:
        snapshot.update(
            {
                "gpu_resource_query_status": "invalid_memory_domains",
                "gpu_resource_query_diagnostic": f"{type(error).__name__}: {error}",
            }
        )
        return snapshot
    if not isinstance(physical_free, int) or domain["nvml_free_bytes"] != physical_free:
        snapshot.update(
            {
                "gpu_resource_query_status": "invalid_memory_domains",
                "gpu_resource_query_diagnostic": "cached NVML free-memory sample is inconsistent",
            }
        )
        return snapshot
    totals_match = total_bytes == memory_info_total
    snapshot.update(
        {
            "gpu_total_bytes": total_bytes,
            "gpu_memory_info_total_bytes": memory_info_total,
            "gpu_memory_totals_match": totals_match,
            "gpu_cuda_usable_total_bytes": domain["cuda_usable_total_bytes"],
            "gpu_cuda_free_bytes": domain["cuda_free_bytes"],
            "gpu_device_used_upper_bound_bytes": domain["device_used_upper_bound_bytes"],
            "gpu_free_bytes": min(free_bytes, physical_free),
            "gpu_resource_query_status": "ok",
            "gpu_resource_query_diagnostic": None,
            "gpu_allocation_measurement_basis": "process_table_pid"
            if snapshot.get("gpu_process_bytes") is not None
            else "physical_total_minus_minimum_free_upper_bound",
        }
    )
    return snapshot


def enforce_active_gpu_observation(
    snapshot: dict[str, Any], process_cap_bytes: int, min_free_bytes: int
) -> None:
    """Validate active GPU measurement, using total-minus-free only as an upper bound."""
    process_bytes = snapshot.get("gpu_process_bytes")
    process_status = snapshot.get("gpu_process_query_status")
    if process_status == "pre_cuda_idle" or snapshot.get("cuda_initialized") is not True:
        raise ValueError("active CUDA process measurement is not initialized")
    if snapshot.get("gpu_resource_query_status") != "ok":
        raise ValueError("active CUDA device-memory measurement is unavailable or inconsistent")
    validate_selected_uuid(snapshot.get("gpu_selected_physical_gpu_uuid"))
    if process_bytes is not None:
        if process_status != "matched_pid" or process_bytes > process_cap_bytes:
            raise ValueError("matched GPU process allocation exceeded its hard cap")
    else:
        if process_status not in {"pid_not_reported", "query_error"}:
            raise ValueError("GPU process query status is not eligible for device-wide fallback")
        if snapshot.get("gpu_memory_totals_match") is not True:
            raise ValueError("device-wide GPU memory total/free measurement is unavailable")
        upper_bound = snapshot.get("gpu_device_used_upper_bound_bytes")
        if not isinstance(upper_bound, int) or upper_bound < 0:
            raise ValueError("device-wide GPU allocation upper bound is unavailable")
        if upper_bound > process_cap_bytes:
            raise ValueError("device-wide GPU usage upper bound exceeded its hard cap")
    free_bytes = snapshot.get("gpu_free_bytes")
    if not isinstance(free_bytes, int) or free_bytes < min_free_bytes:
        raise ValueError("selected GPU free memory fell below its hard floor")
