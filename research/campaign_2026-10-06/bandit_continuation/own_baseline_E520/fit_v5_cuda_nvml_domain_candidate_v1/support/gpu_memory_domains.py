"""Exact CUDA/NVML memory-domain validation for E520 observers."""

from __future__ import annotations

EXPECTED_PHYSICAL_GPU_UUID = "GPU-2f8814a9-d330-e294-0068-7d7cac59cd21"
MIB_BYTES = 1024**2


def validate_selected_uuid(observed_uuid: str | None) -> None:
    """Require the physical UUID observed on the reviewed V8 device."""
    if observed_uuid != EXPECTED_PHYSICAL_GPU_UUID:
        raise ValueError("selected physical GPU UUID is missing or mismatched")


def memory_domain_snapshot(
    *,
    cuda_property_total_bytes: int,
    cuda_memory_info_total_bytes: int,
    cuda_free_bytes: int,
    nvml_physical_total_mib: int,
    nvml_memory_used_mib: int,
    observed_uuid: str | None,
) -> dict[str, int | str]:
    """Validate both memory domains and compute the conservative free/used bounds.

    Raises:
        ValueError: If UUID or memory-domain ordering is unavailable or invalid.
    """
    validate_selected_uuid(observed_uuid)
    values = (
        cuda_property_total_bytes,
        cuda_memory_info_total_bytes,
        cuda_free_bytes,
        nvml_physical_total_mib,
        nvml_memory_used_mib,
    )
    if any(not isinstance(value, int) for value in values):
        raise ValueError("CUDA/NVML memory domains must be integer observations")
    physical_total_bytes = nvml_physical_total_mib * MIB_BYTES
    if (
        cuda_property_total_bytes <= 0
        or cuda_memory_info_total_bytes <= 0
        or cuda_free_bytes <= 0
        or nvml_physical_total_mib <= 0
        or nvml_memory_used_mib < 0
        or cuda_property_total_bytes != cuda_memory_info_total_bytes
        or cuda_free_bytes > cuda_memory_info_total_bytes
        or cuda_memory_info_total_bytes > physical_total_bytes
        or nvml_memory_used_mib > nvml_physical_total_mib
    ):
        raise ValueError("CUDA/NVML memory-domain ordering is invalid")
    nvml_free_bytes = physical_total_bytes - nvml_memory_used_mib * MIB_BYTES
    upper_bound = max(
        physical_total_bytes - cuda_free_bytes,
        physical_total_bytes - nvml_free_bytes,
    )
    return {
        "selected_physical_gpu_uuid": EXPECTED_PHYSICAL_GPU_UUID,
        "cuda_property_total_bytes": cuda_property_total_bytes,
        "cuda_memory_info_total_bytes": cuda_memory_info_total_bytes,
        "cuda_usable_total_bytes": cuda_memory_info_total_bytes,
        "cuda_free_bytes": cuda_free_bytes,
        "nvml_physical_total_bytes": physical_total_bytes,
        "nvml_memory_used_bytes": nvml_memory_used_mib * MIB_BYTES,
        "nvml_free_bytes": nvml_free_bytes,
        "conservative_free_bytes": min(cuda_free_bytes, nvml_free_bytes),
        "device_used_upper_bound_bytes": upper_bound,
    }
