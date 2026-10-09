"""Bind live CUDA visibility to the matching physical NVML GPU."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

MIB_BYTES = 1024**2


def select_nvml_device(
    rows: Sequence[Mapping[str, object]], cuda_visible_devices: str | None
) -> dict[str, object]:
    """Resolve the sole CUDA visibility token to one exact physical inventory row.

    Numeric tokens use the NVML physical index; UUID tokens require exact UUID
    equality. This avoids inferring identity from row order or a prior session.

    Args:
        rows: Complete parsed physical NVML inventory for this observation.
        cuda_visible_devices: The worker's single CUDA visibility token.

    Returns:
        A copy of the unique row selected by the visibility token.

    Raises:
        ValueError: If visibility or the matching physical row is ambiguous,
            missing, or incomplete.
    """
    if not isinstance(cuda_visible_devices, str):
        raise ValueError("CUDA_VISIBLE_DEVICES is unavailable for physical binding")
    tokens = [token.strip() for token in cuda_visible_devices.split(",")]
    if len(tokens) != 1 or not tokens[0]:
        raise ValueError("profile requires one unambiguous CUDA visibility token")
    token = tokens[0]
    if token.isdecimal():
        selected_index = int(token)
        matches = [row for row in rows if row.get("index") == selected_index]
    else:
        matches = [row for row in rows if row.get("uuid") == token]
    if len(matches) != 1:
        raise ValueError("CUDA visibility token did not resolve to one NVML physical GPU")
    selected = dict(matches[0])
    if (
        not isinstance(selected.get("uuid"), str)
        or not selected["uuid"]
        or not isinstance(selected.get("index"), int)
        or not isinstance(selected.get("memory_total_mib"), int)
        or not isinstance(selected.get("memory_used_mib"), int)
    ):
        raise ValueError("selected NVML physical GPU identity or memory is incomplete")
    return selected


def validate_selected_uuid(observed_uuid: str | None, expected_uuid: str | None) -> None:
    """Require an active sample to preserve this worker's startup UUID binding."""
    if not expected_uuid or observed_uuid != expected_uuid:
        raise ValueError("selected physical GPU UUID changed or is unavailable")


def memory_domain_snapshot(
    *,
    cuda_property_total_bytes: int,
    cuda_memory_info_total_bytes: int,
    cuda_free_bytes: int,
    nvml_physical_total_mib: int,
    nvml_memory_used_mib: int,
    observed_uuid: str | None,
    expected_uuid: str | None,
) -> dict[str, int | str]:
    """Validate both memory domains and compute conservative free/used bounds.

    Raises:
        ValueError: If UUID identity or memory-domain ordering is unavailable.
    """
    validate_selected_uuid(observed_uuid, expected_uuid)
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
        "selected_physical_gpu_uuid": observed_uuid,
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
