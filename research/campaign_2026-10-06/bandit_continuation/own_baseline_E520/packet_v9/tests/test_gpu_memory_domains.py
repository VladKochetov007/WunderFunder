"""Synthetic checks for strict CUDA/NVML memory-domain accounting."""

from __future__ import annotations

import pytest
from gpu_memory_domains import (
    EXPECTED_PHYSICAL_GPU_UUID,
    memory_domain_snapshot,
    validate_selected_uuid,
)


def test_v8_observed_cuda_nvml_domains_are_valid_and_conservative() -> None:
    snapshot = memory_domain_snapshot(
        cuda_property_total_bytes=15_636_037_632,
        cuda_memory_info_total_bytes=15_636_037_632,
        cuda_free_bytes=15_526_068_224,
        nvml_physical_total_mib=15_360,
        nvml_memory_used_mib=3,
        observed_uuid=EXPECTED_PHYSICAL_GPU_UUID,
    )
    assert snapshot["cuda_usable_total_bytes"] == 15_636_037_632
    assert snapshot["nvml_physical_total_bytes"] == 16_106_127_360
    assert snapshot["conservative_free_bytes"] == 15_526_068_224
    assert snapshot["device_used_upper_bound_bytes"] == 580_059_136


def test_cuda_usable_total_above_physical_total_fails_closed() -> None:
    with pytest.raises(ValueError, match="ordering is invalid"):
        memory_domain_snapshot(
            cuda_property_total_bytes=16_200_000_000,
            cuda_memory_info_total_bytes=16_200_000_000,
            cuda_free_bytes=15_000_000_000,
            nvml_physical_total_mib=15_360,
            nvml_memory_used_mib=1,
            observed_uuid=EXPECTED_PHYSICAL_GPU_UUID,
        )


@pytest.mark.parametrize("observed", [None, "GPU-wrong"])
def test_missing_or_mismatched_physical_uuid_fails(observed: str | None) -> None:
    with pytest.raises(ValueError, match="UUID is missing or mismatched"):
        validate_selected_uuid(observed)


def test_nvml_free_domain_can_tighten_the_gpu_upper_bound() -> None:
    snapshot = memory_domain_snapshot(
        cuda_property_total_bytes=15_636_037_632,
        cuda_memory_info_total_bytes=15_636_037_632,
        cuda_free_bytes=15_000_000_000,
        nvml_physical_total_mib=15_360,
        nvml_memory_used_mib=10_000,
        observed_uuid=EXPECTED_PHYSICAL_GPU_UUID,
    )
    assert snapshot["conservative_free_bytes"] == 5_620_367_360
    assert snapshot["device_used_upper_bound_bytes"] == 10_485_760_000
