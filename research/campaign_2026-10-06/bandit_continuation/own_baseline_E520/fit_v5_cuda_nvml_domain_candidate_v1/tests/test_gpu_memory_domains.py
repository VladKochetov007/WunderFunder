"""Focused CUDA/NVML memory-domain checks for fit v5."""

from __future__ import annotations

import pytest
from support.gpu_memory_domains import (
    EXPECTED_PHYSICAL_GPU_UUID,
    memory_domain_snapshot,
    validate_selected_uuid,
)


def test_v8_observed_memory_totals_bind_as_distinct_domains() -> None:
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
    assert snapshot["device_used_upper_bound_bytes"] == 580_059_136


def test_cuda_total_above_physical_total_is_rejected() -> None:
    with pytest.raises(ValueError, match="ordering is invalid"):
        memory_domain_snapshot(
            cuda_property_total_bytes=16_200_000_000,
            cuda_memory_info_total_bytes=16_200_000_000,
            cuda_free_bytes=15_000_000_000,
            nvml_physical_total_mib=15_360,
            nvml_memory_used_mib=1,
            observed_uuid=EXPECTED_PHYSICAL_GPU_UUID,
        )


@pytest.mark.parametrize("observed", [None, "GPU-other"])
def test_missing_or_mismatched_uuid_is_rejected(observed: str | None) -> None:
    with pytest.raises(ValueError, match="UUID is missing or mismatched"):
        validate_selected_uuid(observed)
