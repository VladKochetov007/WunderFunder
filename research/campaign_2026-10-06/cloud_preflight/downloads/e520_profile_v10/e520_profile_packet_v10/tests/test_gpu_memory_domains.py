"""Synthetic checks for CUDA-to-NVML UUID binding and strict memory domains."""

from __future__ import annotations

import pytest
from gpu_memory_domains import (
    memory_domain_snapshot,
    select_nvml_device,
    validate_selected_uuid,
)

TEST_UUID = "GPU-selected-test"


def _gpu_row(index: int, uuid: str, total_mib: int = 15_360) -> dict[str, object]:
    return {
        "index": index,
        "uuid": uuid,
        "memory_total_mib": total_mib,
        "memory_used_mib": 3,
    }


def test_cuda_nvml_domains_remain_strict_and_conservative() -> None:
    snapshot = memory_domain_snapshot(
        cuda_property_total_bytes=15_636_037_632,
        cuda_memory_info_total_bytes=15_636_037_632,
        cuda_free_bytes=15_526_068_224,
        nvml_physical_total_mib=15_360,
        nvml_memory_used_mib=3,
        observed_uuid=TEST_UUID,
        expected_uuid=TEST_UUID,
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
            observed_uuid=TEST_UUID,
            expected_uuid=TEST_UUID,
        )


@pytest.mark.parametrize(
    ("observed", "expected"),
    [(None, TEST_UUID), ("GPU-other", TEST_UUID), (TEST_UUID, None)],
)
def test_missing_or_changed_worker_uuid_fails(observed: str | None, expected: str | None) -> None:
    with pytest.raises(ValueError, match="UUID changed or is unavailable"):
        validate_selected_uuid(observed, expected)


def test_cuda_visibility_selects_matching_index_not_first_inventory_row() -> None:
    rows = [_gpu_row(1, TEST_UUID), _gpu_row(0, "GPU-first-row")]
    selected = select_nvml_device(rows, "0")
    assert selected["index"] == 0
    assert selected["uuid"] == "GPU-first-row"

    selected_second = select_nvml_device(rows, "1")
    assert selected_second["index"] == 1
    assert selected_second["uuid"] == TEST_UUID


def test_cuda_visibility_uuid_token_resolves_exact_inventory_uuid() -> None:
    rows = [_gpu_row(0, "GPU-first-row"), _gpu_row(1, TEST_UUID)]
    assert select_nvml_device(rows, TEST_UUID)["uuid"] == TEST_UUID


@pytest.mark.parametrize("visible", [None, "", "0,1", "2", "GPU-missing"])
def test_ambiguous_or_missing_cuda_visibility_binding_fails(
    visible: str | None,
) -> None:
    rows = [_gpu_row(0, "GPU-first-row"), _gpu_row(1, TEST_UUID)]
    with pytest.raises(ValueError):
        select_nvml_device(rows, visible)


def test_nvml_free_domain_can_tighten_the_gpu_upper_bound() -> None:
    snapshot = memory_domain_snapshot(
        cuda_property_total_bytes=15_636_037_632,
        cuda_memory_info_total_bytes=15_636_037_632,
        cuda_free_bytes=15_000_000_000,
        nvml_physical_total_mib=15_360,
        nvml_memory_used_mib=10_000,
        observed_uuid=TEST_UUID,
        expected_uuid=TEST_UUID,
    )
    assert snapshot["conservative_free_bytes"] == 5_620_367_360
    assert snapshot["device_used_upper_bound_bytes"] == 10_485_760_000
