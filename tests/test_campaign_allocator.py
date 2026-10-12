from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from wnn.resources import (
    GPU_FRACTION,
    GPU_MEMORY_SAFETY_RESERVE_BYTES,
    allocator_fraction,
    available_allocator_fraction,
    configure_torch,
    gpu_memory_safety_reserve_bytes,
)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "NaN", "5000000001", "４"])
def test_byte_limit_rejects_invalid_or_oversized_values(value: str) -> None:
    with pytest.raises(ValueError):
        allocator_fraction(value, 5_000_000_000)


def test_decimal_bytes_are_not_gibibytes() -> None:
    assert allocator_fraction("4200000000", 6 * 1024**3) == 4_200_000_000 / (6 * 1024**3)


def test_available_fraction_uses_live_free_memory_and_reserve() -> None:
    total = 6 * 1024**3
    free = 5 * 1024**3
    reserve = 750_000_000

    assert available_allocator_fraction(free, total, reserve) == (free - reserve) / total


def test_available_fraction_rejects_inconsistent_memory_or_exhausted_reserve() -> None:
    with pytest.raises(ValueError, match="observations and reserve"):
        available_allocator_fraction(7, 6, 0)
    with pytest.raises(RuntimeError, match="does not exceed"):
        available_allocator_fraction(500, 1000, 500)


@pytest.mark.parametrize("value", ["-1", "1.5", "NaN", "４"])
def test_configurable_reserve_requires_nonnegative_decimal_bytes(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("WNN_GPU_MEMORY_RESERVE_BYTES", value)
    with pytest.raises(ValueError, match="WNN_GPU_MEMORY_RESERVE_BYTES"):
        gpu_memory_safety_reserve_bytes()


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_bytes_override_fraction_once_without_cuda_allocation(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    calls: list[float] = []
    total_bytes = 6 * 1024**3
    free_bytes = 5 * 1024**3
    fake = SimpleNamespace(
        set_num_threads=lambda _: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(total_memory=total_bytes),
            mem_get_info=lambda _: (free_bytes, total_bytes),
            set_per_process_memory_fraction=calls.append,
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("WNN_GPU_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("WNN_GPU_MEMORY_RESERVE_BYTES", raising=False)
    monkeypatch.setenv("WNN_VRAM_GB", "999")
    if explicit:
        monkeypatch.setenv("WNN_GPU_MEMORY_BYTES", "4200000000")
    configure_torch(duty=1.0)
    expected = (
        4_200_000_000 / total_bytes
        if explicit
        else min(GPU_FRACTION, (free_bytes - GPU_MEMORY_SAFETY_RESERVE_BYTES) / total_bytes)
    )
    assert calls == [expected]


def test_configurable_reserve_changes_live_allocator_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    total_bytes = 6 * 1024**3
    free_bytes = 5 * 1024**3
    fake = SimpleNamespace(
        set_num_threads=lambda _: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            mem_get_info=lambda _: (free_bytes, total_bytes),
            set_per_process_memory_fraction=calls.append,
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("WNN_GPU_MEMORY_BYTES", raising=False)
    monkeypatch.setenv("WNN_GPU_MEMORY_RESERVE_BYTES", "500000000")

    configure_torch(duty=1.0)

    assert calls == [min(GPU_FRACTION, (free_bytes - 500_000_000) / total_bytes)]
