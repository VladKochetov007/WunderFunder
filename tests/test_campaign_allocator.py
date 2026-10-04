from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from wnn.resources import GPU_FRACTION, allocator_fraction, configure_torch


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "NaN", "5000000001", "４"])
def test_byte_limit_rejects_invalid_or_oversized_values(value: str) -> None:
    with pytest.raises(ValueError):
        allocator_fraction(value, 5_000_000_000)


def test_decimal_bytes_are_not_gibibytes() -> None:
    assert allocator_fraction("4200000000", 6 * 1024**3) == 4_200_000_000 / (6 * 1024**3)


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_bytes_override_fraction_once_without_cuda_allocation(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    calls: list[float] = []
    fake = SimpleNamespace(
        set_num_threads=lambda _: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(total_memory=6 * 1024**3),
            set_per_process_memory_fraction=calls.append,
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("WNN_GPU_MEMORY_BYTES", raising=False)
    monkeypatch.setenv("WNN_VRAM_GB", "999")
    if explicit:
        monkeypatch.setenv("WNN_GPU_MEMORY_BYTES", "4200000000")
    configure_torch(duty=1.0)
    assert calls == [4_200_000_000 / (6 * 1024**3) if explicit else GPU_FRACTION]
