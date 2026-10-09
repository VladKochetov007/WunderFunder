from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_initialize_cuda_context = importlib.import_module("h230_profile_then_fit")._initialize_cuda_context


EXPECTED = {
    "device_name_substring": "3090",
    "device_capability": [8, 6],
    "cuda_bf16_supported": True,
}


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        devices: int = 1,
        name: str = "NVIDIA GeForce RTX 3090",
        capability: tuple[int, int] = (8, 6),
        bf16: bool = True,
    ) -> None:
        self.available = available
        self.devices = devices
        self.name = name
        self.capability = capability
        self.bf16 = bf16
        self.calls: list[str] = []
        self.initialized = False
        self.selected_device: int | None = None

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return self.available

    def device_count(self) -> int:
        self.calls.append("device_count")
        return self.devices

    def init(self) -> None:
        self.calls.append("init")
        self.initialized = True

    def set_device(self, index: int) -> None:
        self.calls.append(f"set_device:{index}")
        if not self.initialized:
            raise AssertionError("device selection happened before CUDA init")
        self.selected_device = index

    def get_device_name(self, index: int) -> str:
        self.calls.append(f"get_device_name:{index}")
        return self.name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        self.calls.append(f"get_device_capability:{index}")
        return self.capability

    def is_bf16_supported(self) -> bool:
        self.calls.append("is_bf16_supported")
        return self.bf16

    def reset_peak_memory_stats(self, index: int) -> None:
        self.calls.append(f"reset_peak_memory_stats:{index}")
        if not self.initialized or self.selected_device != index:
            raise AssertionError("peak stats reset before child CUDA init/device selection")


def test_initializes_selects_and_checks_device_before_peak_reset() -> None:
    cuda = FakeCuda()
    receipt = _initialize_cuda_context(SimpleNamespace(cuda=cuda), EXPECTED)

    assert cuda.calls == [
        "is_available",
        "device_count",
        "init",
        "set_device:0",
        "get_device_name:0",
        "get_device_capability:0",
        "is_bf16_supported",
        "reset_peak_memory_stats:0",
    ]
    assert receipt["initialized_in_child"] is True
    assert receipt["peak_memory_stats_reset_after_init"] is True


def test_fails_closed_when_cuda_is_unavailable() -> None:
    cuda = FakeCuda(available=False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _initialize_cuda_context(SimpleNamespace(cuda=cuda), EXPECTED)
    assert "reset_peak_memory_stats:0" not in cuda.calls


def test_fails_closed_when_multiple_devices_are_visible() -> None:
    cuda = FakeCuda(devices=2)
    with pytest.raises(RuntimeError, match="exactly one visible"):
        _initialize_cuda_context(SimpleNamespace(cuda=cuda), EXPECTED)
    assert "init" not in cuda.calls
    assert "reset_peak_memory_stats:0" not in cuda.calls


def test_fails_closed_on_capability_mismatch_before_reset() -> None:
    cuda = FakeCuda(capability=(7, 5))
    with pytest.raises(RuntimeError, match="capability differs"):
        _initialize_cuda_context(SimpleNamespace(cuda=cuda), EXPECTED)
    assert "init" in cuda.calls
    assert "set_device:0" in cuda.calls
    assert "reset_peak_memory_stats:0" not in cuda.calls


def test_fails_closed_on_bf16_mismatch_before_reset() -> None:
    cuda = FakeCuda(bf16=False)
    with pytest.raises(RuntimeError, match="BF16 support differs"):
        _initialize_cuda_context(SimpleNamespace(cuda=cuda), EXPECTED)
    assert "reset_peak_memory_stats:0" not in cuda.calls
