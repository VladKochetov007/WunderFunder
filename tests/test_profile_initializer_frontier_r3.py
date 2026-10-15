"""Focused regression checks for the frozen initializer-profile execution gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wnn.nn import RNNModel

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "scripts/profile_initializer_frontier_r3.py"
SPEC = importlib.util.spec_from_file_location("profile_initializer_frontier_r3", PROFILE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the initializer profile helper")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


@pytest.mark.parametrize("mode", ["constant", "first_row", "prefix"])
def test_profile_wrapper_places_new_parameters_with_meta_backbone(mode: str) -> None:
    backbone = RNNModel(211, hidden=4, layers=2).to(device="meta")
    model = PROFILE.make_initialized_model(backbone, mode, 1.0, torch.device("meta"))

    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert all(buffer.device.type == "meta" for buffer in model.buffers())


@pytest.mark.parametrize(
    ("actual", "accepted"),
    [
        (None, True),
        (5_000_000_000, True),
        (4_999_999_488, True),
        (4_999_995_905, True),
        (4_999_995_904, False),
        (5_000_000_001, False),
        (0, False),
    ],
)
def test_profile_host_limit_accepts_only_declared_cap_or_rounding(
    actual: int | None, accepted: bool
) -> None:
    if accepted:
        PROFILE.validate_declared_host_limit(actual)
    else:
        with pytest.raises(RuntimeError):
            PROFILE.validate_declared_host_limit(actual)


@pytest.mark.parametrize(
    ("requested", "current_index", "expected", "should_query_cuda"),
    [
        ("cuda", 2, "cuda:2", True),
        ("cuda:1", 2, "cuda:1", False),
        ("cpu", 2, "cpu", False),
        ("meta", 2, "meta", False),
    ],
)
def test_profile_resolves_only_implicit_cuda_device_once(
    requested: str, current_index: int, expected: str, should_query_cuda: bool
) -> None:
    calls: list[None] = []

    def current_device() -> int:
        calls.append(None)
        return current_index

    runtime = SimpleNamespace(current_device=current_device)
    torch_module = SimpleNamespace(device=torch.device, cuda=runtime)

    result = PROFILE.resolve_execution_device(torch.device(requested), torch_module)

    assert result == torch.device(expected)
    assert len(calls) == int(should_query_cuda)
