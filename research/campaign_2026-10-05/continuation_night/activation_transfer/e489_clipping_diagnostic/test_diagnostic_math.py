from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAGNOSTIC_DIR))

from diagnostic_math import (  # noqa: E402
    clipping_record,
    compare_delta_groups,
    compare_tensors,
    raw_global_norm,
)


def test_registered_global_gradient_norm_is_norm_of_parameter_norms() -> None:
    gradients = {
        "a": torch.tensor([3.0, 4.0]),
        "b": torch.tensor([12.0]),
    }
    assert float(raw_global_norm(gradients)) == pytest.approx(13.0)


def test_reported_clip_factor_matches_torch_clip_grad_norm() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = torch.linalg.vector_norm(parameter.grad)
    record = clipping_record(norm, max_norm=1.0)
    torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
    observed_factor = float(parameter.grad[0] / 3.0)
    assert record["preclip_global_norm"] == pytest.approx(5.0)
    assert record["clipping_active"] is True
    assert record["clip_factor"] == pytest.approx(observed_factor)


def test_clipping_gate_is_false_below_and_at_threshold() -> None:
    assert clipping_record(torch.tensor(0.5))["clipping_active"] is False
    assert clipping_record(torch.tensor(1.0))["clipping_active"] is False
    assert clipping_record(torch.tensor(1.01))["clipping_active"] is True


def test_finite_parity_failures_are_recorded_instead_of_raising() -> None:
    result = compare_tensors(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.1]), atol=1e-5, rtol=1e-5
    )
    assert result["allclose"] is False
    assert result["max_abs_error"] == pytest.approx(0.1, abs=1e-6)


def test_optimizer_delta_comparisons_preserve_original_gate_limits() -> None:
    reference = {"matrix:g": [torch.tensor([0.1, -0.2])], "adamw": [torch.tensor([0.3])]}
    candidate = {
        "matrix:g": [torch.tensor([0.1, -0.2])],
        "adamw": [torch.tensor([0.300001])],
    }
    result = compare_delta_groups(reference, candidate)
    assert result["matrix:g"]["original_delta_gate_passed"] is True
    assert result["adamw"]["relative_limit"] == 0.02
    assert result["adamw"]["max_abs_limit"] == 2e-5
