"""Tiny source-only tests for the fixed R12 objective comparison."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from idea_screens_r12.probe_execution import _require_r12_service
from idea_screens_r12.run_probe import RUNTIME_IMPORT_PATHS, _cli_module
from idea_screens_r12.training_pair import (
    _fit_loss,
    initial_gradient_norms,
    initialize_pair,
    scale_gamma,
)
from wnn.gmetric import global_wp


class _FixedMean(nn.Module):
    """Expose a trainable prediction tensor for direct loss-gradient checks."""

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = nn.Parameter(values.clone())

    def forward_mean(self, _features: torch.Tensor) -> torch.Tensor:
        return self.values


def test_canonical_objectives_share_clipped_prediction_and_match_pooled_scorer() -> None:
    values = torch.tensor([[-3.0, -0.4], [-0.3, 0.1], [0.2, 0.7], [3.0, 1.8]])
    target = torch.tensor([[-1.0, -0.5], [-0.2, 0.2], [0.4, 0.9], [1.2, 1.7]])
    model = _FixedMean(values)
    features = torch.zeros((len(values), 224))
    valid = torch.ones(len(values), dtype=torch.bool)
    clipped = values.clamp(-2.0, 2.0)
    weights = target.clamp(-2.0, 2.0).abs()
    expected_mse = (weights * (clipped - target.clamp(-2.0, 2.0)).square()).sum() / weights.sum()
    wmse = _fit_loss(model, features, target, valid, "wmse")
    corr_loss = _fit_loss(model, features, target, valid, "pooled_corr")
    score = global_wp(target.numpy(), clipped.numpy(), np.ones(len(target), dtype=bool))
    assert wmse.item() == pytest.approx(expected_mse.item(), rel=1e-6)
    assert corr_loss.item() == pytest.approx(1.0 - float(score["wp"]), abs=2e-5)


def test_zero_weight_and_zero_prediction_variance_have_finite_gradients() -> None:
    features = torch.zeros((5, 224))
    valid = torch.ones(5, dtype=torch.bool)
    for targets, values in (
        (torch.zeros((5, 2)), torch.full((5, 2), 0.3)),
        (
            torch.tensor([[1.0, -0.5], [0.2, 1.1], [-0.3, 0.4], [1.4, -0.8], [0.6, 0.9]]),
            torch.full((5, 2), 0.3),
        ),
    ):
        model = _FixedMean(values)
        for objective in ("wmse", "pooled_corr"):
            loss = _fit_loss(model, features, targets, valid, objective)
            gradient = torch.autograd.grad(loss, model.values)[0]
            assert torch.isfinite(loss)
            assert torch.isfinite(gradient).all()


def test_common_clamp_has_zero_outside_gradient_and_finite_interior_gradient() -> None:
    values = torch.tensor([[-2.5, 0.2], [2.5, -0.7]], requires_grad=False)
    model = _FixedMean(values)
    features = torch.zeros((2, 224))
    targets = torch.tensor([[0.1, -0.1], [-0.2, 0.3]])
    valid = torch.ones(2, dtype=torch.bool)
    for objective in ("wmse", "pooled_corr"):
        gradient = torch.autograd.grad(
            _fit_loss(model, features, targets, valid, objective), model.values
        )[0]
        assert gradient[0, 0].item() == 0.0
        assert gradient[1, 0].item() == 0.0
        assert torch.isfinite(gradient).all()


def test_both_objective_autograd_gradients_match_small_finite_difference() -> None:
    values = torch.tensor(
        [[-0.8, 0.4], [-0.2, -0.6], [0.1, 0.3], [0.5, 0.9], [0.7, -0.1], [1.0, 0.6]]
    )
    targets = torch.tensor(
        [[-0.4, 0.7], [-0.1, -0.2], [0.2, 0.6], [0.8, 0.3], [0.3, -0.4], [0.9, 1.0]]
    )
    features = torch.zeros((len(values), 224))
    valid = torch.ones(len(values), dtype=torch.bool)
    epsilon = 1e-3
    for objective in ("wmse", "pooled_corr"):
        model = _FixedMean(values)
        analytical = torch.autograd.grad(
            _fit_loss(model, features, targets, valid, objective), model.values
        )[0][2, 0].item()
        with torch.no_grad():
            model.values[2, 0] += epsilon
            upper = _fit_loss(model, features, targets, valid, objective).item()
            model.values[2, 0] -= 2 * epsilon
            lower = _fit_loss(model, features, targets, valid, objective).item()
        numeric = (upper - lower) / (2 * epsilon)
        assert analytical == pytest.approx(numeric, rel=2e-2, abs=2e-3)


def test_seeded_pair_and_gamma_prepass_preserve_state_and_unused_head() -> None:
    wmse, pooled_corr, fingerprints = initialize_pair()
    assert len(set(fingerprints.values())) == 1
    rng = np.random.default_rng(901)
    features = rng.normal(size=(24, 224)).astype(np.float32)
    targets = rng.normal(size=(24, 2)).astype(np.float32)
    before = {name: value.detach().clone() for name, value in wmse.state_dict().items()}
    norms = initial_gradient_norms(wmse, features, targets)
    gamma = scale_gamma(norms)
    assert gamma > 0.0
    assert all(parameter.grad is None for parameter in wmse.parameters())
    assert all(torch.equal(before[name], value) for name, value in wmse.state_dict().items())
    assert all(torch.count_nonzero(value) == 0 for value in wmse.logvar_head.state_dict().values())
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            wmse.state_dict().values(), pooled_corr.state_dict().values(), strict=True
        )
    )
    with pytest.raises(FloatingPointError):
        scale_gamma({"wmse": 1.0, "pooled_corr": 1e-8})


def test_exact_r12_cgroup_is_required_before_memory_guard() -> None:
    good = "0::/wnn-campaign.slice/computer-vision-homework-eda19-r12-v1.service"
    _require_r12_service(good)
    for bad in (
        "0::/",
        "0::/user.slice/worker.service",
        "0::/wnn-campaign.slice/computer-vision-homework-eda19-r11-v1.service",
    ):
        with pytest.raises(RuntimeError):
            _require_r12_service(bad)


def test_python_m_entrypoint_identity_guard_runs_before_help() -> None:
    package_parent = Path(__file__).resolve().parent.parent
    source = Path(__file__).resolve().parents[3] / "src"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(package_parent), str(source)))}
    result = subprocess.run(
        [sys.executable, "-m", "idea_screens_r12.run_probe", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-preparation-manifest-sha256" in result.stdout
    direct = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_probe.py")), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert direct.returncode != 0
    assert "must be invoked as python -m" in direct.stderr
    assert len(RUNTIME_IMPORT_PATHS) == 26
    assert RUNTIME_IMPORT_PATHS["wnn.nn"] == "canonical:wnn/nn.py"


def test_named_cli_identity_does_not_require_duplicate_import() -> None:
    module = _cli_module(sys.modules)
    assert Path(module.__file__).resolve() == Path(__file__).resolve().with_name("run_probe.py")
