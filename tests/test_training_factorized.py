from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
import torch

from tests.test_trainer_resume_integration import training
from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel
from wnn.training_factorized import (
    balanced_svd,
    configure_factorized,
    factorized_forward,
    factorized_inference_state,
    parameter_budget,
)
from wnn.training_qat import configure_qat


def test_balanced_svd_known_spectrum_and_full_rank() -> None:
    matrix = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0]))
    left, right, report = balanced_svd(matrix, 2)
    torch.testing.assert_close(left @ right, torch.diag(torch.tensor([8.0, 4.0, 0.0, 0.0])))
    torch.testing.assert_close(left.T @ left, right @ right.T)
    assert report["relative_frobenius_error"] == pytest.approx((5 / 85) ** 0.5)
    left, right, _ = balanced_svd(matrix, 4)
    torch.testing.assert_close(left @ right, matrix)
    torch.testing.assert_close(left, balanced_svd(matrix, 4)[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="rank"):
        balanced_svd(matrix, 5)


def test_factor_chain_rule_matches_effective_dense_gradient() -> None:
    torch.manual_seed(70)
    model = RNNModel(3, hidden=4, layers=2)
    configure_factorized(model, 2)
    dense = RNNModel(3, hidden=4, layers=2)
    dense.load_state_dict(factorized_inference_state(model, "raw")[0])
    features, target = torch.randn(2, 9, 3), torch.randn(2, 9, 2)
    actual, _ = factorized_forward(model, features, None)
    expected, _ = dense(features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual - target).square().mean().backward()
    (expected - target).square().mean().backward()
    for name, pair in model.gru.factors.items():
        gradient = getattr(dense.gru, name).grad
        torch.testing.assert_close(
            pair.left.grad, gradient @ pair.right.detach().T, rtol=2e-6, atol=1e-9
        )
        torch.testing.assert_close(
            pair.right.grad, pair.left.detach().T @ gradient, rtol=2e-6, atol=1e-9
        )
        assert name not in model.gru._parameters
        assert name in model.gru._buffers
    for name, parameter in model.named_parameters():
        if not name.startswith("gru.factors."):
            torch.testing.assert_close(
                parameter.grad, dict(dense.named_parameters())[name].grad, rtol=2e-6, atol=1e-9
            )


def test_full_20000_rows_export_and_causality() -> None:
    torch.manual_seed(71)
    model = RNNModel(3, hidden=4, layers=2).eval()
    configure_factorized(model, 2)
    features = torch.randn(1, 20000, 3)
    exported, metadata = factorized_inference_state(model, "raw")
    dense = RNNModel(3, hidden=4, layers=2).eval()
    dense.load_state_dict(exported)
    with torch.no_grad():
        actual, state = factorized_forward(model, features, None)
        expected, dense_state = dense(features)
        future = features.clone()
        future[:, 101:] *= -2
        changed, _ = factorized_forward(model, future, None)
        first, partial = factorized_forward(model, features[:, :10000], None)
        last, final = factorized_forward(model, features[:, 10000:], partial)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, dense_state, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :101], changed[:, :101], rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.cat((first, last), 1), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state, final, rtol=0, atol=0)
    assert metadata["native_runtime_validated"] is False
    assert not any(name.startswith("gru.factors.") for name in exported)
    for name, pair in metadata["factors"].items():
        torch.testing.assert_close(exported[name], pair["left"] @ pair["right"], rtol=0, atol=0)
    copy.deepcopy(model)  # no retained non-leaf functional overrides


def test_rank_zero_is_noop_and_budget_matches_actual_parameters() -> None:
    torch.manual_seed(72)
    model = RNNModel(13, hidden=16, layers=2, n_out=7)
    reference = copy.deepcopy(model)
    configure_factorized(model, 0)
    features = torch.randn(2, 9, 13)
    torch.testing.assert_close(model(features)[0], reference(features)[0], rtol=0, atol=0)
    assert model.state_dict().keys() == reference.state_dict().keys()
    configure_factorized(model, 5)
    budget = parameter_budget(13, 7, 16, 2, 5)
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == budget["factorized_learned_parameters"]
    )
    assert (
        sum(parameter.numel() for parameter in reference.parameters())
        == budget["same_width_dense_parameters"]
    )
    matched = RNNModel(13, hidden=budget["matched_dense_hidden"], layers=2, n_out=7)
    assert (
        sum(parameter.numel() for parameter in matched.parameters())
        == budget["matched_dense_parameters"]
    )
    with pytest.raises(ValueError, match="combined"):
        configure_qat(model, "recurrent")
    configure_qat(reference, "recurrent")
    with pytest.raises(ValueError, match="combined"):
        configure_factorized(reference, 5)


def test_ema_averages_factors_and_freeze_gru_preserves_trunk() -> None:
    torch.manual_seed(73)
    model = RNNModel(3, hidden=4, layers=1)
    configure_factorized(model, 2)
    initial, _ = factorized_inference_state(model, "raw")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("head."))
    averaged = training.FastEMA(model, 0.9)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output, _ = factorized_forward(model, torch.randn(2, 8, 3), None)
        output.square().mean().backward()
        optimizer.step()
        averaged.update_parameters(model)
    exported, metadata = factorized_inference_state(averaged.module, "ema")
    for name in initial:
        if not name.startswith("head."):
            torch.testing.assert_close(exported[name], initial[name], rtol=0, atol=0)
    assert metadata["snapshot_source"] == "ema"
    assert not torch.equal(exported["head.2.weight"], initial["head.2.weight"])


def test_zero_update_control_preserves_source_and_budget(tmp_path: Path) -> None:
    exporter = importlib.import_module("export_factorized_control")
    model = RNNModel(3, hidden=4, layers=1)
    source, output = tmp_path / "source.pt", tmp_path / "control.pt"
    args = {"hidden": 4, "layers": 1, "dropout": 0, "precision": "fp16"}
    torch.save({"model": model.state_dict(), "args": args, "n_in": 3}, source)
    exporter.export_control(source, output, 2)
    checkpoint = torch.load(output, weights_only=False)
    assert checkpoint["factorization"]["training_updates"] == 0
    assert checkpoint["args"] == args
    assert checkpoint["factorization"]["parameter_budget"]["matched_dense_hidden"] > 0
    with pytest.raises(ValueError, match="unfactorized"):
        exporter.export_control(output, tmp_path / "again.pt", 2)


def test_declared_w384_rank128_shape_and_cpu_forward() -> None:
    """Exercise the proposed factor dimensions and SVD within the isolated CPU cap."""
    torch.manual_seed(74)
    n_in = 112 + N_EVENT_FEATURES
    model = RNNModel(n_in, hidden=384, layers=2).eval()
    configure_factorized(model, 128)
    exported, metadata = factorized_inference_state(model, "raw")
    assert len(metadata["factors"]) == 4
    for factors in metadata["factors"].values():
        assert factors["left"].shape == (1152, 128)
        assert factors["right"].shape == (128, 384)
    assert abs(metadata["parameter_budget"]["matched_dense_relative_difference"]) < 0.005
    dense = RNNModel(n_in, hidden=384, layers=2).eval()
    dense.load_state_dict(exported)
    inputs = torch.randn(1, 8, n_in)
    with torch.no_grad():
        actual, state = factorized_forward(model, inputs, None)
        expected, dense_state = dense(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, dense_state, rtol=0, atol=0)
