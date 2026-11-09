"""Prepared CPU contracts for the optional causal Transformer input stem."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_config import load_config
from wnn.training.sequence_input_stem import CausalInputStem
from wnn.training.sequence_model_helpers import detach_state
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer


@pytest.fixture(autouse=True)
def _bf16x9_is_disabled_for_cpu_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep strict-FP32 CPU contracts independent of the receiver process mode."""
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "transformer",
        "hidden": 16,
        "layers": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "context_rows": 6,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
    }
    config.update(overrides)
    return config


def _assert_state_dicts_equal(left: dict[str, Tensor], right: dict[str, Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_omitted_and_zero_stem_preserve_legacy_config_state_and_rng(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.json"
    zero_path = tmp_path / "zero.json"
    legacy_path.write_text("{}", encoding="utf-8")
    zero_path.write_text(json.dumps({"model": {"input_stem_kernel": 0}}), encoding="utf-8")
    legacy_config = load_config(legacy_path)
    zero_config = load_config(zero_path)
    assert legacy_config == zero_config
    assert "input_stem_kernel" not in legacy_config["model"]

    torch.manual_seed(90210)
    legacy = build_sequence_model(_config())
    legacy_rng = torch.random.get_rng_state()
    torch.manual_seed(90210)
    explicit_zero = build_sequence_model(_config(input_stem_kernel=0))
    zero_rng = torch.random.get_rng_state()
    assert torch.equal(legacy_rng, zero_rng)
    _assert_state_dicts_equal(legacy.state_dict(), explicit_zero.state_dict())
    assert explicit_zero.input_stem is None


@pytest.mark.parametrize("kernel_size", [1, 16])
def test_zero_initialized_stem_preserves_backbone_rng_and_prediction(kernel_size: int) -> None:
    torch.manual_seed(21)
    baseline = build_sequence_model(_config())
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(21)
    stem_model = build_sequence_model(_config(input_stem_kernel=kernel_size))
    stem_rng = torch.random.get_rng_state()
    assert torch.equal(baseline_rng, stem_rng)
    assert torch.count_nonzero(stem_model.input_stem.weight) == 0
    baseline_parameters = baseline.state_dict()
    stem_parameters = stem_model.state_dict()
    assert set(stem_parameters) - set(baseline_parameters) == {"input_stem.weight"}
    for key in baseline_parameters:
        torch.testing.assert_close(baseline_parameters[key], stem_parameters[key], rtol=0, atol=0)
    restored = build_sequence_model(_config(input_stem_kernel=kernel_size))
    restored.load_state_dict(stem_parameters, strict=True)
    _assert_state_dicts_equal(stem_parameters, restored.state_dict())

    values = torch.randn(2, 11, 112)
    baseline.eval()
    stem_model.eval()
    expected = baseline(values)[0]
    actual = stem_model(values)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kernel_size", "error"),
    [
        (True, TypeError),
        (1.0, TypeError),
        (float("nan"), TypeError),
        (-1, ValueError),
        (15, ValueError),
    ],
)
def test_invalid_stem_kernel_fails_closed(kernel_size: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="input_stem_kernel"):
        build_sequence_model(_config(input_stem_kernel=kernel_size))


def test_kernel1_is_per_channel_instant_residual_gain() -> None:
    stem = CausalInputStem(1)
    values = torch.randn(2, 7, 112)
    with torch.no_grad():
        stem.weight.fill_(0.5)
    filtered, history = stem(values)
    torch.testing.assert_close(filtered, values * 1.5, rtol=0, atol=0)
    assert history is None
    with pytest.raises(TypeError, match="integer"):
        CausalInputStem(True)


def test_input_stem_is_rejected_for_gru_in_config_and_direct_factory(tmp_path: Path) -> None:
    recipe = tmp_path / "gru-stem.json"
    recipe.write_text(
        json.dumps({"model": {"architecture": "gru", "input_stem_kernel": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Transformer"):
        load_config(recipe)
    with pytest.raises(ValueError, match="Transformer"):
        build_sequence_model(_config(architecture="gru", input_stem_kernel=16))


def test_kernel16_full_uneven_chunk_and_step_outputs_match_with_raw_history() -> None:
    model = build_sequence_model(_config(input_stem_kernel=16))
    model.eval()
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.5, 0.5, 112))
        model.std.copy_(torch.linspace(0.5, 1.5, 112))
        model.input_stem.weight.copy_(torch.linspace(-0.02, 0.05, 16).repeat(112, 1).unsqueeze(1))
    values = torch.randn(2, 37, 112)
    standardized = (values - model.mean) / model.std
    full_prediction, _, full_state = model(values)
    assert isinstance(full_state, TransformerState)
    torch.testing.assert_close(full_state.input_history, standardized[:, -15:])

    starts = (0, 2, 3, 11, 14, 15, 18, 31)
    ends = (*starts[1:], values.shape[1])
    carried: TransformerState | None = None
    chunk_predictions: list[Tensor] = []
    for start, end in zip(starts, ends, strict=True):
        previous_history = None if carried is None else carried.input_history
        snapshot = None if previous_history is None else previous_history.clone()
        prediction, _, carried = model(values[:, start:end], carried)
        chunk_predictions.append(prediction)
        expected_history = standardized[:, :end][:, -15:]
        torch.testing.assert_close(carried.input_history, expected_history, rtol=0, atol=0)
        assert carried.input_history.untyped_storage().nbytes() == carried.input_history.numel() * 4
        if previous_history is not None:
            torch.testing.assert_close(previous_history, snapshot, rtol=0, atol=0)
    assert carried is not None
    torch.testing.assert_close(torch.cat(chunk_predictions, dim=1), full_prediction)

    step_state: TransformerState | None = None
    step_predictions: list[Tensor] = []
    for row in range(values.shape[1]):
        prediction, _, step_state = model(values[:, row : row + 1], step_state)
        step_predictions.append(prediction)
        assert step_state.next_position == row + 1
    torch.testing.assert_close(torch.cat(step_predictions, dim=1), full_prediction)


def test_kernel16_is_causal_batch_independent_and_resettable() -> None:
    model = build_sequence_model(_config(input_stem_kernel=16, context_rows=1))
    model.eval()
    with torch.no_grad():
        model.input_stem.weight.zero_()
        model.input_stem.weight[:, 0, -2] = 0.5
    values = torch.randn(3, 24, 112)
    prediction, _, state = model(values)
    altered = values.clone()
    altered[:, 16:] += 100.0
    altered_prediction = model(altered)[0]
    torch.testing.assert_close(prediction[:, :16], altered_prediction[:, :16])

    individual = torch.cat([model(values[index : index + 1])[0] for index in range(3)], dim=0)
    torch.testing.assert_close(prediction, individual)
    next_values = torch.randn(3, 5, 112)
    continued = model(next_values, state)[0]
    reset, _, reset_state = model(next_values)
    torch.testing.assert_close(reset, model(next_values, None)[0])
    assert isinstance(reset_state, TransformerState)
    assert reset_state.next_position == next_values.shape[1]
    assert state.next_position == values.shape[1]
    assert not torch.equal(continued, reset)


def test_state_history_validation_and_detachment() -> None:
    model = build_sequence_model(_config(input_stem_kernel=16))
    values = torch.randn(2, 17, 112, requires_grad=True)
    prediction, _, state = model(values)
    assert isinstance(state, TransformerState) and state.input_history is not None
    (prediction.square().mean()).backward()
    assert model.input_stem.weight.grad is not None
    assert torch.isfinite(model.input_stem.weight.grad).all()
    assert torch.count_nonzero(model.input_stem.weight.grad) > 0

    detached = detach_state(state)
    assert isinstance(detached, TransformerState)
    assert detached.input_history is not None and detached.input_history.grad_fn is None
    assert detached.input_history.dtype == torch.float32
    assert state.input_history.grad_fn is not None

    bad_states = (
        replace(state, input_history=torch.empty(1, 15, 112)),
        replace(state, input_history=torch.empty(2, 14, 112)),
        replace(state, input_history=torch.empty(2, 15, 112, dtype=torch.bfloat16)),
        replace(state, input_history=torch.empty(2, 15, 112, device="meta")),
        replace(state, input_history=[0]),
        replace(state, input_history=None),
    )
    for bad_state in bad_states:
        with pytest.raises((TypeError, ValueError), match="input history"):
            model(torch.randn(2, 1, 112), bad_state)
    for position in (True, -1, 1.5):
        with pytest.raises(ValueError, match="absolute position"):
            model(torch.randn(2, 1, 112), replace(state, next_position=position))
    with pytest.raises(TypeError, match="TransformerState"):
        model(torch.randn(2, 1, 112), torch.zeros(2, 17, 112))


def test_legacy_and_kernel1_states_reject_foreign_history() -> None:
    values = torch.randn(2, 4, 112)
    for kernel_size in (0, 1):
        model = build_sequence_model(_config(input_stem_kernel=kernel_size))
        _, _, state = model(values)
        assert isinstance(state, TransformerState)
        assert state.input_history is None
        foreign = replace(state, input_history=torch.zeros(2, 1, 112))
        with pytest.raises(ValueError, match="input history"):
            model(values[:, :1], foreign)


def test_input_stem_parameter_is_owned_by_decay_enabled_adamw_matrix_group() -> None:
    model = build_sequence_model(_config(input_stem_kernel=16))
    optimizer = build_sequence_muon_optimizer(
        model,
        {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "muon_ns_eps": 1e-7,
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.95,
            "adamw_eps": 1e-8,
            "model": {"architecture": "transformer"},
        },
    )
    by_role = {str(group["role"]): group for group in optimizer.param_groups}
    matrix_group = by_role["adamw_matrix"]
    assert "input_stem.weight" in matrix_group["param_names"]
    parameter = model.input_stem.weight
    assert parameter.ndim == 3
    assert any(item is parameter for item in matrix_group["params"])
    assert matrix_group["weight_decay"] == pytest.approx(0.01)
    assert all(
        "input_stem.weight" not in group["param_names"]
        for role, group in by_role.items()
        if role.startswith("muon_")
    )
