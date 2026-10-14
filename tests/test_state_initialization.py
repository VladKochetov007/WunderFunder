"""Causal timing, reset and gradient checks for the separate initializer arm."""

from __future__ import annotations

import copy

import pytest
import torch

from wnn.nn import RNNModel
from wnn.state_initialization import InitializedGRU


def make_model(mode: str) -> InitializedGRU:
    torch.manual_seed(43)
    model = InitializedGRU(RNNModel(211, hidden=8, layers=2), mode)
    model.eval()
    return model


@pytest.mark.parametrize("mode", ["zero", "constant", "first_row", "prefix"])
def test_zero_initialized_arms_preserve_backbone(mode: str) -> None:
    model = make_model(mode)
    inputs = torch.randn(2, 143, 211)
    with torch.no_grad():
        expected, _ = model.backbone(inputs)
        actual, state = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert state.position == 143


@pytest.mark.parametrize("mode", ["constant", "first_row", "prefix"])
def test_chunk_parity_and_complete_reset(mode: str) -> None:
    model = make_model(mode)
    with torch.no_grad():
        if model.constant is not None:
            model.constant.fill_(0.4)
        else:
            model.predictor[-1].weight.normal_(std=0.05)
            model.predictor[-1].bias.fill_(0.2)
        inputs = torch.randn(2, 221, 211)
        full, full_state = model(inputs)
        state = None
        pieces = []
        for begin, end in [(0, 1), (1, 37), (37, 98), (98, 99), (99, 100), (100, 221)]:
            output, state = model(inputs[:, begin:end], state)
            pieces.append(output)
        reset, reset_state = model(inputs, None)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.hidden, full_state.hidden, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reset, full, rtol=0, atol=0)
    torch.testing.assert_close(reset_state.hidden, full_state.hidden, rtol=0, atol=0)
    assert state.prefix_mean is None


def test_prefix_uses_rows_0_to_98_and_adjusts_before_99() -> None:
    model = make_model("prefix")
    inputs = torch.randn(1, 130, 211)
    with torch.no_grad():
        model.predictor[-1].weight.normal_(std=0.1)
        before, state = model(inputs[:, :99])
        warmup, original_hidden = model.backbone(inputs[:, :99])
        normalized = (inputs[:, :99] - model.backbone.mean) / model.backbone.std
        context = torch.cat(
            (normalized.mean(1), normalized.std(1, correction=0), normalized[:, -1]), dim=-1
        )
        expected, _ = model.backbone(inputs[:, 99:100], original_hidden + model._predict(context))
        actual, _ = model(inputs[:, 99:100], state)
        changed_future = inputs.clone()
        changed_future[:, 100:] += 100
        full, _ = model(inputs)
        changed, _ = model(changed_future)
    torch.testing.assert_close(before, warmup, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full[:, :100], changed[:, :100], rtol=0, atol=0)
    assert state.position == 99


def test_frozen_backbone_still_transmits_initializer_gradients() -> None:
    model = make_model("prefix")
    model.backbone.requires_grad_(False)
    inputs = torch.randn(1, 121, 211)
    output, state = model(inputs)
    output[:, 99:].square().mean().backward()
    assert model.predictor[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert not state.detach().hidden.requires_grad


def test_long_sequence_remains_finite_and_state_is_sequence_local() -> None:
    model = make_model("prefix")
    other = copy.deepcopy(model)
    with torch.no_grad():
        model.predictor[-1].bias.fill_(2)
        inputs = torch.randn(1, 20_000, 211)
        output, state = model(inputs)
        untouched, _ = other(inputs[:, :100])
        baseline, _ = other.backbone(inputs[:, :100])
    assert torch.isfinite(output).all()
    assert torch.isfinite(state.hidden).all()
    assert state.position == 20_000
    torch.testing.assert_close(untouched, baseline, rtol=1e-5, atol=1e-6)
