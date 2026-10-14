from __future__ import annotations

import copy

import pytest
import torch

from wnn.initializer_training import initializer_sequence_loss
from wnn.nn import RNNModel
from wnn.state_initialization import InitializedGRU


@pytest.mark.parametrize("mode", ["constant", "first_row", "prefix"])
def test_checkpointed_initializer_loss_and_gradients_match_complete_sequence(mode: str) -> None:
    torch.manual_seed(20261014)
    reference = InitializedGRU(RNNModel(5, hidden=3, layers=2), mode)
    reference.backbone.requires_grad_(False)
    # Nonzero initializer output makes the test exercise all predictor layers.
    if reference.predictor is not None:
        torch.nn.init.normal_(reference.predictor[2].weight, std=0.1)
    chunked = copy.deepcopy(reference)
    inputs = torch.randn(2, 241, 5)
    targets = torch.randn(2, 241, 2)
    weights = targets.clamp(-2, 2).abs()
    weights[:, :99] = 0
    predictions, _ = reference(inputs)
    loss = ((predictions[..., :2] - targets).square() * weights).sum() / weights.sum()
    loss.backward()
    actual = initializer_sequence_loss(chunked, inputs, targets, weights, chunk_rows=100)
    actual.backward()
    torch.testing.assert_close(actual, loss, rtol=2e-6, atol=2e-7)
    assert all(parameter.grad is None for parameter in chunked.backbone.parameters())
    for (name, expected), (other_name, value) in zip(
        reference.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert name == other_name
        if expected.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all()
            torch.testing.assert_close(value.grad, expected.grad, rtol=1e-4, atol=2e-8)


def test_only_late_chunk_loss_still_reaches_initializer() -> None:
    torch.manual_seed(20261014)
    model = InitializedGRU(RNNModel(2, hidden=2, layers=1), "constant")
    # Keep the GRU's retention gate open so the late dependency is measurable.
    with torch.no_grad():
        model.backbone.gru.bias_ih_l0[2:4].fill_(6)
        model.backbone.gru.bias_hh_l0[2:4].zero_()
    model.backbone.requires_grad_(False)
    inputs = torch.randn(1, 301, 2)
    targets = torch.ones(1, 301, 2)
    weights = torch.zeros_like(targets)
    weights[:, 300] = 1
    loss = initializer_sequence_loss(model, inputs, targets, weights, chunk_rows=100)
    loss.backward()
    assert model.constant.grad.abs().max() > 1e-5


def test_initializer_loss_rejects_joint_backbone_updates() -> None:
    model = InitializedGRU(RNNModel(2, hidden=2, layers=1), "constant")
    values = torch.zeros(1, 101, 2)
    with pytest.raises(ValueError, match="frozen backbone"):
        initializer_sequence_loss(model, values, values, values)


def test_initializer_loss_is_invariant_to_weight_scale_above_denominator_floor() -> None:
    torch.manual_seed(20261014)
    model = InitializedGRU(RNNModel(2, hidden=2, layers=1), "constant")
    model.backbone.requires_grad_(False)
    inputs = torch.randn(1, 101, 2)
    targets = torch.ones_like(inputs)
    weights = torch.ones_like(targets)
    ordinary = initializer_sequence_loss(model, inputs, targets, weights)
    ordinary_gradient = torch.autograd.grad(ordinary, model.constant)[0]
    # Scored total mass is below one, but above the shared 1e-8 floor.
    scaled = initializer_sequence_loss(model, inputs, targets, weights * 1e-3)
    scaled_gradient = torch.autograd.grad(scaled, model.constant)[0]
    torch.testing.assert_close(scaled, ordinary)
    torch.testing.assert_close(scaled_gradient, ordinary_gradient)
