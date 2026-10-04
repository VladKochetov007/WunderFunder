from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wnn.nn import RNNModel
from wnn.training_controls import (
    DistilledLoss,
    blended_target,
    distilled_macro_losses,
    ema_decay_per_update,
    shuffled_masses,
)


def _fixture() -> tuple:
    torch.manual_seed(41)
    model = RNNModel(3, hidden=4, layers=1, n_out=10)
    average = copy.deepcopy(model)
    with torch.no_grad():
        average.head[2].bias.add_(0.3)
    inputs = torch.randn(2, 8, 3)
    labels = torch.randn(2, 8, 2) * 2
    teachers = torch.randn(2, 8, 2)
    weights = torch.tensor([[0, 0.5, 7.0, 9.0, 0.2, 0.3, 0.4, 0.5]]).repeat(2, 1)
    next_change = torch.tensor([[2, 2, 6, 6, 6, 6, 8, 8]]).repeat(2, 1)
    structural = tuple(torch.randint(size, (2, 8)) for size in (1, 3, 4))
    next_labels = (torch.rand(2, 8) * 4, torch.rand(2, 8), weights * 0.7)
    # Structural output uses 1+3+4 columns after main two, so next-event is disabled
    # here and exercised separately with a wider head below.
    config = DistilledLoss(0.5, 1, 0.7, False, 0.3, 2, 1, 0, 10)
    return (
        model,
        average,
        inputs,
        labels,
        teachers,
        weights,
        next_change,
        structural,
        next_labels,
        config,
    )


def _oracle(output: torch.Tensor, averaged: torch.Tensor, fixture: tuple) -> torch.Tensor:
    _, _, _, labels, teachers, weights, change, structural, next_labels, config = fixture
    clipped = labels[:, :4].clamp(-2, 2)
    target = clipped.clone()
    for batch in range(2):
        for row in range(4):
            ahead = row + config.bootstrap_horizon
            if (
                config.bootstrap_horizon
                and change[batch, row] > ahead
                and ahead < teachers.shape[1]
            ):
                target[batch, row] = teachers[batch, ahead].clamp(-2, 2)
    target = (1 - config.alpha) * target + config.alpha * teachers[:, :4].clamp(-2, 2)
    weighted = clipped.abs() * weights[:, :4, None]
    loss = (weighted * (output[..., :2] - target).square()).sum() / weighted.sum()
    consistency = weights[:, :4, None]
    if config.ema_metric_weight:
        consistency = consistency * clipped.abs()
    multiplier = 2 if config.ema_metric_weight else 1
    loss = (
        loss
        + config.ema
        * multiplier
        * (consistency * (output[..., :2] - averaged[..., :2]).square()).sum()
        / consistency.sum()
    )
    if config.structural:
        offset = 2
        structural_loss = 0
        for classes, label in zip((1, 3, 4), structural, strict=True):
            ce = F.cross_entropy(
                output[..., offset : offset + classes].reshape(-1, classes),
                label[:, :4].reshape(-1),
                reduction="none",
            )
            structural_loss = structural_loss + (ce * weights[:, :4].reshape(-1)).sum()
            offset += classes
        loss = loss + config.structural * structural_loss / weights[:, :4].sum()
    if config.next_event:
        distance, sign, weight = next_labels
        error = (output[..., 10] - (distance[:, :4] - 3) / 2).square()
        error = error + F.binary_cross_entropy_with_logits(
            output[..., 11], sign[:, :4], reduction="none"
        )
        loss = loss + config.next_event * (error * weight[:, :4]).sum() / weight[:, :4].sum()
    return loss


def _controlled(fixture: tuple, micro: int, detach: int, **kwargs: object) -> tuple:
    model, average, inputs, labels, teacher, weights, change, structural, next_labels, config = (
        fixture
    )
    parts = distilled_macro_losses(
        forward=lambda rows, state: model(inputs[:, rows], state),
        ema_forward=lambda rows, state: average(inputs[:, rows], state),
        state=None,
        ema_state=None,
        labels=labels,
        teacher=teacher,
        row_weights=weights,
        next_change=change,
        structural_labels=structural,
        next_labels=next_labels,
        rows=slice(0, 4),
        micro_rows=micro,
        detach_rows=detach,
        config=config,
        **kwargs,
    )
    value = 0.0
    count = 0
    for loss, _state, _averaged_state in parts:
        value += loss.item()
        count += 1
        loss.backward()
    gradients = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    return value, _state, _averaged_state, gradients, count


@pytest.mark.parametrize("metric_weight", [False, True])
@pytest.mark.parametrize("micro", [2, 4])
def test_pooled_loss_and_uncut_gradient_match_independent_whole_oracle(
    metric_weight: bool, micro: int
) -> None:
    fixture = list(_fixture())
    fixture[0] = RNNModel(3, hidden=4, layers=1, n_out=12)
    fixture[1] = copy.deepcopy(fixture[0])
    fixture[-1] = fixture[-1]._replace(next_event=0.3, ema_metric_weight=metric_weight)
    reference = copy.deepcopy(fixture[0])
    output, state = reference(fixture[2][:, :4])
    with torch.no_grad():
        averaged, ema_state = fixture[1](fixture[2][:, :4])
    loss = _oracle(output, averaged, tuple(fixture))
    loss.backward()
    expected_gradients = torch.cat(
        [parameter.grad.flatten() for parameter in reference.parameters()]
    )
    actual, hidden, actual_ema, gradients, count = _controlled(tuple(fixture), micro, 4)
    assert actual == pytest.approx(loss.item(), abs=1e-6)
    torch.testing.assert_close(hidden, state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_ema, ema_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradients, expected_gradients, atol=2e-6, rtol=1e-5)
    assert count == 1


def test_detach_cuts_credit_but_carries_identical_state_and_loss() -> None:
    fixture = _fixture()
    short = _controlled(copy.deepcopy(fixture), 2, 2)
    long = _controlled(copy.deepcopy(fixture), 2, 4)
    assert short[0] == pytest.approx(long[0], abs=1e-6)
    assert torch.equal(short[1], long[1])
    assert not torch.allclose(short[3], long[3], atol=1e-7, rtol=1e-7)
    assert short[-1] == 2 and long[-1] == 1


def test_strict_bootstrap_boundary_and_end_of_sequence() -> None:
    labels = torch.ones(1, 4, 2)
    teacher = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) / 4
    change = torch.tensor([[1, 3, 3, 4]])
    result = blended_target(labels, teacher, change, slice(0, 4), 0, 1)
    assert torch.equal(result[0, 0], labels[0, 0])  # equality at the boundary is not alive
    assert torch.equal(result[0, 1], teacher[0, 2])
    assert torch.equal(result[0, 2:], labels[0, 2:])


def test_normalization_fixed_and_sham_agree_for_constant_masses() -> None:
    fixture = _fixture()
    mass = (fixture[3][:, :4].clamp(-2, 2).abs() * fixture[5][:, :4, None]).sum()
    fixed = _controlled(
        copy.deepcopy(fixture), 2, 2, denominator="fixed", reference_mass=mass.item()
    )
    sham = _controlled(
        copy.deepcopy(fixture),
        2,
        2,
        denominator="shuffled",
        reference_mass=mass.item(),
        donor_mass=mass,
    )
    assert fixed[0] == sham[0]
    assert torch.equal(fixed[3], sham[3])


def test_mass_permutation_is_private_and_preserves_multiset() -> None:
    weights = torch.arange(1, 17, dtype=torch.float32).reshape(2, 4, 2)
    rng = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    result = shuffled_masses(weights, 2, 13)
    expected = torch.stack([weights[:, :2].sum(), weights[:, 2:].sum()])
    assert torch.equal(torch.sort(result).values, torch.sort(expected).values)
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_ema_row_clock_and_legacy_default() -> None:
    assert ema_decay_per_update(0.999, 1000, 0) == 0.999
    assert ema_decay_per_update(0.999, 1000, 2000) ** 2 == pytest.approx(0.999)
    with pytest.raises(ValueError, match="invalid"):
        ema_decay_per_update(0.999, 1000, -1)
