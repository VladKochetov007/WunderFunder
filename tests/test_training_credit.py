from __future__ import annotations

import copy
import itertools

import pytest
import torch

from tests.test_training_controls import _controlled, _fixture
from tests.training_dynamics_helpers import options
from wnn.training_credit import credit_edge_scales, scale_hidden_credit
from wnn.training_dynamics import Dynamics, build_optimizer, build_scheduler


def _gradient(mask: tuple[bool, bool] | None) -> torch.Tensor:
    """Independent nonlinear two-state tied-parameter graph, including direct loss paths."""
    parameter = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    state = torch.tensor([0.7, -0.3], dtype=torch.float64)
    loss = parameter.new_zeros(())
    for row in range(3):
        state = torch.stack(
            (
                parameter[0] * state[0].square() + state[1],
                parameter[1] * state[0] + 0.5 * state[1].square(),
            )
        )
        loss = loss + (state[0] + parameter[1] * state[1]).square()
        if row < 2 and mask is not None:
            state = scale_hidden_credit(state, 2.0) if mask[row] else state.detach()
    return torch.autograd.grad(loss, parameter)[0]


def test_exact_enumeration_matches_nonlinear_shared_graph_gradient() -> None:
    gradients = [_gradient(mask) for mask in itertools.product((False, True), repeat=2)]
    expected = _gradient(None)
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_affine_sgd_step_expectation_given_fixed_incoming_momentum(momentum: float) -> None:
    def step(gradient: torch.Tensor) -> torch.Tensor:
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=momentum, weight_decay=0.03)
        if momentum:
            optimizer.state[parameter]["momentum_buffer"] = torch.tensor(
                [0.1, -0.3], dtype=torch.float64
            )
        parameter.grad = gradient.clone()
        optimizer.step()
        return parameter.detach()

    estimates = [step(_gradient(mask)) for mask in itertools.product((False, True), repeat=2)]
    torch.testing.assert_close(
        torch.stack(estimates).mean(0), step(_gradient(None)), rtol=2e-14, atol=2e-14
    )
    # This scope does not extend to a nonlinear clipped-gradient transform.
    values = torch.tensor([-1.0, 3.0])
    assert values.mean() == 1
    assert values.clamp(-1, 1).mean() != values.mean().clamp(-1, 1)


def test_actual_distilled_gru_compensation_matches_full_window_gradient() -> None:
    fixture = _fixture()
    reference = _controlled(copy.deepcopy(fixture), 1, 4)
    compensated, ordinary = [], []
    for mask in itertools.product((False, True), repeat=3):
        result = _controlled(
            copy.deepcopy(fixture),
            1,
            4,
            credit_scales=tuple(2.0 if alive else 0.0 for alive in mask),
        )
        assert result[0] == pytest.approx(reference[0], abs=2e-6)
        assert torch.equal(result[1], reference[1])
        compensated.append(result[3])
        ordinary.append(
            _controlled(
                copy.deepcopy(fixture), 1, 4, credit_scales=tuple(float(alive) for alive in mask)
            )[3]
        )
    # Loss arithmetic intentionally follows the trainer's fp32 path.
    torch.testing.assert_close(torch.stack(compensated).mean(0), reference[3], rtol=3e-6, atol=3e-6)
    assert not torch.allclose(torch.stack(ordinary).mean(0), reference[3], rtol=1e-5, atol=1e-6)
    full = _controlled(copy.deepcopy(fixture), 1, 4, credit_scales=(1.0, 1.0, 1.0))
    assert torch.equal(full[3], reference[3])


def test_private_gates_share_exact_random_cuts_between_paired_arms() -> None:
    state = torch.get_rng_state().clone()
    compensated = credit_edge_scales(100, 0.8, 13, 7, True)
    ordinary = credit_edge_scales(100, 0.8, 13, 7, False)
    assert [factor > 0 for factor in compensated] == [factor > 0 for factor in ordinary]
    assert compensated == credit_edge_scales(100, 0.8, 13, 7, True)
    assert set(compensated) == {0.0, 1.25}
    assert torch.equal(state, torch.get_rng_state())


def test_sgd_schedule_preserves_declared_momentum_and_unclipped_errors() -> None:
    model = torch.nn.Linear(2, 1)
    args = options(optim="sgd", momentum=0.3, no_gradient_clip=True)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, 60)
    dynamics = Dynamics(model, args, 60)
    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 100.0)
        dynamics.perturb_and_clip(0)
        assert all((parameter.grad == 100).all() for parameter in model.parameters())
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["momentum"] == 0.3
    next(model.parameters()).grad.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        dynamics.perturb_and_clip(0)


@pytest.mark.parametrize("probability", [0.0, -1.0, float("nan"), 1.1])
def test_zero_support_or_invalid_probability_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        credit_edge_scales(3, probability, 1, 0, True)
