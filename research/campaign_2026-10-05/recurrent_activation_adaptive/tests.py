"""Synthetic invariants for the E479 adaptive candidate recurrence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E474_DIR))
import fit_activation_placement as e474  # noqa: E402
from fit_recurrent_activation_adaptive import (  # noqa: E402
    ALPHA_INIT_POSITIVE_RAW,
    ALPHA_MAX,
    candidate_from_tanh_tensor,
    initialize_arm,
    make_optimizer,
    seed_states,
)


def _assert_close(left: Tensor, right: Tensor, atol: float = 2e-5, rtol: float = 2e-4) -> None:
    """Assert numerical closeness and finite values with explicit tolerances."""
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_candidate_bound_and_monotonicity() -> None:
    """Check registered scalar map properties across raw coefficient domain."""
    raw = torch.linspace(-20.0, 20.0, 1001, dtype=torch.float64)
    alpha = ALPHA_MAX * torch.tanh(raw)
    u = torch.linspace(-1.0, 1.0, 20_001, dtype=torch.float64)
    for coefficient in (alpha.min(), alpha.max(), alpha[250], alpha[750]):
        values = u + coefficient * torch.sin(math.pi * u)
        assert values.min() >= -1.0 - 1e-12
        assert values.max() <= 1.0 + 1e-12
        assert torch.all(values[1:] >= values[:-1])
        derivative = 1.0 + coefficient * math.pi * torch.cos(math.pi * u)
        assert derivative.min() >= 1.0 - 0.2 * math.pi - 1e-12
    torch.testing.assert_close(
        alpha.min(), torch.tensor(-0.2, dtype=torch.float64), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        alpha.max(), torch.tensor(0.2, dtype=torch.float64), atol=1e-9, rtol=0
    )


def test_alpha_zero_full_model_parity_outputs_states_and_gradients() -> None:
    """Compare independent custom scan to incumbent nn.GRU at alpha zero."""
    torch.manual_seed(412)
    baseline = e474.PlacementRNN("gelu", "gelu")
    state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    candidate = initialize_arm("fixed0", state)
    x_base = torch.randn(2, 11, 42, requires_grad=True)
    x_candidate = x_base.detach().clone().requires_grad_(True)
    output_base, hidden_base = baseline(x_base)
    output_candidate, hidden_candidate = candidate(x_candidate)
    _assert_close(output_candidate, output_base)
    _assert_close(hidden_candidate, hidden_base)
    (output_base.square().mean() + hidden_base.square().mean()).backward()
    (output_candidate.square().mean() + hidden_candidate.square().mean()).backward()
    _assert_close(x_candidate.grad, x_base.grad, atol=5e-5, rtol=5e-4)
    base_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert set(base_parameters) == set(candidate_parameters) - {"raw_alpha"}
    for name, parameter in base_parameters.items():
        _assert_close(candidate_parameters[name].grad, parameter.grad, atol=5e-5, rtol=5e-4)
    base_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    candidate_optimizer = make_optimizer(candidate, "adamw")
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
    base_optimizer.step()
    candidate_optimizer.step()
    updated_candidate = dict(candidate.named_parameters())
    for name, parameter in baseline.named_parameters():
        _assert_close(updated_candidate[name], parameter, atol=1e-4, rtol=1e-3)


def test_initialization_parity_and_state_chunk_causality() -> None:
    """Verify matched effective-alpha inits, streaming state and future isolation."""
    states = seed_states()
    for fixed_name, learned_name in (("fixed0", "learned0"), ("fixed01", "learned01")):
        fixed = initialize_arm(fixed_name, states[271828])
        learned = initialize_arm(learned_name, states[271828])
        expected = 0.0 if fixed_name == "fixed0" else 0.1
        torch.testing.assert_close(
            learned.effective_alpha(), torch.full((2,), expected), atol=1e-7, rtol=0
        )
        torch.manual_seed(98)
        x = torch.randn(2, 13, 42)
        fixed_output, fixed_state = fixed(x)
        learned_output, learned_state = learned(x)
        _assert_close(learned_output, fixed_output)
        _assert_close(learned_state, fixed_state)
        first, state = learned(x[:, :5])
        second, state = learned(x[:, 5:], state)
        _assert_close(torch.cat((first, second), dim=1), learned_output)
        _assert_close(state, learned_state)
        perturbed = x.clone()
        perturbed[:, 9:, :] += 100.0
        changed, _ = learned(perturbed)
        _assert_close(changed[:, :9], learned_output[:, :9])
        reset_a, reset_state = learned(x[:, :4])
        reset_b, fresh_state = learned(x[:, :4])
        _assert_close(reset_a, reset_b)
        _assert_close(reset_state, fresh_state)
        assert reset_state.shape == (2, 2, 64)


def test_learned_alpha_autograd_bounds_and_optimizer_groups() -> None:
    """Require live finite nonzero coefficient gradients and zero-decay groups."""
    states = seed_states()
    for condition, expected_raw in (("learned0", 0.0), ("learned01", ALPHA_INIT_POSITIVE_RAW)):
        model = initialize_arm(condition, states[271828])
        assert model.raw_alpha is not None and model.raw_alpha.requires_grad
        torch.testing.assert_close(model.raw_alpha, torch.full((2,), expected_raw))
        optimizer = make_optimizer(model, "muon_adamw")
        alpha_groups = [
            group
            for group in optimizer.param_groups
            if any(parameter is model.raw_alpha for parameter in group["params"])
        ]
        assert len(alpha_groups) == 1
        assert alpha_groups[0]["weight_decay"] == 0.0
        assert not alpha_groups[0].get("muon", False)
        torch.manual_seed(19)
        x = torch.randn(4, 17, 42)
        prediction, state = model(x)
        loss = prediction.square().mean() + state.square().mean()
        loss.backward()
        assert model.raw_alpha.grad is not None
        assert torch.isfinite(model.raw_alpha.grad).all()
        assert (model.raw_alpha.grad.abs() > 0).all()
        alpha0 = model.effective_alpha().detach().clone()
        optimizer.step()
        assert (model.raw_alpha.detach() != torch.full_like(model.raw_alpha, expected_raw)).any()
        alpha1 = model.effective_alpha()
        assert torch.isfinite(alpha1).all()
        assert (alpha1.abs() < ALPHA_MAX).all()
        assert not torch.equal(alpha0, alpha1)
        # The scripted scalar operation must preserve gradient to a tensor alpha.
        scalar_alpha = torch.tensor(0.07, requires_grad=True)
        u = torch.linspace(-0.8, 0.8, 9)
        candidate_from_tanh_tensor(u, scalar_alpha).sum().backward()
        assert scalar_alpha.grad is not None and torch.isfinite(scalar_alpha.grad)
        assert scalar_alpha.grad.abs() > 0


def test_optimizer_exact_partition_for_both_families() -> None:
    """Ensure alpha is an AdamW zero-decay group and Muon sees only GRU mats."""
    state = seed_states()[271829]
    for condition in ("fixed0", "fixed01", "learned0", "learned01"):
        model = initialize_arm(condition, state)
        for optimizer_name in ("adamw", "muon_adamw"):
            optimizer = make_optimizer(model, optimizer_name)
            ids = [
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ]
            expected = {id(parameter) for parameter in model.parameters()}
            assert len(ids) == len(set(ids))
            assert set(ids) == expected
            if condition.startswith("learned"):
                alpha_group = next(
                    group
                    for group in optimizer.param_groups
                    if any(parameter is model.raw_alpha for parameter in group["params"])
                )
                assert alpha_group["weight_decay"] == 0.0


def main() -> None:
    """Run all E479 synthetic source checks on CPU."""
    torch.set_num_threads(1)
    for test in (
        test_candidate_bound_and_monotonicity,
        test_alpha_zero_full_model_parity_outputs_states_and_gradients,
        test_initialization_parity_and_state_chunk_causality,
        test_learned_alpha_autograd_bounds_and_optimizer_groups,
        test_optimizer_exact_partition_for_both_families,
    ):
        test()
        print(f"PASS {test.__name__}", flush=True)


if __name__ == "__main__":
    main()
