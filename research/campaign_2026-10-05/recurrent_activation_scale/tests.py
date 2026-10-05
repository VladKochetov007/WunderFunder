"""CPU parity, boundedness and stream checks for E478."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(ROOT / "src"))

import fit_activation_optimizer as e470  # noqa: E402
from fit_recurrent_activation_scale import (  # noqa: E402
    ALPHAS,
    EXPECTED_PARAMETERS,
    GRAD_ATOL,
    GRAD_RTOL,
    MUON_NAMES,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    SEEDS,
    UPDATE_ATOL,
    UPDATE_RTOL,
    initialize_arm,
    make_initial_states,
    make_muon_optimizer,
)
from fit_recurrent_candidate import candidate_from_tanh  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402


def check_candidate_bounds() -> None:
    """The candidate is monotone and remains in [-1,1] for both fixed alphas."""
    u = torch.linspace(-1, 1, 20_001, dtype=torch.float64)
    for alpha in ALPHAS:
        candidate = candidate_from_tanh(u, alpha)
        derivative = 1 + alpha * torch.pi * torch.cos(torch.pi * u)
        if not torch.isfinite(candidate).all() or not torch.isfinite(derivative).all():
            raise AssertionError(f"nonfinite candidate at alpha={alpha}")
        if candidate.min() < -1 - 1e-12 or candidate.max() > 1 + 1e-12:
            raise AssertionError(f"candidate left [-1,1] at alpha={alpha}")
        if derivative.min() <= 0:
            raise AssertionError(f"candidate is not monotone at alpha={alpha}")
    torch.testing.assert_close(candidate_from_tanh(u[[0, -1]], 0.1), u[[0, -1]])


def check_initialization_names_and_count() -> None:
    """Both alphas clone each seed's actual RNNModel parameter/buffer state."""
    states = make_initial_states()
    for seed in SEEDS:
        for alpha in ALPHAS:
            candidate = initialize_arm(alpha, states[seed])
            baseline = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2)
            baseline.load_state_dict(states[seed], strict=True)
            if (
                sum(parameter.numel() for parameter in candidate.parameters())
                != EXPECTED_PARAMETERS
            ):
                raise AssertionError("trainable parameter count changed")
            if tuple(candidate.state_dict()) != tuple(baseline.state_dict()):
                raise AssertionError("candidate state_dict key names changed")
            if candidate.gru.dropout != 0.0:
                raise AssertionError("recurrent dropout must remain zero")
            for key, value in candidate.state_dict().items():
                if not torch.equal(value, baseline.state_dict()[key]):
                    raise AssertionError(f"initial values differ at {key}")
            optimizer = make_muon_optimizer(candidate)
            trainable = {id(parameter) for parameter in candidate.parameters()}
            assigned = [
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ]
            if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
                raise AssertionError("Muon and AdamW do not form an exact parameter partition")
            selected = {
                name
                for name, parameter in candidate.named_parameters()
                if any(
                    id(parameter) == id(item)
                    for group in optimizer.param_groups
                    if group.get("muon")
                    for item in group["params"]
                )
            }
            if selected != MUON_NAMES:
                raise AssertionError(f"unexpected Muon matrix partition {selected}")


def check_alpha_zero_full_model_parity_and_update() -> None:
    """Compare full RNNModel outputs, state, all gradients and Muon updates."""
    states = make_initial_states()
    torch.manual_seed(88)
    reference = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2)
    reference.load_state_dict(states[SEEDS[0]], strict=True)
    candidate = initialize_arm(0.0, states[SEEDS[0]])
    x0 = torch.randn(4, 256, 42)
    x_reference = x0.clone().requires_grad_(True)
    x_candidate = x0.clone().requires_grad_(True)
    target = torch.randn(4, 256, 2).clamp(-2, 2)
    need = torch.rand(4, 256) > 0.2
    output_reference, state_reference = reference(x_reference)
    output_candidate, state_candidate = candidate(x_candidate)
    torch.testing.assert_close(
        output_candidate, output_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(state_candidate, state_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    loss_reference = e470.loss_for_batch(output_reference, target, need)
    loss_candidate = e470.loss_for_batch(output_candidate, target, need)
    if not (torch.isfinite(loss_reference) and torch.isfinite(loss_candidate)):
        raise AssertionError("nonfinite parity loss")
    loss_reference.backward()
    loss_candidate.backward()
    torch.testing.assert_close(x_candidate.grad, x_reference.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if reference_parameters.keys() != candidate_parameters.keys():
        raise AssertionError("baseline and candidate parameter names changed")
    for name in reference_parameters:
        parameter_reference = reference_parameters[name]
        parameter_candidate = candidate_parameters[name]
        if parameter_reference.grad is None or parameter_candidate.grad is None:
            if parameter_reference.grad is not parameter_candidate.grad:
                raise AssertionError(f"gradient presence differs for {name}")
            continue
        torch.testing.assert_close(
            parameter_candidate.grad,
            parameter_reference.grad,
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            msg=name,
        )

    reference_optimizer = make_muon_optimizer(reference)
    candidate_optimizer = make_muon_optimizer(candidate)
    nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
    reference_optimizer.step()
    candidate_optimizer.step()
    for name, parameter_reference in reference.named_parameters():
        parameter_candidate = candidate_parameters[name]
        torch.testing.assert_close(
            parameter_candidate,
            parameter_reference,
            atol=UPDATE_ATOL,
            rtol=UPDATE_RTOL,
            msg=name,
        )


def check_causality_chunking_state_bounds_and_finiteness() -> None:
    """Check state reset, full/chunk parity, future isolation and finiteness."""
    torch.manual_seed(312)
    x = torch.randn(2, 73, 42)
    changed = x.clone()
    changed[:, 29:] += 8
    stress = x * 1000
    states = make_initial_states()
    for seed in SEEDS:
        for alpha in ALPHAS:
            model = initialize_arm(alpha, states[seed]).eval()
            full, final_state = model(x)
            altered, _ = model(changed)
            if not torch.equal(full[:, :29], altered[:, :29]):
                raise AssertionError(f"future perturbation affected alpha={alpha} prefix")
            left, state = model(x[:, :29])
            right, chunked_state = model(x[:, 29:], state)
            torch.testing.assert_close(torch.cat((left, right), dim=1), full, atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(chunked_state, final_state, atol=3e-6, rtol=3e-6)
            reset, reset_state = model(x, None)
            torch.testing.assert_close(reset, full, atol=0, rtol=0)
            torch.testing.assert_close(reset_state, final_state, atol=0, rtol=0)
            stress_prediction, stress_state = model(stress)
            if not (torch.isfinite(stress_prediction).all() and torch.isfinite(stress_state).all()):
                raise AssertionError(f"nonfinite extreme input at alpha={alpha}")
            if final_state.abs().max() > 1 + 2e-6:
                raise AssertionError(f"bounded hidden state escaped at alpha={alpha}")


def main() -> None:
    torch.set_num_threads(1)
    check_candidate_bounds()
    check_initialization_names_and_count()
    check_alpha_zero_full_model_parity_and_update()
    check_causality_chunking_state_bounds_and_finiteness()
    print("E478 recurrent activation scale CPU invariants passed")


if __name__ == "__main__":
    main()
