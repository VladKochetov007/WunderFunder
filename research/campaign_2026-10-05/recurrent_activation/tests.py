"""CPU parity, stability and streaming checks for the E471 custom GRU scan."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))

import fit_activation_optimizer as e470  # noqa: E402
from fit_activation_optimizer import ActivationGRU as E470ActivationGRU  # noqa: E402
from fit_recurrent_candidate import (  # noqa: E402
    ALPHAS,
    ARMS,
    CandidateGRUModel,
    candidate_from_tanh,
    make_optimizer,
    scripted_candidate_scan,
)


def map_model_parameters(
    reference: nn.Module, candidate: CandidateGRUModel
) -> list[tuple[str, nn.Parameter, nn.Parameter]]:
    reference_params = dict(reference.named_parameters())
    candidate_params = dict(candidate.named_parameters())
    if reference_params.keys() != candidate_params.keys():
        raise AssertionError("E470 reference and candidate parameter names diverged")
    return [(name, reference_params[name], candidate_params[name]) for name in reference_params]


def check_candidate_bound() -> None:
    u = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    for _, alpha in ALPHAS:
        candidate = candidate_from_tanh(u, alpha)
        if not torch.isfinite(candidate).all():
            raise AssertionError(f"nonfinite candidate for alpha={alpha}")
        if candidate.min() < -1.0 - 1e-12 or candidate.max() > 1.0 + 1e-12:
            raise AssertionError(f"candidate escaped [-1,1] for alpha={alpha}")
        torch.testing.assert_close(candidate[[0, -1]], torch.tensor([-1.0, 1.0], dtype=u.dtype))


def check_custom_scan_alpha_zero_parity() -> None:
    """Compare fused nn.GRU output, input/weight gradients and AdamW update."""
    torch.manual_seed(115)
    reference = nn.GRU(32, 32, batch_first=True)
    custom = copy.deepcopy(reference)
    x_reference = torch.randn(2, 29, 32, requires_grad=True)
    x_custom = x_reference.detach().clone().requires_grad_(True)
    h0 = torch.randn(2, 32)
    out_reference, hn_reference = reference(x_reference, h0.unsqueeze(0))
    out_custom, hn_custom = scripted_candidate_scan(
        x_custom,
        h0,
        custom.weight_ih_l0,
        custom.weight_hh_l0,
        custom.bias_ih_l0,
        custom.bias_hh_l0,
        0.0,
    )
    torch.testing.assert_close(out_custom, out_reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(hn_custom, hn_reference[0], atol=2e-6, rtol=2e-6)
    loss_reference = out_reference.square().mean() + 0.1 * hn_reference.square().mean()
    loss_custom = out_custom.square().mean() + 0.1 * hn_custom.square().mean()
    loss_reference.backward()
    loss_custom.backward()
    torch.testing.assert_close(x_custom.grad, x_reference.grad, atol=3e-6, rtol=3e-5)
    for (name_ref, parameter_ref), (name_custom, parameter_custom) in zip(
        reference.named_parameters(), custom.named_parameters(), strict=True
    ):
        if name_ref != name_custom:
            raise AssertionError("reference/custom GRU parameter ordering diverged")
        torch.testing.assert_close(parameter_custom.grad, parameter_ref.grad, atol=4e-6, rtol=4e-5)
    optimizer_reference = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_custom = torch.optim.AdamW(custom.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_reference.step()
    optimizer_custom.step()
    for (name_ref, parameter_ref), (name_custom, parameter_custom) in zip(
        reference.named_parameters(), custom.named_parameters(), strict=True
    ):
        if name_ref != name_custom:
            raise AssertionError("parameter names diverged after optimizer step")
        torch.testing.assert_close(parameter_custom, parameter_ref, atol=4e-6, rtol=4e-5)


def check_e470_initialization_and_full_model_step() -> None:
    """Alpha zero must match E470 tanh embedding/head initialization and step."""
    torch.manual_seed(271828)
    reference = E470ActivationGRU("tanh")
    torch.manual_seed(271828)
    custom = CandidateGRUModel(0.0)
    mapped = map_model_parameters(reference, custom)
    if any(
        not torch.equal(parameter_ref, parameter_custom)
        for _, parameter_ref, parameter_custom in mapped
    ):
        raise AssertionError("alpha-zero candidate model changed E470 initialization")
    x = torch.randn(4, 32, 42)
    target = torch.randn(4, 32, 2).clamp(-2, 2)
    need = torch.ones(4, 32, dtype=torch.bool)
    prediction_reference, state_reference = reference(x)
    prediction_custom, state_custom = custom(x)
    torch.testing.assert_close(prediction_custom, prediction_reference, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state_custom, state_reference, atol=3e-6, rtol=3e-6)
    loss_reference = e470.loss_for_batch(prediction_reference, target, need)
    loss_custom = e470.loss_for_batch(prediction_custom, target, need)
    loss_reference.backward()
    loss_custom.backward()
    for name, parameter_ref, parameter_custom in mapped:
        torch.testing.assert_close(
            parameter_custom.grad, parameter_ref.grad, atol=5e-6, rtol=5e-5, msg=name
        )
    optimizer_reference = torch.optim.AdamW(
        reference.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    optimizer_custom = torch.optim.AdamW(
        custom.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    optimizer_reference.step()
    optimizer_custom.step()
    for name, parameter_ref, parameter_custom in map_model_parameters(reference, custom):
        torch.testing.assert_close(parameter_custom, parameter_ref, atol=6e-6, rtol=6e-5, msg=name)


def check_streaming_and_future_causality() -> None:
    torch.manual_seed(206)
    x = torch.randn(2, 43, 42)
    changed = x.clone()
    changed[:, 17:] += 7.0
    for alpha_name, alpha in ALPHAS:
        model = CandidateGRUModel(alpha).eval()
        full, final_state = model(x)
        altered, _ = model(changed)
        if not torch.equal(full[:, :17], altered[:, :17]):
            raise AssertionError(f"future input changed prefix for alpha={alpha}")
        left, state = model(x[:, :17])
        right, chunked_state = model(x[:, 17:], state)
        torch.testing.assert_close(torch.cat((left, right), dim=1), full, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(chunked_state, final_state, atol=3e-6, rtol=3e-6)
        reset_output, reset_state = model(x, None)
        torch.testing.assert_close(reset_output, full, atol=0, rtol=0)
        torch.testing.assert_close(reset_state, final_state, atol=0, rtol=0)
        if not torch.isfinite(full).all() or not torch.isfinite(final_state).all():
            raise AssertionError(f"nonfinite stream for alpha={alpha_name}")
        if final_state.abs().max() > 1.0 + 5e-6:
            raise AssertionError(f"bounded recurrent state escaped range for alpha={alpha}")


def check_six_arm_optimizer_updates() -> None:
    torch.manual_seed(82)
    x = torch.randn(4, 256, 42)
    target = torch.randn(4, 256, 2).clamp(-2, 2)
    need = torch.ones(4, 256, dtype=torch.bool)
    for alpha_name, alpha in ALPHAS:
        for optimizer_name in ("adamw", "muon_adamw"):
            torch.manual_seed(271828)
            model = CandidateGRUModel(alpha)
            optimizer = make_optimizer(model, optimizer_name)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            prediction, _ = model(x)
            loss = e470.loss_for_batch(prediction, target, need)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            e470.ensure_finite_parameters(model, f"{alpha_name}/{optimizer_name}")
            if not any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            ):
                raise AssertionError(f"optimizer made no update for {alpha_name}/{optimizer_name}")


def main() -> None:
    torch.set_num_threads(1)
    if len(ARMS) != 6:
        raise AssertionError("expected three candidate alphas crossed with two optimizers")
    check_candidate_bound()
    check_custom_scan_alpha_zero_parity()
    check_e470_initialization_and_full_model_step()
    check_streaming_and_future_causality()
    check_six_arm_optimizer_updates()
    print("E471 recurrent candidate CPU tests passed")


if __name__ == "__main__":
    main()
