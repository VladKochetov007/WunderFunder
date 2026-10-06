"""Evaluate the frozen native/custom recurrence on one E489 fit-only batch."""

from __future__ import annotations

import numpy as np
import torch
from diagnostic_math import (  # noqa: E402
    CLIP_NORM,
    GRAD_ATOL,
    GRAD_RTOL,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    clipping_record,
    compare_delta_groups,
    compare_tensors,
    raw_global_norm,
)
from frozen_runtime import RNNModel, actor, e470, e487


def grouped_update_deltas(
    before: dict[str, torch.Tensor], after: torch.nn.Module
) -> dict[str, list[torch.Tensor]]:
    """Group exact E489 Muon matrices and remaining AdamW parameters."""
    parameters = dict(after.named_parameters())
    if set(parameters) != set(before):
        raise ValueError("E489 parameter names changed during diagnostic update")
    groups = {"matrix:" + name: [name] for name in sorted(actor.MUON_NAMES)}
    groups["adamw"] = sorted(set(parameters) - actor.MUON_NAMES)
    if {name for names in groups.values() for name in names} != set(parameters):
        raise ValueError("E489 update groups do not cover every parameter")
    return {
        group: [
            parameters[name].detach() - before[name].to(parameters[name].device) for name in names
        ]
        for group, names in groups.items()
    }


def _gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
    }
    if len(gradients) != len(list(model.parameters())):
        raise ValueError("an E489 parameter did not receive a gradient")
    for gradient in gradients.values():
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite E489 parameter gradient")
    return gradients


def _make_updated_model(
    implementation: str,
    mode: str,
    initial: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, float | bool]]:
    if implementation == "native":
        model = RNNModel(actor.FEATURES, hidden=actor.HIDDEN, layers=2, dropout=0.0, n_out=2)
        model.load_state_dict(initial, strict=True)
    elif implementation == "custom":
        model = actor.initialize_arm(0.0, initial)
    else:
        raise ValueError(f"unknown implementation: {implementation}")
    model = model.to(device).float().train()
    if model.state_dict().keys() != initial.keys():
        raise ValueError("E489 model state keys differ from frozen initialization")
    named_parameters = dict(model.named_parameters())
    for name, parameter in named_parameters.items():
        if name not in gradients:
            raise ValueError(f"missing frozen raw gradient for {name}")
        parameter.grad = gradients[name].to(device=device).clone()
    norm = raw_global_norm({name: value.grad for name, value in named_parameters.items()})
    clip = clipping_record(norm)
    before = {name: parameter.detach().clone() for name, parameter in named_parameters.items()}
    optimizer = e487.make_optimizer(model, precision="fp32_diagnostic")
    if mode == "clip1":
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    elif mode != "raw":
        raise ValueError(f"unknown update mode: {mode}")
    optimizer.step()
    e470.ensure_finite_parameters(model, f"E489 diagnostic/{implementation}/{mode}")
    e487.require_optimizer_state_finite(optimizer, f"E489 diagnostic/{implementation}/{mode}")
    return grouped_update_deltas(before, model), clip


def diagnose_seed(
    seed: int,
    initial: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    need: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare recurrence, gradients and one-step updates for one frozen seed."""
    native = RNNModel(actor.FEATURES, hidden=actor.HIDDEN, layers=2, dropout=0.0, n_out=2)
    native.load_state_dict(initial, strict=True)
    native = native.to(device).float().train()
    custom = actor.initialize_arm(0.0, initial).to(device).float().train()
    native_x = x.detach().clone().requires_grad_(True)
    custom_x = x.detach().clone().requires_grad_(True)
    native_prediction, native_state = native(native_x)
    custom_prediction, custom_state, custom_hidden = custom.forward_details(custom_x)
    if not all(
        torch.isfinite(item).all()
        for item in (
            native_prediction,
            native_state,
            custom_prediction,
            custom_state,
            custom_hidden,
        )
    ):
        raise ValueError(f"nonfinite E489 forward at seed {seed}")
    native_loss = e470.loss_for_batch(native_prediction, y, need)
    custom_loss = e470.loss_for_batch(custom_prediction, y, need)
    native_loss.backward()
    custom_loss.backward()
    native_grads = _gradients(native)
    custom_grads = _gradients(custom)
    native_input_grad = native_x.grad.detach().clone()
    custom_input_grad = custom_x.grad.detach().clone()
    if not (torch.isfinite(native_input_grad).all() and torch.isfinite(custom_input_grad).all()):
        raise ValueError(f"nonfinite E489 input gradient at seed {seed}")

    gradient_comparisons = {
        name: compare_tensors(
            native_grads[name], custom_grads[name], atol=GRAD_ATOL, rtol=GRAD_RTOL
        )
        for name in sorted(native_grads)
    }
    native_clip = clipping_record(raw_global_norm(native_grads))
    custom_clip = clipping_record(raw_global_norm(custom_grads))

    updates: dict[str, dict[str, list[torch.Tensor]]] = {}
    update_clipping: dict[str, dict[str, float | bool]] = {}
    for implementation, gradients in (("native", native_grads), ("custom", custom_grads)):
        for mode in ("raw", "clip1"):
            delta, clip = _make_updated_model(implementation, mode, initial, gradients, device)
            key = f"{implementation}_{mode}"
            updates[key] = delta
            update_clipping[key] = clip

    report: dict[str, object] = {
        "seed": seed,
        "fit_only": True,
        "scoring_performed": False,
        "native_custom_output": compare_tensors(
            native_prediction, custom_prediction, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
        ),
        "native_custom_state": compare_tensors(
            native_state, custom_state, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
        ),
        "native_custom_input_gradient": compare_tensors(
            native_input_grad, custom_input_grad, atol=GRAD_ATOL, rtol=GRAD_RTOL
        ),
        "native_custom_parameter_gradients": gradient_comparisons,
        "native_preclip": native_clip,
        "custom_preclip": custom_clip,
        "original_clipping_coverage_gate_passed": bool(
            native_clip["clipping_active"] and custom_clip["clipping_active"]
        ),
        "native_custom_raw_update_deltas": compare_delta_groups(
            updates["native_raw"], updates["custom_raw"]
        ),
        "native_custom_clip1_update_deltas": compare_delta_groups(
            updates["native_clip1"], updates["custom_clip1"]
        ),
        "native_raw_vs_clip1_deltas": compare_delta_groups(
            updates["native_raw"], updates["native_clip1"]
        ),
        "custom_raw_vs_clip1_deltas": compare_delta_groups(
            updates["custom_raw"], updates["custom_clip1"]
        ),
        "update_branch_preclip": update_clipping,
        "interpretation": "diagnostic only; cannot pass E489 or admit full fits",
    }
    arrays: dict[str, np.ndarray] = {
        "input_gradient/native": native_input_grad.detach().cpu().numpy(),
        "input_gradient/custom": custom_input_grad.detach().cpu().numpy(),
    }
    arrays.update(
        {
            f"parameter_gradient/native/{name}": value.detach().cpu().numpy()
            for name, value in native_grads.items()
        }
    )
    arrays.update(
        {
            f"parameter_gradient/custom/{name}": value.detach().cpu().numpy()
            for name, value in custom_grads.items()
        }
    )
    return report, arrays
