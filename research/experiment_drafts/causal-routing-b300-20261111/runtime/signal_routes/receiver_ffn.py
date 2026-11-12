"""Server-only synthetic checks for the matched static and conditional FFN routers."""

from __future__ import annotations

import torch

from signal_routes.ffn_block import route_feedforward_activations, router_weights
from signal_routes.receiver_support import synthetic_tensor


def check_router_contract(device: torch.device, seed: int) -> dict[str, float | bool]:
    """Check uniform initialization, static input independence and conditional gradients."""
    hidden = 48
    values = synthetic_tensor((3, 5, hidden), device, seed)
    matrix = torch.zeros((2, hidden), dtype=torch.float32, device=device)
    bias = torch.zeros(2, dtype=torch.float32, device=device)
    static_weights = router_weights(values, matrix, bias, "ffn_static")
    conditional_initial = router_weights(values, matrix, bias, "ffn_conditional")
    uniform = torch.full_like(static_weights, 0.5)
    if not torch.equal(static_weights, uniform) or not torch.equal(conditional_initial, uniform):
        raise AssertionError("zero-initialized FFN routers must start at an exact uniform mix")
    changed_static = router_weights(values * 3.0 + 2.0, matrix, bias, "ffn_static")
    if not torch.equal(static_weights, changed_static):
        raise AssertionError("static FFN routing depends on token inputs")

    activations = synthetic_tensor((3, 5, 24), device, seed + 1)
    routed = route_feedforward_activations(activations, uniform)
    if not torch.equal(routed, activations):
        raise AssertionError("uniform FFN routing must reproduce canonical activation channels")

    conditional_input = values.detach().clone().requires_grad_(True)
    conditional_matrix = torch.nn.Parameter(torch.zeros((2, hidden), device=device))
    conditional_bias = torch.nn.Parameter(torch.zeros(2, device=device))
    with torch.no_grad():
        conditional_matrix[0, 0] = 1.0
        conditional_matrix[1, 1] = -1.0
    conditional_weights = router_weights(
        conditional_input, conditional_matrix, conditional_bias, "ffn_conditional"
    )
    if float(conditional_weights[..., 0].std().item()) <= 1e-5:
        raise AssertionError("conditional router did not vary across token inputs")
    (conditional_weights[..., 0].sum()).backward()
    if conditional_input.grad is None or float(conditional_input.grad.abs().sum().item()) <= 0:
        raise AssertionError("conditional routing has no gradient to the token representation")
    if conditional_matrix.grad is None or not bool(torch.isfinite(conditional_matrix.grad).all()):
        raise AssertionError("conditional routing matrix gradient is missing or nonfinite")
    if float(conditional_matrix.grad.abs().sum().item()) <= 0:
        raise AssertionError("conditional routing matrix has a zero gradient")

    static_matrix = torch.nn.Parameter(torch.zeros((2, hidden), device=device))
    static_bias = torch.nn.Parameter(torch.zeros(2, device=device))
    static_loss = router_weights(values, static_matrix, static_bias, "ffn_static")[..., 0].sum()
    static_loss.backward()
    if static_matrix.grad is None or bool(static_matrix.grad.count_nonzero()):
        raise AssertionError("static router matrix must receive only a zero input gradient")
    if static_bias.grad is None or float(static_bias.grad.abs().sum().item()) <= 0:
        raise AssertionError("static router bias must remain trainable")
    return {
        "initial_uniform_parity": True,
        "static_input_independence": True,
        "conditional_gate_std": float(conditional_weights[..., 0].std().item()),
        "conditional_input_gradient_l1": float(conditional_input.grad.abs().sum().item()),
        "conditional_matrix_gradient_l1": float(conditional_matrix.grad.abs().sum().item()),
        "static_matrix_zero_gradient": True,
        "static_bias_trainable": True,
    }
