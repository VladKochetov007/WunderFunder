"""Formula, numerical-stability and initialization tests for E473."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))

import fit_activation_optimizer as e470  # noqa: E402
from fit_activation_mechanism import (  # noqa: E402
    ACTIVATIONS,
    ARMS,
    MechanismActivation,
    MechanismGRU,
)


def check_formulas_and_zero_derivatives() -> None:
    """Check each formula, value at zero and local derivative at zero."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    formulas = {
        "gelu": F.gelu(x),
        "gelu_plus_sine_01": F.gelu(x) + 0.1 * torch.sin(x),
        "gelu_plus_linear_01": F.gelu(x) + 0.1 * x,
        "scaled_gelu_11": 1.1 * F.gelu(x),
        "jax_tanh_gelu_plus_sine_01": F.gelu(x, approximate="tanh") + 0.1 * torch.sin(x),
    }
    for name, expected in formulas.items():
        torch.testing.assert_close(MechanismActivation(name)(x), expected, atol=0, rtol=0)

    zero_derivatives = {
        "gelu": 0.5,
        "gelu_plus_sine_01": 0.6,
        "gelu_plus_linear_01": 0.6,
        "scaled_gelu_11": 0.55,
        "jax_tanh_gelu_plus_sine_01": 0.6,
    }
    zero = torch.zeros((), dtype=torch.float64, requires_grad=True)
    for name, expected_derivative in zero_derivatives.items():
        output = MechanismActivation(name)(zero)
        derivative = torch.autograd.grad(output, zero, retain_graph=True)[0]
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        torch.testing.assert_close(
            derivative, torch.tensor(expected_derivative, dtype=zero.dtype), atol=1e-12, rtol=0
        )

    moderate = x
    jax_tanh_formula = (
        0.5
        * moderate
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (moderate + 0.044715 * moderate.pow(3))))
    )
    expected = jax_tanh_formula + 0.1 * torch.sin(moderate)
    torch.testing.assert_close(
        MechanismActivation("jax_tanh_gelu_plus_sine_01")(moderate),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def check_extreme_finite_values_and_gradients() -> None:
    """Require finite formulas/derivatives over saturated positive and negative inputs."""
    values = torch.tensor(
        [-1000.0, -100.0, -20.0, -8.0, 0.0, 8.0, 20.0, 100.0, 1000.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    for name in ACTIVATIONS:
        values.grad = None
        output = MechanismActivation(name)(values)
        if not torch.isfinite(output).all():
            raise AssertionError(f"nonfinite output for {name}")
        output.sum().backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise AssertionError(f"nonfinite derivative for {name}")


def check_model_initialization_and_baseline_parity() -> None:
    """Preserve E470 parameter initialization/layout and exact GELU control."""
    torch.manual_seed(271828)
    reference = e470.ActivationGRU("gelu")
    torch.manual_seed(271828)
    baseline = MechanismGRU("gelu")
    reference_parameters = dict(reference.named_parameters())
    baseline_parameters = dict(baseline.named_parameters())
    if reference_parameters.keys() != baseline_parameters.keys():
        raise AssertionError("E473/E470 parameter names diverged")
    for name, parameter in reference_parameters.items():
        if not torch.equal(parameter, baseline_parameters[name]):
            raise AssertionError(f"E473 changed seeded E470 initialization at {name}")

    torch.manual_seed(141)
    x = torch.randn(4, 32, 42)
    prediction_reference, state_reference = reference(x)
    prediction_baseline, state_baseline = baseline(x)
    torch.testing.assert_close(prediction_baseline, prediction_reference, atol=0, rtol=0)
    torch.testing.assert_close(state_baseline, state_reference, atol=0, rtol=0)
    if torch.count_nonzero(baseline.correction_output.weight) != 0:
        raise AssertionError("E470 zero-init correction anchor was not preserved")
    if torch.count_nonzero(baseline.correction_output.bias) != 0:
        raise AssertionError("E470 zero-init correction bias was not preserved")

    for activation in ACTIVATIONS:
        torch.manual_seed(271828)
        model = MechanismGRU(activation)
        if dict(model.named_parameters()).keys() != reference_parameters.keys():
            raise AssertionError(f"parameter names changed for {activation}")
        for name, parameter in model.named_parameters():
            if not torch.equal(parameter, reference_parameters[name]):
                raise AssertionError(f"initialization changed for {activation}/{name}")
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != 8235:
            raise AssertionError(f"unexpected parameter count for {activation}: {count}")


def check_arm_matrix() -> None:
    """Ensure the planned 5x2 factorial contains exactly ten unique arms."""
    if len(ARMS) != 10 or len(set(ARMS)) != 10:
        raise AssertionError("E473 must contain ten unique activation/optimizer arms")


def main() -> None:
    """Run formula and matched-model CPU checks."""
    torch.set_num_threads(1)
    check_formulas_and_zero_derivatives()
    check_extreme_finite_values_and_gradients()
    check_model_initialization_and_baseline_parity()
    check_arm_matrix()
    print("E473 activation-mechanism CPU tests passed")


if __name__ == "__main__":
    main()
