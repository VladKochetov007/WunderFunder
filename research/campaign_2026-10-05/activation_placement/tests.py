"""CPU invariants for E474 activation placement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from fit_activation_placement import (  # noqa: E402
    ACTIVATION_FORMULAS,
    EXPECTED_PARAMETERS,
    MODELS,
    SEEDS,
    PlacementActivation,
    initialize_arm,
    make_seed_states,
)

from wnn.nn import RNNModel  # noqa: E402


def check_initialization_and_parameter_count() -> None:
    """All variants must be exact activation-only changes to one baseline."""
    states = make_seed_states()
    for seed in SEEDS:
        baseline = initialize_arm(seed, "gelu", "gelu", states[seed])
        reference = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2)
        reference.load_state_dict(states[seed], strict=True)
        reference.eval()
        baseline.eval()
        if sum(p.numel() for p in reference.parameters()) != EXPECTED_PARAMETERS:
            raise AssertionError("live RNNModel parameter count changed")
        if baseline.gru.dropout != 0.0:
            raise AssertionError("recurrent dropout must remain disabled")
        for name, embedding, head in MODELS:
            arm = initialize_arm(seed, embedding, head, states[seed])
            if sum(p.numel() for p in arm.parameters()) != EXPECTED_PARAMETERS:
                raise AssertionError(f"parameter count changed for {name}")
            if tuple(arm.state_dict()) != tuple(reference.state_dict()):
                raise AssertionError(f"state keys changed for {name}")
            for key, value in arm.state_dict().items():
                if not torch.equal(value, reference.state_dict()[key]):
                    raise AssertionError(f"initial value differs at {name}/{key}")
            x = torch.randn(2, 19, 42)
            actual, actual_state = arm(x)
            if name == "base":
                expected, expected_state = reference(x)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def check_activation_formulas_and_gradients() -> None:
    """Verify exact pointwise formulas and finite derivatives at stress values."""
    x = torch.tensor([-1000.0, -3.0, -1e-6, 0.0, 1e-6, 3.0, 1000.0], requires_grad=True)
    gelu = torch.nn.functional.gelu(x, approximate="none")
    for name, formula in ACTIVATION_FORMULAS.items():
        activation = PlacementActivation(name)
        if sum(parameter.numel() for parameter in activation.parameters()) != 0:
            raise AssertionError("placement activation must be parameter-free")
        actual = activation(x)
        if name == "gelu":
            expected = gelu
        elif name == "sine":
            expected = gelu + 0.1 * torch.sin(x)
        else:
            expected = gelu + 0.1 * x
        if actual.shape != x.shape or not torch.isfinite(actual).all():
            raise AssertionError(f"nonfinite output for {formula}")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        gradient = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        if not torch.isfinite(gradient).all():
            raise AssertionError(f"nonfinite derivative for {formula}")
        if name in {"sine", "slope"} and not torch.isclose(
            gradient[3], torch.tensor(0.6), atol=1e-6, rtol=0
        ):
            raise AssertionError(f"matched derivative at zero changed for {name}")


def check_causality_chunking_and_reset() -> None:
    """Verify ordinary-GRU state carry, future isolation, and zero-state reset."""
    torch.manual_seed(73)
    x = torch.randn(2, 41, 42)
    perturbed = x.clone()
    perturbed[:, 17:] += 5.0
    states = make_seed_states()
    for seed in SEEDS:
        for name, embedding, head in MODELS:
            model = initialize_arm(seed, embedding, head, states[seed]).eval()
            full, final_state = model(x)
            changed, _ = model(perturbed)
            if not torch.equal(full[:, :17], changed[:, :17]):
                raise AssertionError(f"future perturbation leaked for {name}")
            prefix, state = model(x[:, :17])
            suffix, carried_state = model(x[:, 17:], state)
            torch.testing.assert_close(
                torch.cat((prefix, suffix), dim=1), full, atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(carried_state, final_state, atol=1e-6, rtol=1e-6)
            reset_output, reset_state = model(x, None)
            torch.testing.assert_close(reset_output, full, atol=0, rtol=0)
            torch.testing.assert_close(reset_state, final_state, atol=0, rtol=0)


def main() -> None:
    check_initialization_and_parameter_count()
    check_activation_formulas_and_gradients()
    check_causality_chunking_and_reset()
    print("E474 activation placement CPU invariants passed")


if __name__ == "__main__":
    main()
