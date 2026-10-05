"""CPU invariants for the E477 readout-mixture/dropout screen."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(ROOT / "src"))

import fit_activation_placement as e474  # noqa: E402
from fit_activation_head_mixture import (  # noqa: E402
    ACTIVATIONS,
    DROPOUTS,
    EXPECTED_PARAMETERS,
    SEEDS,
    ReadoutActivationDropout,
    initialize_arm,
    make_initial_states,
)


def check_parameter_identity_and_baseline_parity() -> None:
    """Each arm clones one baseline, and GELU/p=0 matches E474 exactly."""
    states = make_initial_states()
    for seed in SEEDS:
        reference = e474.initialize_arm(seed, "gelu", "gelu", states[seed])
        reference.eval()
        for activation in ACTIVATIONS:
            for dropout in DROPOUTS:
                model = initialize_arm(activation, dropout, states[seed]).eval()
                if (
                    sum(parameter.numel() for parameter in model.parameters())
                    != EXPECTED_PARAMETERS
                ):
                    raise AssertionError("trainable parameter count changed")
                if model.dropout_probability != dropout:
                    raise AssertionError("head dropout probability metadata mismatch")
                if tuple(model.state_dict()) != tuple(reference.state_dict()):
                    raise AssertionError("model state keys changed")
                for key, tensor in model.state_dict().items():
                    if not torch.equal(tensor, reference.state_dict()[key]):
                        raise AssertionError(f"initial parameter/buffer differs: {key}")
                if activation == "gelu" and dropout == 0.0:
                    x = torch.randn(2, 23, 42)
                    actual, state = model(x)
                    expected, expected_state = reference(x)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
                    actual.square().mean().backward()
                    expected.square().mean().backward()
                    for (_, left), (_, right) in zip(
                        model.named_parameters(), reference.named_parameters(), strict=True
                    ):
                        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def check_activation_values_and_finite_output() -> None:
    """Check channel split and same-channel mixture formulas at large values."""
    x = torch.tensor(
        [[-1000.0, -3.0, -1.0, 0.0, 0.5, 2.0, 8.0, 1000.0]],
        requires_grad=True,
    )
    for activation in ACTIVATIONS:
        layer = ReadoutActivationDropout(activation, 0.0, hidden=8).eval()
        output = layer(x)
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise AssertionError(f"invalid readout values for {activation}")
        output.sum().backward(retain_graph=True)
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise AssertionError(f"invalid readout derivatives for {activation}")
        x.grad.zero_()
    gelu = torch.nn.functional.gelu(x, approximate="none")
    leaky = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    split = torch.cat((gelu[..., :4], leaky[..., 4:]), dim=-1)
    mixed = 0.5 * gelu + 0.5 * leaky
    torch.testing.assert_close(ReadoutActivationDropout("channel_split", 0.0, 8)(x), split)
    torch.testing.assert_close(ReadoutActivationDropout("same_channel_mix", 0.0, 8)(x), mixed)


def check_dropout_and_causal_chunk_state() -> None:
    """Dropout is train-only; evaluation remains causal and chunk-streamable."""
    states = make_initial_states()
    torch.manual_seed(91)
    x = torch.randn(2, 47, 42)
    x[:, 31:] += 1000.0
    future = x.clone()
    future[:, 19:] -= 23.0
    for seed in SEEDS:
        for activation in ACTIVATIONS:
            for dropout in DROPOUTS:
                model = initialize_arm(activation, dropout, states[seed]).eval()
                full, final_state = model(x)
                altered, _ = model(future)
                if not torch.equal(full[:, :19], altered[:, :19]):
                    raise AssertionError(f"future values leaked for {activation}/{dropout}")
                prefix, state = model(x[:, :19])
                suffix, carried = model(x[:, 19:], state)
                torch.testing.assert_close(
                    torch.cat((prefix, suffix), dim=1), full, atol=1e-5, rtol=1e-5
                )
                torch.testing.assert_close(carried, final_state, atol=1e-5, rtol=1e-5)
                reset, reset_state = model(x, None)
                torch.testing.assert_close(reset, full, atol=0, rtol=0)
                torch.testing.assert_close(reset_state, final_state, atol=0, rtol=0)
                repeated, _ = model(x)
                torch.testing.assert_close(repeated, full, atol=0, rtol=0)
    dropout_model = initialize_arm("same_channel_mix", 0.1, states[SEEDS[0]]).train()
    dropout_input = torch.ones(4, 16, 42)
    first, _ = dropout_model(dropout_input)
    second, _ = dropout_model(dropout_input)
    if torch.equal(first, second):
        raise AssertionError("head dropout p=0.1 appears inactive during training")


def main() -> None:
    check_parameter_identity_and_baseline_parity()
    check_activation_values_and_finite_output()
    check_dropout_and_causal_chunk_state()
    print("E477 activation-head mixture CPU invariants passed")


if __name__ == "__main__":
    main()
