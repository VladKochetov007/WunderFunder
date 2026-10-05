"""Small deterministic CPU checks for the isolated sequence components."""

from __future__ import annotations

import torch
from components import AdditiveSpline, DiagonalSSM, KernelAttention, PointwiseMLPControl


def _assert_finite_tree(value: object) -> None:
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, tuple):
        for item in value:
            _assert_finite_tree(item)
    else:
        raise AssertionError(f"unexpected state value: {type(value)}")


def test_streaming_parity_and_reset() -> None:
    """Chunked execution must match full execution and None must reset state."""
    torch.manual_seed(13)
    x = torch.randn(2, 37, 8)
    for model in (DiagonalSSM(8, 16), KernelAttention(8, 8, 8)):
        full, full_state = model(x)
        assert full.abs().max() <= 1.00001
        first, state = model(x[:, :11])
        second, state = model(x[:, 11:29], state)
        third, state = model(x[:, 29:], state)
        streamed = torch.cat((first, second, third), dim=1)
        assert torch.allclose(streamed, full, atol=2e-6, rtol=2e-6)
        _assert_finite_tree(state)
        _assert_finite_tree(full_state)
        reset, reset_state = model(x)
        assert torch.equal(reset, full)
        _assert_finite_tree(reset_state)


def test_future_perturbation_cannot_change_prefix() -> None:
    """Causal recurrent components must not read later positions."""
    torch.manual_seed(17)
    x = torch.randn(2, 19, 8)
    changed = x.clone()
    changed[:, 12:] += 20.0
    for model in (DiagonalSSM(8, 16), KernelAttention(8, 8, 8)):
        original, _ = model(x)
        perturbed, _ = model(changed)
        assert torch.equal(original[:, :12], perturbed[:, :12])


def test_spline_anchor_and_gradients() -> None:
    """Spline edges start at the zero residual anchor and receive gradients."""
    torch.manual_seed(23)
    spline = AdditiveSpline(8, 2, knots=7)
    x = torch.randn(3, 13, 8, requires_grad=True)
    output = spline(x)
    assert torch.equal(output, torch.zeros_like(output))
    (output + torch.randn_like(output)).square().mean().backward()
    assert spline.edge_values.grad is not None
    assert torch.isfinite(spline.edge_values.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_matched_mlp_control_budget() -> None:
    """The pointwise control uses a hidden width within the requested budget."""
    control = PointwiseMLPControl(8, 2, parameter_budget=400)
    count = sum(parameter.numel() for parameter in control.parameters())
    assert count <= 400
    assert control(torch.randn(4, 17, 8)).shape == (4, 17, 2)


def test_finite_gradients_and_state_over_20k_rows() -> None:
    """Streaming chunks keep values and truncated gradients finite over 20k rows."""
    torch.manual_seed(31)
    x = torch.randn(1, 20_000, 8)
    models = (DiagonalSSM(8, 16), KernelAttention(8, 8, 8))
    for model in models:
        state = None
        for start in range(0, x.shape[1], 500):
            chunk = x[:, start : start + 500].detach().requires_grad_(True)
            output, state = model(chunk, state)
            assert torch.isfinite(output).all()
            assert output.abs().max() <= 1.00001
            _assert_finite_tree(state)
            output.square().mean().backward()
            assert chunk.grad is not None and torch.isfinite(chunk.grad).all()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
            model.zero_grad(set_to_none=True)
            state = (
                tuple(item.detach() for item in state)
                if isinstance(state, tuple)
                else state.detach()
            )


def run() -> None:
    test_streaming_parity_and_reset()
    test_future_perturbation_cannot_change_prefix()
    test_spline_anchor_and_gradients()
    test_matched_mlp_control_budget()
    test_finite_gradients_and_state_over_20k_rows()
    print("sequence component checks passed")


if __name__ == "__main__":
    run()
