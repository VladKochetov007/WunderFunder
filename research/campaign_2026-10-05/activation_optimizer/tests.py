"""CPU correctness checks for the bounded activation/optimizer screen."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_activation_optimizer import (  # noqa: E402
    ACTIVATIONS,
    ARMS,
    ActivationGRU,
    PointwiseActivation,
    clip_finite_labels,
    ensure_finite_prediction,
    make_optimizer,
    weighted_pearson,
)


def check_activations() -> None:
    values = torch.tensor(
        [[[-1000.0, 0.0, 1000.0, -0.25], [0.1, -2.0, 3.0, 8.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    for name in ACTIVATIONS:
        activation = PointwiseActivation(name)
        output = activation(values)
        if output.shape != values.shape or not torch.isfinite(output).all():
            raise AssertionError(f"invalid large-input output for {name}")
        output.square().mean().backward(retain_graph=True)
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise AssertionError(f"invalid large-input gradient for {name}")
        values.grad.zero_()
    x = torch.tensor([-1000.0, 0.0, 1000.0])
    actual = PointwiseActivation("gelu_sinc_05")(x)
    expected = torch.nn.functional.gelu(x) * (1.0 + 0.5 * torch.sinc(x))
    torch.testing.assert_close(actual, expected)
    zero = PointwiseActivation("gelu_sine_cosine_01")(torch.tensor([0.0]))
    torch.testing.assert_close(zero, torch.zeros_like(zero))


def check_nonfinite_data_guards() -> None:
    """Invalid raw values must fail before clipping can hide infinities."""
    finite = torch.tensor([0.0, 1.0]).numpy()
    for invalid in (float("inf"), float("-inf"), float("nan")):
        raw = finite.copy()
        raw[1] = invalid
        try:
            weighted_pearson(finite, raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"metric prediction accepted raw {invalid}")
        try:
            weighted_pearson(raw, finite)
        except ValueError:
            pass
        else:
            raise AssertionError(f"metric target accepted raw {invalid}")
        try:
            clip_finite_labels(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"label loader accepted raw {invalid}")
        try:
            ensure_finite_prediction(raw, "fixture")
        except ValueError:
            pass
        else:
            raise AssertionError(f"prediction guard accepted raw {invalid}")
    clipped = clip_finite_labels(np.array([-20.0, 0.0, 20.0]))
    if not (clipped == [-2.0, 0.0, 2.0]).all():
        raise AssertionError("finite label clipping recipe changed")


def check_causality_and_reset() -> None:
    torch.manual_seed(41)
    x = torch.randn(2, 37, 42)
    changed = x.clone()
    changed[:, 13:] += 8.0
    for activation in ACTIVATIONS:
        model = ActivationGRU(activation).eval()
        full, full_state = model(x)
        altered, _ = model(changed)
        if not torch.equal(full[:, :13], altered[:, :13]):
            raise AssertionError(f"future perturbation changed prefix for {activation}")
        left, state = model(x[:, :13])
        right, reset_state = model(x[:, 13:], state)
        torch.testing.assert_close(torch.cat((left, right), dim=1), full, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(reset_state, full_state, atol=1e-6, rtol=1e-6)
        reset_output, reset_state = model(x, None)
        torch.testing.assert_close(reset_output, full, atol=0, rtol=0)
        torch.testing.assert_close(reset_state, full_state, atol=0, rtol=0)
        if (
            model.correction_output.weight.count_nonzero()
            or model.correction_output.bias.count_nonzero()
        ):
            raise AssertionError(f"residual head is not zero-initialized for {activation}")
        encoded = model.embedding_activation(model.embedding(x))
        memory, _ = model.gru(encoded)
        torch.testing.assert_close(full, model.anchor(memory), atol=0, rtol=0)


def check_optimizer_partition_and_updates() -> None:
    torch.manual_seed(271828)
    x = torch.randn(4, 32, 42)
    target = torch.randn(4, 32, 2).clamp(-2, 2)
    need = torch.ones(4, 32, dtype=torch.bool)
    for activation in ACTIVATIONS:
        torch.manual_seed(271828)
        adam_model = ActivationGRU(activation)
        muon_model = copy.deepcopy(adam_model)
        adam = make_optimizer(adam_model, "adamw")
        muon = make_optimizer(muon_model, "muon_adamw")
        for model, optimizer in ((adam_model, adam), (muon_model, muon)):
            prediction, _ = model(x)
            weights = target.abs() * need[..., None]
            loss = (weights * (prediction - target).square()).sum() / weights.sum().clamp_min(1e-12)
            loss.backward()
            if not torch.isfinite(loss):
                raise AssertionError(f"nonfinite {activation} loss")
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"nonfinite {activation} gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise AssertionError(f"nonfinite updated parameters for {activation}")
        if not any(
            not torch.equal(a, b)
            for a, b in zip(adam_model.parameters(), muon_model.parameters(), strict=True)
        ):
            raise AssertionError(f"optimizer choices made identical updates for {activation}")


def check_long_stream_gradients() -> None:
    """Exercise all pointwise choices over 20,000 rows with detached 256-row chunks."""
    torch.manual_seed(2026)
    x = torch.randn(1, 20_000, 42)
    for activation in ACTIVATIONS:
        model = ActivationGRU(activation)
        state = None
        for start in range(0, len(x[0]), 256):
            output, state = model(x[:, start : start + 256], state)
            loss = output.square().mean()
            loss.backward()
            state = state.detach()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise AssertionError(f"nonfinite long-stream gradient for {activation}")


def main() -> None:
    torch.set_num_threads(1)
    if len(ARMS) != 16:
        raise AssertionError("expected all sixteen activation/optimizer combinations")
    if (
        len(
            {
                sum(parameter.numel() for parameter in ActivationGRU(name).parameters())
                for name in ACTIVATIONS
            }
        )
        != 1
    ):
        raise AssertionError("activation choices changed the model parameter count")
    check_activations()
    check_nonfinite_data_guards()
    check_causality_and_reset()
    check_optimizer_partition_and_updates()
    check_long_stream_gradients()
    print("activation/optimizer CPU tests passed")


if __name__ == "__main__":
    main()
