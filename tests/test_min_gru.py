from __future__ import annotations

import copy

import pytest
import torch

from wnn.min_gru import MinGRULayer, MinGRUModel


def serial(layer: MinGRULayer, inputs: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    states = []
    hidden = initial
    for row in inputs.unbind(dim=1):
        hidden = layer.step(row, hidden)
        states.append(hidden)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("length", [1, 2, 7, 32, 37])
def test_parallel_matches_serial_and_actual_initial_state(length: int) -> None:
    torch.manual_seed(753)
    layer = MinGRULayer(3, 5).double()
    inputs = torch.randn(2, length, 3, dtype=torch.float64)
    initial = torch.randn(2, 5, dtype=torch.float64)
    actual, final = layer(inputs, initial)
    expected = serial(layer, inputs, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(final, expected[:, -1], rtol=1e-13, atol=1e-13)


def test_gradients_match_serial_including_exact_zero_initial_state() -> None:
    torch.manual_seed(579)
    parallel = MinGRULayer(3, 4).double()
    sequential = copy.deepcopy(parallel)
    inputs = torch.randn(2, 19, 3, dtype=torch.float64, requires_grad=True)
    serial_inputs = inputs.detach().clone().requires_grad_()
    initial = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    serial_initial = initial.detach().clone().requires_grad_()
    weights = torch.randn(2, 19, 4, dtype=torch.float64)
    (parallel(inputs, initial)[0] * weights).sum().backward()
    (serial(sequential, serial_inputs, serial_initial) * weights).sum().backward()
    for actual, expected in [
        (inputs.grad, serial_inputs.grad),
        (initial.grad, serial_initial.grad),
        *[
            (p.grad, q.grad)
            for p, q in zip(parallel.parameters(), sequential.parameters(), strict=True)
        ],
    ]:
        assert actual is not None and expected is not None
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(initial.grad) > 0


def test_zero_is_zero_not_transformed_half_state() -> None:
    layer = MinGRULayer(1, 1)
    with torch.no_grad():
        layer.input_projection.weight.zero_()
        layer.input_projection.bias.zero_()
    outputs, final = layer(torch.zeros(1, 3, 1))
    # z=0.5, candidate=g(0)=0.5, h0=0: h1=.25, h2=.375, h3=.4375.
    assert torch.equal(outputs, torch.tensor([[[0.25], [0.375], [0.4375]]]))
    assert torch.equal(final, torch.tensor([[0.4375]]))


def test_chunk_causality_reset_and_twenty_thousand_rows() -> None:
    torch.manual_seed(951)
    layer = MinGRULayer(3, 5)
    inputs = torch.randn(1, 20_000, 3)
    with torch.inference_mode():
        full, final = layer(inputs)
        chunks = []
        state = None
        for chunk in inputs.split(997, dim=1):
            output, state = layer(chunk, state)
            chunks.append(output)
        torch.testing.assert_close(full, torch.cat(chunks, dim=1), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(final, state, rtol=2e-5, atol=2e-6)
        prefix, _ = layer(inputs[:, :99])
        torch.testing.assert_close(prefix, full[:, :99], rtol=2e-5, atol=2e-6)
        replay, _ = layer(inputs)
        assert torch.equal(full, replay)
        assert torch.isfinite(full).all()
        assert full.min() >= 0


def test_empty_chunk_preserves_state_and_state_shape_is_checked() -> None:
    layer = MinGRULayer(3, 5)
    initial = torch.ones(2, 5)
    output, hidden = layer(torch.empty(2, 0, 3), initial)
    assert output.shape == (2, 0, 5)
    assert hidden is initial
    with pytest.raises(ValueError, match="state"):
        layer(torch.ones(2, 3, 3), torch.ones(2, 4))


def test_full_model_matches_row_callback_and_carries_every_layer() -> None:
    torch.manual_seed(819)
    model = MinGRUModel(
        211, layers=3, encoder_width=12, recurrent_width=8, readout_width=6
    ).double()
    with torch.no_grad():
        model.mean.copy_(torch.randn(211, dtype=torch.float64))
        model.std.copy_(torch.rand(211, dtype=torch.float64) + 0.5)
    inputs = torch.randn(2, 101, 211, dtype=torch.float64)
    initial = torch.randn(3, 2, 8, dtype=torch.float64)
    with torch.inference_mode():
        full, final = model(inputs, initial)
        state = initial
        outputs = []
        for row in inputs.unbind(dim=1):
            projection = torch.nn.functional.gelu(model.inp((row - model.mean) / model.std))
            value = projection
            next_states = []
            for index, cell in enumerate(model.cells):
                value = cell.step(value, state[index])
                next_states.append(value)
            state = torch.stack(next_states)
            outputs.append(model.head(torch.cat((value, projection), dim=-1)))
        torch.testing.assert_close(full, torch.stack(outputs, dim=1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(final, state, rtol=1e-12, atol=1e-12)
        first, carry = model(inputs[:, :99], initial)
        last, chunk_final = model(inputs[:, 99:], carry)
        torch.testing.assert_close(full, torch.cat((first, last), dim=1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(final, chunk_final, rtol=1e-12, atol=1e-12)
        reset, _ = model(inputs)
        replay, _ = model(inputs, torch.zeros_like(initial))
        assert torch.equal(reset, replay)


def test_full_model_has_finite_trainable_gradients_and_expected_matrix_count() -> None:
    torch.manual_seed(651)
    model = MinGRUModel(211, hidden=8, layers=2)
    inputs = torch.randn(2, 17, 211)
    prediction, _ = model(inputs)
    prediction.square().sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    matrix_count = sum(parameter.numel() for parameter in model.parameters() if parameter.ndim == 2)
    assert matrix_count == (2 * 2 + 2) * 8**2 + 213 * 8
