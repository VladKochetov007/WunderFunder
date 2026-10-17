from __future__ import annotations

import copy

import pytest
import torch

from wnn.causal_transformer import CausalTransformer, CausalTransformerState

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def test_sequence_reset_and_batch_independence() -> None:
    model = _model()
    inputs = torch.randn(2, 19, 5, device="cuda")
    with torch.no_grad():
        batched, _ = model(inputs)
        separate = torch.cat([model(inputs[index : index + 1])[0] for index in range(2)])
        _, state = model(inputs[:, :7])
        model(inputs[:, 7:], state)
        reset, _ = model(inputs, None)
    torch.testing.assert_close(batched, separate, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(reset, batched, rtol=0, atol=0)


def _model() -> CausalTransformer:
    torch.manual_seed(143)
    return (
        CausalTransformer(
            n_in=5,
            hidden=32,
            layers=2,
            heads=4,
            window=5,
            ffn_mult=2,
            n_out=2,
        )
        .cuda()
        .eval()
    )


def _run_chunks(
    model: CausalTransformer, inputs: torch.Tensor, sizes: list[int]
) -> tuple[torch.Tensor, CausalTransformerState]:
    state = None
    outputs = []
    start = 0
    for size in sizes:
        prediction, state = model(inputs[:, start : start + size], state)
        outputs.append(prediction)
        start += size
    assert start == inputs.shape[1]
    assert state is not None
    return torch.cat(outputs, dim=1), state


def test_full_step_and_ragged_chunk_parity_and_bounded_state() -> None:
    model = _model()
    inputs = torch.randn(2, 17, 5, device="cuda")

    full, full_state = model(inputs)
    stepped, step_state = _run_chunks(model, inputs, [1] * inputs.shape[1])
    ragged, ragged_state = _run_chunks(model, inputs, [2, 1, 6, 3, 5])

    torch.testing.assert_close(full, stepped, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(full, ragged, rtol=3e-5, atol=3e-5)
    assert full_state.position_offset == step_state.position_offset == 17
    assert ragged_state.position_offset == 17
    for full_key, step_key, ragged_key, full_value, step_value, ragged_value in zip(
        full_state.keys,
        step_state.keys,
        ragged_state.keys,
        full_state.values,
        step_state.values,
        ragged_state.values,
        strict=True,
    ):
        assert full_key.shape[-2] == step_key.shape[-2] == ragged_key.shape[-2] == 4
        torch.testing.assert_close(full_key, step_key, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(full_key, ragged_key, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(full_value, step_value, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(full_value, ragged_value, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("window", [256, 2048])
def test_production_windows_keep_exact_history_and_chunk_parity(window: int) -> None:
    torch.manual_seed(319)
    model = (
        CausalTransformer(
            n_in=3,
            hidden=16,
            layers=1,
            heads=4,
            window=window,
            ffn_mult=2,
        )
        .cuda()
        .eval()
    )
    inputs = torch.randn(1, window + 3, 3, device="cuda")
    first = window // 2
    second = window // 2 + 1

    full, full_state = model(inputs)
    ragged, ragged_state = _run_chunks(model, inputs, [first, second, 2])

    torch.testing.assert_close(full, ragged, rtol=4e-5, atol=4e-5)
    assert full_state.position_offset == ragged_state.position_offset == window + 3
    for full_key, ragged_key, full_value, ragged_value in zip(
        full_state.keys, ragged_state.keys, full_state.values, ragged_state.values, strict=True
    ):
        assert full_key.shape[-2] == ragged_key.shape[-2] == window - 1
        torch.testing.assert_close(full_key, ragged_key, rtol=4e-5, atol=4e-5)
        torch.testing.assert_close(full_value, ragged_value, rtol=4e-5, atol=4e-5)


def test_future_rows_do_not_change_earlier_predictions() -> None:
    model = _model()
    inputs = torch.randn(2, 17, 5, device="cuda")
    changed = inputs.clone()
    changed[:, 9:] += 100 * torch.randn_like(changed[:, 9:])

    original_prediction, _ = model(inputs)
    changed_prediction, _ = model(changed)

    torch.testing.assert_close(
        original_prediction[:, :9], changed_prediction[:, :9], rtol=0, atol=0
    )


def test_state_gradients_and_explicit_tbptt_detach() -> None:
    model = _model()
    first = torch.randn(2, 5, 5, device="cuda", requires_grad=True)
    second = torch.randn(2, 4, 5, device="cuda", requires_grad=True)

    _, attached_state = model(first)
    prediction, _ = model(second, attached_state)
    prediction.square().sum().backward()

    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
    assert first.grad.abs().sum() > 0
    assert second.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    first_detached = torch.randn(2, 5, 5, device="cuda", requires_grad=True)
    second_current = torch.randn(2, 4, 5, device="cuda", requires_grad=True)
    _, state = model(first_detached)
    detached_state = state.detach()
    assert all(not key.requires_grad for key in detached_state.keys)
    assert all(not value.requires_grad for value in detached_state.values)
    prediction, _ = model(second_current, detached_state)
    prediction.square().sum().backward()

    assert first_detached.grad is None
    assert second_current.grad is not None and torch.isfinite(second_current.grad).all()
    assert second_current.grad.abs().sum() > 0


def test_autograd_dtype_and_state_dtype_follow_model_dtype() -> None:
    model = _model().to(dtype=torch.float64)
    inputs = torch.randn(1, 8, 5, device="cuda", dtype=torch.float64, requires_grad=True)

    prediction, state = model(inputs)
    prediction.sum().backward()

    assert prediction.dtype == torch.float64
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        key.dtype == value.dtype == torch.float64
        for key, value in zip(state.keys, state.values, strict=True)
    )


def test_state_dict_reload_preserves_predictions_and_export_head_shape() -> None:
    model = _model()
    inputs = torch.randn(1, 13, 5, device="cuda")
    expected, _ = model(inputs)

    restored = _model()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    actual, _ = restored(inputs)

    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert restored.state_dict()["head.2.weight"].shape == (2, 32)
