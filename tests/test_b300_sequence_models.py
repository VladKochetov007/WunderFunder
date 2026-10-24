"""Remote-execution coverage for the causal B300 sequence adapters."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from wnn.nn import RNNModel
from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_models import build_sequence_model, detach_state


@pytest.fixture(autouse=True)
def _strict_precision() -> Iterator[None]:
    """Exercise the same explicit precision contract as fit and prediction setup."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    configure_strict_fp32()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "transformer",
        "hidden": 32,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 4,
        "context_rows": 5,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
    }
    config.update(overrides)
    return config


def test_transformer_prefix_causality_and_chunk_stream_parity() -> None:
    torch.manual_seed(21)
    model = build_sequence_model(_config())
    model.eval()
    values = torch.randn(2, 13, 112)
    full_prediction, full_hints, full_state = model(values)
    altered = values.clone()
    altered[:, 7:] += 100.0
    altered_prediction, _, _ = model(altered)
    torch.testing.assert_close(full_prediction[:, :7], altered_prediction[:, :7])

    predictions: list[Tensor] = []
    hints: list[Tensor] = []
    state: TransformerState | None = None
    for start, end in ((0, 2), (2, 8), (8, 9), (9, 13)):
        prediction, hint, state = model(values[:, start:end], state)
        predictions.append(prediction)
        hints.append(hint)
    torch.testing.assert_close(torch.cat(predictions, dim=1), full_prediction)
    torch.testing.assert_close(torch.cat(hints, dim=1), full_hints)
    assert state is not None
    assert state.next_position == values.shape[1]
    assert full_state.next_position == values.shape[1]
    assert all(key.shape[-2] <= model.context_rows - 1 for key in state.keys)


def test_transformer_reset_and_state_detachment() -> None:
    model = build_sequence_model(_config(hint_dim=3))
    model.eval()
    first = torch.randn(2, 4, 112)
    second = torch.randn(2, 3, 112)
    _, _, carried = model(first)
    assert isinstance(carried, TransformerState)
    _, _, continued = model(second, carried)
    _, hints, reset = model(second)
    assert isinstance(continued, TransformerState)
    assert isinstance(reset, TransformerState)
    assert continued.next_position == first.shape[1] + second.shape[1]
    assert reset.next_position == second.shape[1]
    assert hints.shape == (2, second.shape[1], 3)
    detached = detach_state(continued)
    assert isinstance(detached, TransformerState)
    assert all(key.grad_fn is None for key in detached.keys)
    assert all(value.grad_fn is None for value in detached.values)


def test_gru_adapter_matches_canonical_rnn_model() -> None:
    torch.manual_seed(1729)
    adapter = build_sequence_model(_config(architecture="gru", hint_dim=0))
    reference = RNNModel(112, hidden=32, layers=2, dropout=0.0, n_out=2, cell="gru", enc="linear")
    reference.load_state_dict(adapter.state_dict(), strict=True)
    adapter.eval()
    reference.eval()
    values = torch.randn(3, 7, 112)
    expected, expected_state = reference(values)
    actual, hints, actual_state = adapter(values)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)
    assert hints.shape == (3, 7, 0)


def test_fixed_hint_projection_preserves_initialization_rng() -> None:
    torch.manual_seed(99)
    build_sequence_model(_config(hint_dim=0))
    no_hint_rng = torch.random.get_rng_state()
    torch.manual_seed(99)
    fixed = build_sequence_model(_config(hint_dim=8, hint_projection="fixed"))
    fixed_hint_rng = torch.random.get_rng_state()
    assert torch.equal(no_hint_rng, fixed_hint_rng)
    assert fixed.hint_projector.fixed_projection is not None
    assert fixed.hint_projector.fixed_projection.shape == (8, 32)
    assert not fixed.hint_projector.fixed_projection.requires_grad


def test_checkpointed_transformer_backpropagates_and_detaches_explicitly() -> None:
    model = build_sequence_model(_config(checkpoint_blocks=True, hint_dim=4))
    model.train()
    values = torch.randn(2, 6, 112, requires_grad=True)
    prediction, hints, state = model(values)
    (prediction.square().mean() + hints.square().mean()).backward()
    assert values.grad is not None
    assert model.input_projection.weight.grad is not None
    assert all(block.qkv.weight.grad is not None for block in model.blocks)
    detached = detach_state(state)
    assert isinstance(detached, TransformerState)
    assert all(key.grad_fn is None for key in detached.keys)
