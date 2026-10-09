from __future__ import annotations

import numpy as np
import torch

from wnn.conditional_readout import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_WIDTH,
    ConditionalReadoutHeads,
    freeze_parent,
    iter_frozen_parent_chunks,
    parent_inputs_from_raw,
    primary_pair_wmse,
)
from wnn.features import FEATURE_NAMES
from wnn.nn import RNNModel


def _parent() -> RNNModel:
    torch.manual_seed(41)
    model = RNNModel(n_in=211, hidden=8, layers=2, n_out=23).eval()
    freeze_parent(model)
    return model


def _features(batch: int = 2, rows: int = 23) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(73)
    raw = torch.randn(batch, rows, 112, dtype=torch.float32)
    inputs, context = parent_inputs_from_raw(raw)
    return raw, inputs, context


def _stream(
    model: RNNModel, inputs: torch.Tensor, context: torch.Tensor, chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = list(
        iter_frozen_parent_chunks(
            model,
            inputs,
            context,
            chunk_rows,
            amp_enabled=False,
        )
    )
    return tuple(torch.cat([chunk[index] for chunk in chunks], dim=1) for index in (1, 2, 3))


def test_event_context_is_exactly_parent_ordered_raw_plus_99() -> None:
    assert CONTEXT_WIDTH == 99
    assert tuple(FEATURE_NAMES) == CONTEXT_FEATURE_NAMES
    _, inputs, context = _features()
    assert inputs.shape[-1] == 211
    torch.testing.assert_close(inputs[..., 112:], context, rtol=0, atol=0)


def test_zero_initialized_controls_are_identity_and_parameters_are_matched() -> None:
    _, inputs, context = _features()
    parent = _parent()
    base, hidden, context_stream = _stream(parent, inputs, context, chunk_rows=7)
    heads = ConditionalReadoutHeads(hidden_width=16, rank=4)
    outputs = heads.forward_all(base, hidden, context_stream)
    assert len(set(heads.parameter_counts().values())) == 1
    for name in heads.parameter_counts():
        torch.testing.assert_close(outputs[name], base[..., :2], rtol=0, atol=0)


def test_streaming_matches_full_gru_and_resets_between_sequences() -> None:
    raw, inputs, context = _features(batch=2, rows=29)
    model = _parent()
    projected = model.encode((inputs - model.mean) / model.std)
    recurrent, _ = model.gru(projected)
    expected_hidden = torch.cat((recurrent, projected), dim=-1)
    expected_base = model.head(expected_hidden)
    base, hidden, event_context = _stream(model, inputs, context, chunk_rows=8)
    torch.testing.assert_close(base, expected_base, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden, expected_hidden, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(event_context, context, rtol=0, atol=0)

    prefix_inputs, prefix_context = parent_inputs_from_raw(raw[:, :13])
    torch.testing.assert_close(prefix_inputs, inputs[:, :13], rtol=0, atol=0)
    torch.testing.assert_close(prefix_context, context[:, :13], rtol=0, atol=0)
    prefix_base, prefix_hidden, _ = _stream(model, prefix_inputs, prefix_context, chunk_rows=5)
    torch.testing.assert_close(prefix_base, base[:, :13], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(prefix_hidden, hidden[:, :13], rtol=1e-6, atol=1e-6)

    separate_base, separate_hidden, _ = _stream(model, inputs[1:], context[1:], chunk_rows=6)
    torch.testing.assert_close(separate_base, base[1:], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(separate_hidden, hidden[1:], rtol=1e-6, atol=1e-6)


def test_loss_matches_original_label_magnitude_recipe() -> None:
    prediction = torch.tensor([[[0.5, -0.25], [0.0, 0.75]]])
    target = torch.tensor([[[0.2, -0.1], [0.3, 0.4]]])
    labels = torch.tensor([[[1.5, -0.5], [0.25, -1.0]]])
    row_weights = torch.tensor([[0.4, 0.8]])
    weights = labels.numpy().clip(-2, 2).__abs__() * row_weights.numpy()[..., None]
    expected = (
        weights * (prediction.numpy() - target.numpy().clip(-2, 2)) ** 2
    ).sum() / weights.sum()
    observed = primary_pair_wmse(prediction, target, row_weights, labels)
    assert np.isclose(float(observed), float(expected), rtol=1e-6, atol=1e-7)


def test_head_gradients_are_finite_parent_stays_frozen_and_first_step_learns() -> None:
    _, inputs, context = _features(batch=1, rows=19)
    parent = _parent()
    original_parent = {key: value.detach().clone() for key, value in parent.state_dict().items()}
    base, hidden, event_context = _stream(parent, inputs, context, chunk_rows=7)
    heads = ConditionalReadoutHeads(hidden_width=16, rank=4)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    labels = torch.ones_like(base[..., :2])
    target = base[..., :2].detach() + 0.25
    row_weights = torch.ones_like(target[..., 0])
    outputs = heads.forward_all(base, hidden, event_context)
    loss = torch.stack(
        [
            primary_pair_wmse(outputs[name], target, row_weights, labels)
            for name in heads.parameter_counts()
        ]
    ).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in heads.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert all(
        heads.controls[name].output.weight.grad is not None for name in heads.parameter_counts()
    )
    assert all(parameter.grad is None for parameter in parent.parameters())
    torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    changed = heads.forward_all(base, hidden, event_context)
    assert any(not torch.equal(changed[name], base[..., :2]) for name in heads.parameter_counts())
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, original_parent[key], rtol=0, atol=0)
