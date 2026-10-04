from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from wnn.nn import RNNModel
from wnn_jax.rnn import (
    adamw_init,
    export_npz_state,
    forward_chunk,
    forward_chunks,
    load_npz_state,
    train_chunk_step,
    train_step,
    weighted_fixed_target_mse,
)


def _torch_model() -> RNNModel:
    torch.manual_seed(1234)
    model = RNNModel(n_in=7, hidden=5, layers=2, dropout=0.0, n_out=3).eval()
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.2, 0.2, 7))
        model.std.copy_(torch.linspace(0.8, 1.2, 7))
    return model


def _jax_params(model: RNNModel) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }


def test_two_layer_gru_matches_torch_and_npz_layout_roundtrips(tmp_path) -> None:
    model = _torch_model()
    params = _jax_params(model)
    values = torch.randn(2, 111, 7)
    with torch.no_grad():
        expected, expected_state = model(values)
    actual, actual_state = forward_chunk(params, jnp.asarray(values.numpy()))

    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-5, atol=4e-6)
    np.testing.assert_allclose(
        np.asarray(actual_state), expected_state.numpy(), rtol=2e-5, atol=4e-6
    )

    path = tmp_path / "weights.npz"
    export_npz_state(path, params)
    loaded = load_npz_state(path, n_in=7, hidden=5, layers=2, n_out=3)
    assert set(loaded) == set(model.state_dict())
    for name, expected_tensor in model.state_dict().items():
        np.testing.assert_array_equal(np.asarray(loaded[name]), expected_tensor.numpy())


def test_causal_full_sequence_and_carried_chunks_match_through_prefix_boundary() -> None:
    model = _torch_model()
    params = _jax_params(model)
    values = jax.random.normal(jax.random.PRNGKey(11), (2, 137, 7))
    full, full_state = forward_chunk(params, values)

    chunks = []
    state = None
    for start, stop in ((0, 37), (37, 99), (99, 137)):
        output, state = forward_chunk(params, values[:, start:stop], state)
        chunks.append(output)
    np.testing.assert_allclose(np.asarray(jnp.concatenate(chunks, axis=1)), np.asarray(full),
                               rtol=2e-5, atol=4e-6)
    np.testing.assert_allclose(np.asarray(state), np.asarray(full_state), rtol=2e-5, atol=4e-6)

    altered = values.at[:, 99:, :].add(50.0)
    changed, _ = forward_chunk(params, altered)
    np.testing.assert_allclose(np.asarray(changed[:, :99]), np.asarray(full[:, :99]),
                               rtol=0, atol=0)


def test_tbptt_stop_gradient_blocks_carry_but_preserves_forward_values() -> None:
    params = _jax_params(_torch_model())
    values = jax.random.normal(jax.random.PRNGKey(12), (1, 12, 7))
    carried, _ = forward_chunks(params, values, chunk_rows=4)
    detached, _ = forward_chunks(params, values, chunk_rows=4, detach_between_chunks=True)
    np.testing.assert_allclose(np.asarray(detached), np.asarray(carried), rtol=0, atol=0)

    def suffix_loss(inputs: jax.Array) -> jax.Array:
        predictions, _ = forward_chunks(
            params, inputs, chunk_rows=4, detach_between_chunks=True
        )
        return jnp.square(predictions[:, 8:, :]).sum()

    gradient = jax.grad(suffix_loss)(values)
    np.testing.assert_array_equal(np.asarray(gradient[:, :8, :]), 0.0)
    assert np.linalg.norm(np.asarray(gradient[:, 8:, :])) > 0


def test_weighted_fixed_target_loss_respects_batch_row_mask_and_torch_formula() -> None:
    prediction = jnp.asarray([[[0.5, -0.25], [1.0, 2.0]], [[-2.0, 1.5], [0.0, 0.0]]])
    target = jnp.asarray([[[0.0, 0.5], [1.0, 1.0]], [[-1.0, 0.5], [4.0, 4.0]]])
    label = jnp.asarray([[[1.0, -2.0], [0.5, 0.5]], [[-1.5, 1.0], [2.0, 2.0]]])
    mask = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    actual = weighted_fixed_target_mse(prediction, target, label, mask)

    clipped_label = torch.tensor(np.asarray(label)).clamp(-2, 2)
    clipped_target = torch.tensor(np.asarray(target)).clamp(-2, 2)
    weights = clipped_label.abs() * torch.tensor(np.asarray(mask))[..., None]
    expected = (weights * (torch.tensor(np.asarray(prediction)) - clipped_target).square()).sum()
    expected = expected / weights.sum().clamp_min(1e-8)
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-6, atol=1e-7)

    empty = weighted_fixed_target_mse(prediction, target, label, jnp.zeros_like(mask))
    assert float(empty) == 0.0


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_training_step_updates_fp32_parameters_and_runs_both_precisions(precision: str) -> None:
    model = _torch_model()
    params = _jax_params(model)
    optimizer = adamw_init(params)
    inputs = jax.random.normal(jax.random.PRNGKey(21), (2, 9, 7))
    target = jnp.zeros((2, 9, 2), dtype=jnp.float32)
    labels = jnp.ones_like(target)
    mask = jnp.asarray([[1.0] * 9, [1.0, 1.0, 0, 0, 0, 0, 0, 0, 0]])

    updated, next_optimizer, loss = train_step(
        params,
        optimizer,
        inputs,
        target,
        labels,
        mask,
        chunk_rows=4,
        learning_rate=1e-3,
        precision=precision,
    )
    assert np.isfinite(float(loss))
    assert int(next_optimizer.count) == 1
    np.testing.assert_array_equal(np.asarray(updated["mean"]), np.asarray(params["mean"]))
    np.testing.assert_array_equal(np.asarray(updated["std"]), np.asarray(params["std"]))
    assert all(np.isfinite(np.asarray(value)).all() for value in updated.values())


def test_chunk_level_updates_carry_state_and_step_adamw_once_per_chunk() -> None:
    params = _jax_params(_torch_model())
    optimizer = adamw_init(params)
    inputs = jax.random.normal(jax.random.PRNGKey(22), (2, 5, 7))
    target = jnp.zeros((2, 5, 2), dtype=jnp.float32)
    label = jnp.ones_like(target)
    mask = jnp.ones((2, 5), dtype=jnp.float32)

    after_first, optimizer, state, first_loss = train_chunk_step(
        params,
        optimizer,
        inputs,
        None,
        target,
        label,
        mask,
        learning_rate=1e-3,
    )
    after_second, optimizer, next_state, second_loss = train_chunk_step(
        after_first,
        optimizer,
        inputs,
        state,
        target,
        label,
        mask,
        learning_rate=1e-3,
    )
    assert int(optimizer.count) == 2
    assert state.shape == next_state.shape == (2, 2, 5)
    assert np.isfinite(float(first_loss)) and np.isfinite(float(second_loss))
    assert not np.array_equal(np.asarray(params["inp.weight"]), np.asarray(after_second["inp.weight"]))
