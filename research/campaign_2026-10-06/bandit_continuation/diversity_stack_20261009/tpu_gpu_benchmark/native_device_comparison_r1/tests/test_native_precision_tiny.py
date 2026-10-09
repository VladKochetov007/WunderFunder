"""Tiny BF16 compute check; actual full-shape TPU gate remains mandatory."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-04/staging/jax_tpu_e465/src"))
from jax_step import train_chunk_step  # noqa: E402
from rnn_native import adamw_init, expected_state_shapes  # noqa: E402


def test_batched_input_gates_preserve_forward_gradient_and_adamw():
    from rnn_native import adamw_update, forward_chunk
    from wnn_jax import rnn as reference

    rng = np.random.default_rng(9)
    params = {
        key: jnp.asarray(rng.normal(0, 0.05, shape).astype(np.float32))
        for key, shape in expected_state_shapes(4, 8, 1, 2).items()
    }
    params["mean"] = jnp.zeros(4, dtype=jnp.float32)
    params["std"] = jnp.ones(4, dtype=jnp.float32)
    inputs = jnp.asarray(rng.normal(size=(2, 5, 4)).astype(np.float32))

    def objective(current, forward):
        prediction, state = forward(current, inputs, precision="fp32")
        return jnp.sum(jnp.square(prediction)), (prediction, state)

    native = jax.jit(jax.value_and_grad(lambda current: objective(current, forward_chunk), has_aux=True))(params)
    original = jax.jit(jax.value_and_grad(lambda current: objective(current, reference.forward_chunk), has_aux=True))(params)
    jax.block_until_ready((native, original))
    for left, right in zip(jax.tree_util.tree_leaves(native), jax.tree_util.tree_leaves(original), strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=4e-4, atol=2e-6)
    updated, optimizer = adamw_update(params, native[1], adamw_init(params), learning_rate=0.001, beta1=0.95)
    baseline, reference_optimizer = reference.adamw_update(params, original[1], reference.adamw_init(params), learning_rate=0.001, beta1=0.95)
    for left, right in zip(jax.tree_util.tree_leaves((updated, optimizer)), jax.tree_util.tree_leaves((baseline, reference_optimizer)), strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=4e-4, atol=2e-6)


@pytest.mark.parametrize("loss", ["wmse", "huber"])
def test_bf16_compute_keeps_updated_masters_and_moments_fp32(tmp_path, loss):
    rng = np.random.default_rng(71)
    params = {
        key: jnp.asarray(rng.normal(0, 0.05, shape).astype(np.float32))
        for key, shape in expected_state_shapes(4, 8, 1, 2).items()
    }
    params["mean"] = jnp.zeros(4, dtype=jnp.float32)
    params["std"] = jnp.ones(4, dtype=jnp.float32)
    inputs = jnp.asarray(rng.normal(size=(2, 3, 4)).astype(np.float32))
    targets = jnp.ones((2, 3, 2), dtype=jnp.float32)
    mask = jnp.asarray([[False, True, True], [False, False, False]])
    result = train_chunk_step(params, adamw_init(params), inputs, None, targets, mask,
                              learning_rate=0.001, beta1=0.95,
                              precision="bf16", loss_kind=loss)
    jax.block_until_ready(result)
    assert bool(result[-1])
    assert result[2].dtype == jnp.bfloat16
    leaves = jax.tree_util.tree_leaves((result[0], result[1]))
    for value in leaves:
        if value.dtype.kind == "f":
            assert value.dtype == jnp.float32
    arrays = {f"leaf_{index}": np.asarray(value) for index, value in enumerate(leaves)}
    path = tmp_path / "updated_masters.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as reloaded:
        for key, expected in arrays.items():
            assert reloaded[key].dtype == expected.dtype
            np.testing.assert_array_equal(reloaded[key], expected)
