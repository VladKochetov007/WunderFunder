from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from wnn_jax.rnn import forward_chunk


def _small_params():
    key = jax.random.PRNGKey(719)
    shapes = {
        "mean": (3,),
        "std": (3,),
        "inp.weight": (2, 3),
        "inp.bias": (2,),
        "head.0.weight": (2, 4),
        "head.0.bias": (2,),
        "head.2.weight": (23, 2),
        "head.2.bias": (23,),
        "gru.weight_ih_l0": (6, 2),
        "gru.weight_hh_l0": (6, 2),
        "gru.bias_ih_l0": (6,),
        "gru.bias_hh_l0": (6,),
    }
    keys = jax.random.split(key, len(shapes))
    values = {}
    for (name, shape), subkey in zip(shapes.items(), keys, strict=True):
        if name == "mean":
            values[name] = jnp.zeros(shape, jnp.float32)
        elif name == "std":
            values[name] = jnp.ones(shape, jnp.float32)
        else:
            values[name] = jax.random.normal(subkey, shape) * 0.08
    return values


def _carry_loss(params, ema_params, values):
    student_first, student_state = forward_chunk(params, values[:, :2])
    ema_first, ema_state = forward_chunk(ema_params, values[:, :2])
    student_second, student_final = forward_chunk(
        params, values[:, 2:], lax.stop_gradient(student_state)
    )
    ema_second, ema_final = forward_chunk(ema_params, values[:, 2:], lax.stop_gradient(ema_state))
    consistency = jnp.mean((student_second[..., :2] - lax.stop_gradient(ema_second[..., :2])) ** 2)
    objective = jnp.mean(student_first[..., :2] ** 2) + consistency
    return objective, (student_first, student_second, student_final, ema_final)


def test_replicated_two_chunk_state_and_detached_gradient_match_unsharded() -> None:
    ndev = 2
    assert jax.local_device_count() >= ndev, "run with two forced CPU devices"
    params = _small_params()
    ema_params = jax.tree_util.tree_map(lambda value: value * 0.9, params)
    key = jax.random.PRNGKey(720)
    values = jax.random.normal(key, (ndev, 1, 4, 3), dtype=jnp.float32) * 0.2
    replicated_params = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (ndev, *value.shape)), params
    )
    replicated_ema = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (ndev, *value.shape)), ema_params
    )

    def replica_objective(current, teacher, local_values):
        return _carry_loss(current, teacher, local_values)

    def replica(current, teacher, local_values):
        (loss, aux), gradients = jax.value_and_grad(replica_objective, has_aux=True)(
            current, teacher, local_values
        )
        pooled_loss = lax.pmean(loss, "replicas")
        gradients = jax.tree_util.tree_map(lambda value: lax.pmean(value, "replicas"), gradients)
        return pooled_loss, aux, gradients

    losses, aux, gradients = jax.pmap(replica, axis_name="replicas")(
        replicated_params, replicated_ema, values
    )

    combined = values.reshape((ndev, 4, 3))
    ref_loss, ref_aux = _carry_loss(params, ema_params, combined)
    ref_gradients = jax.grad(lambda current: _carry_loss(current, ema_params, combined)[0])(params)
    np.testing.assert_allclose(
        np.asarray(losses),
        np.broadcast_to(np.asarray(ref_loss), (ndev,)),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(aux[0]).reshape((ndev, 2, 23)),
        np.asarray(ref_aux[0]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(aux[1]).reshape((ndev, 2, 23)),
        np.asarray(ref_aux[1]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(aux[2])[:, 0].transpose((1, 0, 2)),
        np.asarray(ref_aux[2]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(aux[3])[:, 0].transpose((1, 0, 2)),
        np.asarray(ref_aux[3]),
        rtol=2e-6,
        atol=2e-7,
    )
    for name in params:
        np.testing.assert_allclose(
            np.asarray(gradients[name]),
            np.broadcast_to(
                np.asarray(ref_gradients[name]),
                (ndev, *ref_gradients[name].shape),
            ),
            rtol=4e-5,
            atol=2e-7,
            err_msg=name,
        )
