from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from wnn_jax.distributed_combo import (
    global_combo_loss_and_local_surrogate,
    pmean_and_clip_gradient,
)
from wnn_jax.rnn import adamw_init, adamw_update

CLIP_NORM = 0.01  # Deliberately low so the synthetic parity fixture exercises clipping.


def _production_combo_loss():
    source_path = Path(__file__).resolve().parents[1] / "src/wnn_jax/bridge.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "combo_sbd_loss"
    )
    namespace = {"Mapping": Mapping, "jax": jax, "jnp": jnp, "lax": jax.lax}
    isolated = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["combo_sbd_loss"]


combo_sbd_loss = _production_combo_loss()


def _batch(shards: int = 2, batch_size: int = 2, rows: int = 3):
    key = jax.random.PRNGKey(503)
    labels = jax.random.uniform(key, (shards, batch_size, rows, 2), minval=-1.5, maxval=1.5)
    weights = jnp.asarray(
        [
            [[0.2, 0.4, 0.0], [0.1, 0.3, 0.5]],
            [[0.02, 0.04, 0.0], [0.01, 0.03, 0.05]],
        ],
        jnp.float32,
    )
    struct = jnp.zeros((shards, batch_size, rows, 3), dtype=jnp.int32)
    struct = struct.at[..., 0].set(1).at[..., 1].set(2).at[..., 2].set(3)
    return {
        "labels": labels,
        "main_weights": weights,
        "fixed_target": jnp.zeros_like(labels),
        "structural_labels": struct,
        "next_weights": weights * jnp.asarray([1.0, 0.5, 2.0]),
        "next_rows": jnp.full((shards, batch_size, rows), 3.0),
        "next_up": jnp.asarray(weights > 0, jnp.float32),
    }


def _combine_shards(batch):
    return {name: value.reshape((-1, *value.shape[2:])) for name, value in batch.items()}


def _predictions(params, batch):
    shape = (*batch["labels"].shape[:-1], 23)
    return jnp.broadcast_to(params, shape)


def _check_distributed_matches_reference(
    batch, *, expect_clipping: bool = True, weight_decay: float = 0.01
) -> None:
    ndev = 2
    assert jax.local_device_count() >= ndev, "run with two forced CPU devices"
    batch = {name: value[:ndev] for name, value in batch.items()}
    ema = jnp.linspace(-0.4, 0.4, 23, dtype=jnp.float32)
    ema_shards = jnp.broadcast_to(ema, (ndev, 23))
    initial = jnp.linspace(-0.2, 0.3, 23, dtype=jnp.float32)
    replicated = jnp.broadcast_to(initial, (ndev, 23))

    def one_replica(params, local_batch, local_ema):
        pred = _predictions(params, local_batch)

        def surrogate(current):
            current_pred = _predictions(current, local_batch)
            _, contribution = global_combo_loss_and_local_surrogate(
                current_pred,
                local_batch,
                _predictions(local_ema, local_batch),
                axis_name="replicas",
                replica_count=ndev,
            )
            return contribution

        raw_gradient = jax.grad(surrogate)(params)
        gradient, norm = pmean_and_clip_gradient(
            {"toy": raw_gradient}, axis_name="replicas", max_norm=CLIP_NORM
        )
        params_tree = {"toy": params}
        updated, optimizer = adamw_update(
            params_tree,
            gradient,
            adamw_init(params_tree),
            learning_rate=3e-4,
            weight_decay=weight_decay,
        )
        report, _ = global_combo_loss_and_local_surrogate(
            pred,
            local_batch,
            _predictions(local_ema, local_batch),
            axis_name="replicas",
            replica_count=ndev,
        )
        return (
            report,
            gradient["toy"],
            norm,
            updated["toy"],
            optimizer.count,
            optimizer.first["toy"],
            optimizer.second["toy"],
        )

    distributed = jax.pmap(one_replica, axis_name="replicas")(replicated, batch, ema_shards)
    reports, gradients, norms, updates, counts, first_moments, second_moments = distributed

    merged = _combine_shards(batch)
    reference_ema = _predictions(ema, merged)

    def reference(params):
        return combo_sbd_loss(_predictions(params, merged), merged, reference_ema)

    ref_loss, ref_grad = jax.value_and_grad(reference)(initial)
    ref_norm = jnp.linalg.norm(ref_grad)
    if expect_clipping:
        assert float(ref_norm) > CLIP_NORM, "positive-mass fixture should exercise clipping"
    else:
        assert float(ref_norm) == 0.0, "all-zero mass should have exactly zero global gradient norm"
        assert np.isfinite(np.asarray(ref_loss)).all(), "clamped all-zero loss must be finite"
        np.testing.assert_array_equal(np.asarray(ref_grad), np.zeros_like(np.asarray(ref_grad)))
    ref_clipped = ref_grad * jnp.minimum(1.0, CLIP_NORM / (ref_norm + 1e-6))
    ref_params, ref_optimizer = adamw_update(
        {"toy": initial},
        {"toy": ref_clipped},
        adamw_init({"toy": initial}),
        learning_rate=3e-4,
        weight_decay=weight_decay,
    )
    np.testing.assert_allclose(
        np.asarray(reports),
        np.broadcast_to(np.asarray(ref_loss), (ndev,)),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(norms),
        np.broadcast_to(np.asarray(ref_norm), (ndev,)),
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        np.asarray(gradients),
        np.broadcast_to(np.asarray(ref_clipped), (ndev, 23)),
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        np.asarray(updates),
        np.broadcast_to(np.asarray(ref_params["toy"]), (ndev, 23)),
        rtol=3e-6,
        atol=3e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(counts), np.broadcast_to(np.asarray(ref_optimizer.count), (ndev,))
    )
    for distributed_state, reference_state in (
        (first_moments, ref_optimizer.first["toy"]),
        (second_moments, ref_optimizer.second["toy"]),
    ):
        assert np.isfinite(np.asarray(distributed_state)).all()
        np.testing.assert_allclose(
            np.asarray(distributed_state),
            np.broadcast_to(np.asarray(reference_state), (ndev, 23)),
            rtol=3e-6,
            atol=3e-7,
        )
    if not expect_clipping:
        np.testing.assert_array_equal(
            np.asarray(gradients), np.zeros_like(np.asarray(gradients))
        )
        np.testing.assert_array_equal(
            np.asarray(updates), np.broadcast_to(np.asarray(initial), (ndev, 23))
        )
        assert np.isfinite(np.asarray(gradients)).all()
        assert np.isfinite(np.asarray(updates)).all()
        assert np.isfinite(np.asarray(first_moments)).all()
        assert np.isfinite(np.asarray(second_moments)).all()


def test_unequal_positive_mass_shards_match_pooled_loss_and_gradient() -> None:
    _check_distributed_matches_reference(_batch())


def test_one_zero_mass_shard_matches_pooled_loss_and_gradient() -> None:
    batch = _batch()
    batch["main_weights"] = batch["main_weights"].at[1].set(0.0)
    batch["next_weights"] = batch["next_weights"].at[1].set(0.0)
    _check_distributed_matches_reference(batch)


def test_all_zero_global_mass_matches_reference_clamped_denominators() -> None:
    batch = _batch()
    batch["main_weights"] = jnp.zeros_like(batch["main_weights"])
    batch["next_weights"] = jnp.zeros_like(batch["next_weights"])
    _check_distributed_matches_reference(batch, expect_clipping=False, weight_decay=0.0)
