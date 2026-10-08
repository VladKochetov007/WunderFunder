"""Matched chunk-update primitives for the GRU256 TPU benchmark.

The model forward and FP32 AdamW implementation come from the project's
generic JAX GRU core. This module adds only the WMSE/Huber objectives, global
gradient clipping and optional replica reduction required by LOSS_PAIR_R2.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from wnn_jax.rnn import AdamWState, ComputePrecision, adamw_update, forward_chunk

LossKind = Literal["wmse", "huber"]


def weighted_loss_terms(
    prediction: jax.Array,
    target: jax.Array,
    row_mask: jax.Array,
    *,
    loss_kind: LossKind,
    huber_delta: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Return the exact weighted objective numerator and denominator."""
    if prediction.shape != target.shape or target.shape[-1] != 2:
        raise ValueError("prediction and target must share [..., time, 2] shape")
    if row_mask.shape != target.shape[:-1]:
        raise ValueError("row_mask must match prediction rows")
    clipped_target = jnp.clip(target.astype(jnp.float32), -2.0, 2.0)
    error = prediction.astype(jnp.float32) - clipped_target
    weight = jnp.abs(clipped_target) * row_mask.astype(jnp.float32)[..., None]
    if loss_kind == "wmse":
        element_loss = jnp.square(error)
    elif loss_kind == "huber":
        absolute = jnp.abs(error)
        delta = jnp.asarray(huber_delta, dtype=jnp.float32)
        element_loss = jnp.where(
            absolute <= delta,
            0.5 * jnp.square(error),
            delta * (absolute - 0.5 * delta),
        )
    else:
        raise ValueError(f"unsupported loss kind: {loss_kind}")
    return jnp.sum(weight * element_loss), jnp.sum(weight)


@partial(jax.jit, static_argnames=("loss_kind", "precision", "replica_axis_name"))
def train_chunk_step(
    params: Mapping[str, jax.Array],
    optimizer_state: AdamWState,
    inputs: jax.Array,
    recurrent_state: jax.Array | None,
    targets: jax.Array,
    row_mask: jax.Array,
    *,
    learning_rate: jax.Array | float,
    beta1: jax.Array | float,
    weight_decay: float = 0.01,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    gradient_clip_norm: float = 1.0,
    huber_delta: float = 1.0,
    loss_kind: LossKind,
    precision: ComputePrecision,
    replica_axis_name: str | None = None,
) -> tuple[
    dict[str, jax.Array],
    AdamWState,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Perform one chunk-level update and carry pre-update state detached.

    With ``replica_axis_name``, inputs/targets/masks have a leading replica
    axis removed by pmap. Gradients are summed from local numerators over the
    global denominator before one identical AdamW update on every replica.
    """
    clipped_targets = jnp.clip(targets.astype(jnp.float32), -2.0, 2.0)
    weights = jnp.abs(clipped_targets) * row_mask.astype(jnp.float32)[..., None]
    local_den = jnp.sum(weights)
    if replica_axis_name is not None:
        global_den = lax.psum(local_den, replica_axis_name)
    else:
        global_den = local_den
    global_den = jnp.maximum(global_den, 1e-8)

    def objective(current: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        prediction, next_state = forward_chunk(
            current, inputs, recurrent_state, precision=precision
        )
        numerator, _ = weighted_loss_terms(
            prediction, targets, row_mask, loss_kind=loss_kind, huber_delta=huber_delta
        )
        return numerator / global_den, next_state

    (local_loss, next_state), local_gradients = jax.value_and_grad(objective, has_aux=True)(params)
    if replica_axis_name is not None:
        gradients = jax.tree_util.tree_map(
            lambda gradient: lax.psum(gradient, replica_axis_name), local_gradients
        )
        reported_loss = lax.psum(local_loss, replica_axis_name)
    else:
        gradients = local_gradients
        reported_loss = local_loss
    squared_norm = sum(
        jnp.sum(jnp.square(gradient.astype(jnp.float32)))
        for gradient in jax.tree_util.tree_leaves(gradients)
    )
    gradient_norm = jnp.sqrt(squared_norm)
    scale = jnp.minimum(
        1.0,
        jnp.asarray(gradient_clip_norm, dtype=jnp.float32)
        / (gradient_norm + jnp.asarray(1e-6, dtype=jnp.float32)),
    )
    clipped_gradients = jax.tree_util.tree_map(lambda gradient: gradient * scale, gradients)
    finite = jnp.isfinite(reported_loss) & jnp.isfinite(gradient_norm)
    finite = finite & jnp.all(
        jnp.stack(
            [jnp.all(jnp.isfinite(gradient)) for gradient in jax.tree_util.tree_leaves(gradients)]
        )
    )

    def apply_update(_: None) -> tuple[dict[str, jax.Array], AdamWState]:
        return adamw_update(
            params,
            clipped_gradients,
            optimizer_state,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    def skip_update(_: None) -> tuple[dict[str, jax.Array], AdamWState]:
        return dict(params), optimizer_state

    updated, next_optimizer = lax.cond(finite, apply_update, skip_update, operand=None)
    return (
        updated,
        next_optimizer,
        lax.stop_gradient(next_state),
        reported_loss,
        gradient_norm,
        scale,
        finite,
    )
