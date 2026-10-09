"""Global-denominator objective helpers for the E465 eight-core TPU draft.

Each replica computes local sufficient statistics. Denominators are reduced
across replicas before ratios are formed. Training differentiates a scaled
local numerator contribution and pmeans the resulting replicated-parameter
gradients; this is algebraically the gradient of the globally pooled loss.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import lax


def combo_local_sufficient_statistics(
    prediction: jax.Array,
    batch: Mapping[str, jax.Array],
    ema_prediction: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return four local objective numerators and their exact denominators.

    Structural CE is intentionally accumulated as one numerator over the
    common row-weight denominator, matching the sum of the three production
    structural-loss ratios.
    """
    prediction = prediction.astype(jnp.float32)
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    primary_num = jnp.sum(
        primary_weights
        * (prediction[..., :2] - jnp.clip(batch["fixed_target"].astype(jnp.float32), -2.0, 2.0))
        ** 2
    )
    primary_den = jnp.sum(primary_weights)

    consistency_num = jnp.sum(
        row_weights[..., None]
        * (prediction[..., :2] - lax.stop_gradient(ema_prediction[..., :2].astype(jnp.float32)))
        ** 2
    )
    consistency_den = jnp.sum(row_weights)

    struct_labels = batch["structural_labels"].astype(jnp.int32)
    structural_num = jnp.asarray(0.0, jnp.float32)
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        log_probs = jax.nn.log_softmax(prediction[..., start : start + width], axis=-1)
        selected = jnp.take_along_axis(log_probs, struct_labels[..., target_index, None], axis=-1)[
            ..., 0
        ]
        structural_num = structural_num - jnp.sum(selected * row_weights)

    next_weights = batch["next_weights"].astype(jnp.float32)
    next_distance = (batch["next_rows"].astype(jnp.float32) - 3.0) / 2.0
    next_mse = (prediction[..., 21] - next_distance) ** 2
    next_logit = prediction[..., 22]
    next_bce = jax.nn.softplus(next_logit) - batch["next_up"].astype(jnp.float32) * next_logit
    next_num = jnp.sum((next_mse + next_bce) * next_weights)
    next_den = jnp.sum(next_weights)
    return (
        jnp.stack((primary_num, consistency_num, structural_num, next_num)),
        jnp.stack((primary_den, consistency_den, consistency_den, next_den)),
    )


def global_combo_loss_and_local_surrogate(
    prediction: jax.Array,
    batch: Mapping[str, jax.Array],
    ema_prediction: jax.Array,
    *,
    axis_name: str,
    replica_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Return pooled report loss and scaled local contribution for gradients.

    The caller must differentiate the second return value and then `pmean`
    gradients over `axis_name`. The first value is the globally pooled scalar
    on every replica, suitable for metrics and finite-value checks.
    """
    if replica_count < 1:
        raise ValueError("replica_count must be positive")
    local_num, local_den = combo_local_sufficient_statistics(prediction, batch, ema_prediction)
    global_num = lax.psum(local_num, axis_name)
    global_den = lax.psum(local_den, axis_name)
    coefficients = jnp.asarray((1.0, 1.0, 0.3, 0.3), dtype=jnp.float32)
    pooled_loss = jnp.sum(coefficients * global_num / jnp.maximum(global_den, 1e-8))
    local_surrogate = replica_count * jnp.sum(
        coefficients * local_num / jnp.maximum(global_den, 1e-8)
    )
    return pooled_loss, local_surrogate


def pmean_and_clip_gradient(tree, *, axis_name: str, max_norm: float = 1.0):
    """Pmean replicated-parameter gradients, then apply Torch's global norm clip.

    Clipping follows `clip_grad_norm_`: the norm is computed after replica
    reduction and the divisor includes its 1e-6 numerical guard. Clipping
    per-replica gradients before pmean is not equivalent.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    reduced = jax.tree_util.tree_map(lambda value: lax.pmean(value, axis_name), tree)
    squared_norm = sum(
        jnp.sum(jnp.square(value.astype(jnp.float32))) for value in jax.tree.leaves(reduced)
    )
    total_norm = jnp.sqrt(squared_norm)
    coefficient = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    clipped = jax.tree_util.tree_map(lambda value: value * coefficient, reduced)
    return clipped, total_norm
