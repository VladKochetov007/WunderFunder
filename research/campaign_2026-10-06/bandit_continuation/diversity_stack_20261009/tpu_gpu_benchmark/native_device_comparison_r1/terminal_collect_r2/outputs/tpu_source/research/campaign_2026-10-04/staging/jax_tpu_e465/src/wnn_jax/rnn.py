"""JAX implementation of the repository's Torch linear-GRU RNNModel."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ComputePrecision = Literal["fp32", "bf16"]


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy-like or Torch tensor state value without importing Torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def expected_state_shapes(n_in: int, hidden: int, layers: int, n_out: int) -> dict[str, tuple[int, ...]]:
    """Return exact RNNModel state-dict names and shapes for a linear GRU."""
    if min(n_in, hidden, layers, n_out) < 1:
        raise ValueError("model dimensions must be positive")
    shapes: dict[str, tuple[int, ...]] = {
        "mean": (n_in,),
        "std": (n_in,),
        "inp.weight": (hidden, n_in),
        "inp.bias": (hidden,),
        "head.0.weight": (hidden, 2 * hidden),
        "head.0.bias": (hidden,),
        "head.2.weight": (n_out, hidden),
        "head.2.bias": (n_out,),
    }
    for layer in range(layers):
        input_size = hidden
        shapes.update(
            {
                f"gru.weight_ih_l{layer}": (3 * hidden, input_size),
                f"gru.weight_hh_l{layer}": (3 * hidden, hidden),
                f"gru.bias_ih_l{layer}": (3 * hidden,),
                f"gru.bias_hh_l{layer}": (3 * hidden,),
            }
        )
    return shapes


def validate_state_dict(
    state: Mapping[str, Any], n_in: int, hidden: int, layers: int, n_out: int
) -> dict[str, np.ndarray]:
    """Validate and normalize a state dict to float32 PyTorch-layout arrays."""
    expected = expected_state_shapes(n_in, hidden, layers, n_out)
    expected_keys = set(expected)
    actual_keys = set(state.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(f"state keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
    arrays: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        array = _as_numpy(state[name])
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite numeric values")
        arrays[name] = array.astype(np.float32, copy=False)
    if np.any(arrays["std"] <= 0):
        raise ValueError("input normalization std must be positive")
    return arrays


def load_npz_state(
    path: Path, n_in: int, hidden: int, layers: int, n_out: int
) -> dict[str, jax.Array]:
    """Load a no-pickle NPZ with the exact named PyTorch RNNModel tensors."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = validate_state_dict(archive, n_in, hidden, layers, n_out)
    return {name: jnp.asarray(value, dtype=jnp.float32) for name, value in arrays.items()}


def export_npz_state(path: Path, params: Mapping[str, Any]) -> None:
    """Export the exact named PyTorch tensor layout to a compressed NPZ."""
    arrays = {name: np.asarray(jax.device_get(value), dtype=np.float32) for name, value in params.items()}
    np.savez(path, **arrays)


def _dtype(precision: ComputePrecision) -> jnp.dtype:
    if precision == "fp32":
        return jnp.float32
    if precision == "bf16":
        return jnp.bfloat16
    raise ValueError(f"unsupported compute precision: {precision}")


def _linear(x: jax.Array, params: Mapping[str, jax.Array], prefix: str) -> jax.Array:
    weight = params[f"{prefix}.weight"].astype(x.dtype)
    bias = params[f"{prefix}.bias"].astype(x.dtype)
    return jnp.matmul(x, weight.T) + bias


def _gelu_exact(x: jax.Array) -> jax.Array:
    """Match `torch.nn.GELU(approximate='none')`, not tanh-approximate GELU."""
    half = jnp.asarray(0.5, x.dtype)
    inv_sqrt_two = jnp.asarray(1.0 / np.sqrt(2.0), x.dtype)
    return half * x * (1.0 + lax.erf(x * inv_sqrt_two))


def _gru_cell(
    x: jax.Array, hidden: jax.Array, params: Mapping[str, jax.Array], layer: int
) -> jax.Array:
    """One PyTorch GRU cell, preserving PyTorch's reset-after projection equation."""
    input_gates = jnp.matmul(x, params[f"gru.weight_ih_l{layer}"].astype(x.dtype).T)
    input_gates = input_gates + params[f"gru.bias_ih_l{layer}"].astype(x.dtype)
    hidden_gates = jnp.matmul(hidden, params[f"gru.weight_hh_l{layer}"].astype(x.dtype).T)
    hidden_gates = hidden_gates + params[f"gru.bias_hh_l{layer}"].astype(x.dtype)
    input_reset, input_update, input_new = jnp.split(input_gates, 3, axis=-1)
    hidden_reset, hidden_update, hidden_new = jnp.split(hidden_gates, 3, axis=-1)
    reset = jax.nn.sigmoid(input_reset + hidden_reset)
    update = jax.nn.sigmoid(input_update + hidden_update)
    candidate = jnp.tanh(input_new + reset * hidden_new)
    return (1.0 - update) * candidate + update * hidden


def _scan_gru(
    inputs: jax.Array,
    initial: jax.Array,
    params: Mapping[str, jax.Array],
    layer: int,
) -> tuple[jax.Array, jax.Array]:
    """Scan one GRU layer over time, returning batch-major rows and final state."""
    time_major = jnp.swapaxes(inputs, 0, 1)

    def step(hidden: jax.Array, row: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_hidden = _gru_cell(row, hidden, params, layer)
        return next_hidden, next_hidden

    final, outputs = lax.scan(step, initial, time_major)
    return jnp.swapaxes(outputs, 0, 1), final


def forward_chunk(
    params: Mapping[str, jax.Array],
    inputs: jax.Array,
    state: jax.Array | None = None,
    *,
    precision: ComputePrecision = "fp32",
) -> tuple[jax.Array, jax.Array]:
    """Run one causal input chunk and return predictions plus `[layers,batch,H]` state."""
    if inputs.ndim != 3:
        raise ValueError("inputs must have [batch,time,features] shape")
    compute_dtype = _dtype(precision)
    if inputs.shape[-1] != params["mean"].shape[0]:
        raise ValueError("input width differs from the loaded state dict")
    mean = lax.stop_gradient(params["mean"])
    std = lax.stop_gradient(params["std"])
    normalized = (inputs.astype(jnp.float32) - mean) / std
    projected = _gelu_exact(_linear(normalized.astype(compute_dtype), params, "inp"))
    batch, _, hidden = projected.shape
    layer_count = sum(1 for name in params if name.startswith("gru.weight_ih_l"))
    if layer_count < 1:
        raise ValueError("state dict has no GRU layers")
    if state is None:
        state = jnp.zeros((layer_count, batch, hidden), dtype=compute_dtype)
    elif state.shape != (layer_count, batch, hidden):
        raise ValueError("GRU state must have [layers,batch,hidden] shape")

    recurrent = projected
    next_layers: list[jax.Array] = []
    for layer in range(layer_count):
        recurrent, final = _scan_gru(recurrent, state[layer].astype(compute_dtype), params, layer)
        next_layers.append(final)
    features = jnp.concatenate((recurrent, projected), axis=-1)
    head_hidden = _gelu_exact(_linear(features, params, "head.0"))
    output = _linear(head_hidden, params, "head.2")
    return output, jnp.stack(next_layers, axis=0)


def forward_chunks(
    params: Mapping[str, jax.Array],
    inputs: jax.Array,
    chunk_rows: int,
    state: jax.Array | None = None,
    *,
    detach_between_chunks: bool = False,
    precision: ComputePrecision = "fp32",
) -> tuple[jax.Array, jax.Array]:
    """Run fixed TBPTT chunks, optionally stopping gradient through carried state."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    outputs: list[jax.Array] = []
    for start in range(0, inputs.shape[1], chunk_rows):
        if detach_between_chunks and state is not None:
            state = lax.stop_gradient(state)
        output, state = forward_chunk(
            params,
            inputs[:, start : start + chunk_rows],
            state,
            precision=precision,
        )
        outputs.append(output)
    if not outputs or state is None:
        raise ValueError("inputs must contain at least one time row")
    return jnp.concatenate(outputs, axis=1), state


def weighted_fixed_target_mse(
    prediction: jax.Array,
    fixed_target: jax.Array,
    label: jax.Array,
    row_mask: jax.Array,
) -> jax.Array:
    """Torch-compatible weighted MSE against a precomputed teacher/label target.

    `fixed_target` is already the training bridge's clipped distillation blend.
    Only its channels are compared, so auxiliary model heads (for example the
    23-output trainer head) remain outside this primary objective. The metric
    denominator follows `weighted_mse_loss`: weight each target channel by
    `abs(clip(label,-2,2)) * row_mask`, then divide by total weight.
    """
    if fixed_target.shape != label.shape:
        raise ValueError("fixed_target and label shapes must match")
    if (prediction.shape[:-1] != fixed_target.shape[:-1]
            or prediction.shape[-1] < fixed_target.shape[-1]):
        raise ValueError("prediction must contain the fixed target channels and match its batch/time shape")
    if row_mask.shape != prediction.shape[:-1]:
        raise ValueError("row_mask must match prediction batch/time dimensions")
    clipped_label = jnp.clip(label.astype(jnp.float32), -2.0, 2.0)
    clipped_target = jnp.clip(fixed_target.astype(jnp.float32), -2.0, 2.0)
    weights = jnp.abs(clipped_label) * row_mask.astype(jnp.float32)[..., None]
    main_prediction = prediction[..., : fixed_target.shape[-1]].astype(jnp.float32)
    numerator = jnp.sum(weights * (main_prediction - clipped_target) ** 2)
    denominator = jnp.maximum(jnp.sum(weights), 1e-8)
    return numerator / denominator


class AdamWState(NamedTuple):
    """Pytree optimizer state for custom FP32 AdamW moments."""

    count: jax.Array
    first: Mapping[str, jax.Array]
    second: Mapping[str, jax.Array]


def adamw_init(params: Mapping[str, jax.Array]) -> AdamWState:
    """Initialize optimizer moments for trainable tensors, excluding buffers."""
    trainable = {name: value for name, value in params.items() if name not in {"mean", "std"}}
    return AdamWState(
        jnp.asarray(0, dtype=jnp.int32),
        {name: jnp.zeros_like(value, dtype=jnp.float32) for name, value in trainable.items()},
        {name: jnp.zeros_like(value, dtype=jnp.float32) for name, value in trainable.items()},
    )


def adamw_update(
    params: Mapping[str, jax.Array],
    gradients: Mapping[str, jax.Array],
    state: AdamWState,
    *,
    learning_rate: float,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[dict[str, jax.Array], AdamWState]:
    """Apply decoupled FP32 AdamW to model parameters, preserving mean/std buffers."""
    count = state.count + 1
    first: dict[str, jax.Array] = {}
    second: dict[str, jax.Array] = {}
    updated = dict(params)
    correction1 = 1.0 - beta1**count
    correction2 = 1.0 - beta2**count
    for name, parameter in params.items():
        if name in {"mean", "std"}:
            continue
        gradient = gradients[name].astype(jnp.float32)
        moment1 = beta1 * state.first[name] + (1.0 - beta1) * gradient
        moment2 = beta2 * state.second[name] + (1.0 - beta2) * gradient**2
        normalized = (moment1 / correction1) / (jnp.sqrt(moment2 / correction2) + epsilon)
        value = parameter.astype(jnp.float32)
        updated[name] = value - learning_rate * (normalized + weight_decay * value)
        first[name], second[name] = moment1, moment2
    return updated, AdamWState(count, first, second)


@partial(jax.jit, static_argnames=("chunk_rows", "precision"))
def train_step(
    params: Mapping[str, jax.Array],
    optimizer_state: AdamWState,
    inputs: jax.Array,
    fixed_target: jax.Array,
    label: jax.Array,
    row_mask: jax.Array,
    *,
    chunk_rows: int,
    learning_rate: float,
    weight_decay: float = 0.01,
    precision: ComputePrecision = "fp32",
) -> tuple[dict[str, jax.Array], AdamWState, jax.Array]:
    """Run one causal batch update with detached TBPTT carry and fixed targets."""

    def objective(current: Mapping[str, jax.Array]) -> jax.Array:
        prediction, _ = forward_chunks(
            current,
            inputs,
            chunk_rows,
            detach_between_chunks=True,
            precision=precision,
        )
        return weighted_fixed_target_mse(prediction, fixed_target, label, row_mask)

    loss, gradients = jax.value_and_grad(objective)(params)
    updated, next_optimizer_state = adamw_update(
        params,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return updated, next_optimizer_state, loss


@partial(jax.jit, static_argnames=("precision",))
def train_chunk_step(
    params: Mapping[str, jax.Array],
    optimizer_state: AdamWState,
    inputs: jax.Array,
    recurrent_state: jax.Array | None,
    fixed_target: jax.Array,
    label: jax.Array,
    row_mask: jax.Array,
    *,
    learning_rate: float,
    weight_decay: float = 0.01,
    precision: ComputePrecision = "fp32",
) -> tuple[dict[str, jax.Array], AdamWState, jax.Array, jax.Array]:
    """Apply one chunk-level update and return a detached state for the next chunk.

    This matches the root trainer's optimizer cadence: each chunk gets its own
    weighted objective/update while recurrent values carry forward with TBPTT
    gradients stopped at the boundary.
    """

    def objective(current: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        prediction, next_state = forward_chunk(
            current, inputs, recurrent_state, precision=precision
        )
        loss = weighted_fixed_target_mse(prediction, fixed_target, label, row_mask)
        return loss, next_state

    (loss, next_state), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    updated, next_optimizer_state = adamw_update(
        params,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return updated, next_optimizer_state, lax.stop_gradient(next_state), loss
