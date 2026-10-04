from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from wnn.nn import RNNModel
from wnn_jax.rnn import (
    adamw_init,
    forward_chunk,
    train_chunk_step,
    weighted_fixed_target_mse,
)


def _model() -> RNNModel:
    torch.manual_seed(314159)
    model = RNNModel(n_in=7, hidden=5, layers=2, dropout=0.0, n_out=23).eval()
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.2, 0.2, 7))
        model.std.copy_(torch.linspace(0.8, 1.2, 7))
    return model


def _jax_params(model: RNNModel) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }


def _loss_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    clipped_label = label.float().clamp(-2.0, 2.0)
    clipped_target = target.float().clamp(-2.0, 2.0)
    weights = clipped_label.abs() * row_mask.float()[..., None]
    error = prediction[..., : target.shape[-1]].float() - clipped_target
    return (weights * error.square()).sum() / weights.sum().clamp_min(1e-8)


def _assert_mapping_close(
    actual: dict[str, jax.Array],
    expected: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> None:
    assert set(actual) == set(expected)
    for name in actual:
        np.testing.assert_allclose(
            np.asarray(actual[name]),
            expected[name].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=name,
        )


def test_nonzero_initial_state_and_unequal_chunks_match_torch() -> None:
    model = _model()
    params = _jax_params(model)
    generator = torch.Generator().manual_seed(21)
    values = torch.randn((2, 11, 7), generator=generator)
    initial_state = torch.randn((2, 2, 5), generator=generator) * 0.2

    with torch.no_grad():
        expected, expected_state = model(values, initial_state)

    outputs: list[jax.Array] = []
    state = jnp.asarray(initial_state.numpy())
    for start, stop in ((0, 5), (5, 8), (8, 11)):
        output, state = forward_chunk(
            params,
            jnp.asarray(values[:, start:stop].numpy()),
            state,
        )
        outputs.append(output)

    np.testing.assert_allclose(
        np.asarray(jnp.concatenate(outputs, axis=1)),
        expected.numpy(),
        rtol=2e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state), expected_state.numpy(), rtol=2e-5, atol=5e-6
    )


def test_two_disjoint_chunk_updates_match_torch_gradients_and_adamw() -> None:
    model = _model()
    params = _jax_params(model)
    optimizer = adamw_init(params)
    generator = torch.Generator().manual_seed(27)
    inputs = torch.randn((2, 11, 7), generator=generator)
    targets = torch.randn((2, 11, 2), generator=generator).clamp(-1.5, 1.5)
    labels = torch.randn((2, 11, 2), generator=generator).clamp(-1.5, 1.5)
    row_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]
    )
    initial_state = torch.randn((2, 2, 5), generator=generator) * 0.1
    torch_state: torch.Tensor | None = initial_state
    jax_state: jax.Array | None = jnp.asarray(initial_state.numpy())
    first: dict[str, torch.Tensor] = {
        name: torch.zeros_like(value)
        for name, value in model.state_dict().items()
        if name not in {"mean", "std"}
    }
    second = {name: torch.zeros_like(value) for name, value in first.items()}
    learning_rate = 7e-4
    beta1, beta2, epsilon, weight_decay = 0.9, 0.999, 1e-8, 0.01

    for count, (start, stop) in enumerate(((0, 5), (5, 11)), start=1):
        chunk_x = inputs[:, start:stop]
        chunk_target = targets[:, start:stop]
        chunk_label = labels[:, start:stop]
        chunk_mask = row_mask[:, start:stop]
        model.zero_grad(set_to_none=True)
        torch_prediction, torch_next_state = model(chunk_x, torch_state)
        torch_loss = _loss_torch(
            torch_prediction, chunk_target, chunk_label, chunk_mask
        )
        torch_loss.backward()

        jax_x = jnp.asarray(chunk_x.numpy())
        jax_target = jnp.asarray(chunk_target.numpy())
        jax_label = jnp.asarray(chunk_label.numpy())
        jax_mask = jnp.asarray(chunk_mask.numpy())

        def objective(current: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
            prediction, next_state = forward_chunk(
                current, jax_x, jax_state
            )
            return (
                weighted_fixed_target_mse(
                    prediction, jax_target, jax_label, jax_mask
                ),
                next_state,
            )

        (jax_loss, jax_next_state), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        np.testing.assert_allclose(
            float(jax_loss), float(torch_loss.detach()), rtol=2e-5, atol=5e-7
        )

        torch_gradients: dict[str, torch.Tensor] = {
            "mean": torch.zeros_like(model.mean),
            "std": torch.zeros_like(model.std),
        }
        for name, value in model.named_parameters():
            torch_gradients[name] = value.grad.detach()
        _assert_mapping_close(gradients, torch_gradients, rtol=5e-4, atol=2e-5)
        np.testing.assert_allclose(
            np.asarray(jax_next_state),
            torch_next_state.detach().numpy(),
            rtol=2e-5,
            atol=5e-6,
        )

        updated, next_optimizer, returned_state, returned_loss = train_chunk_step(
            params,
            optimizer,
            jax_x,
            jax_state,
            jax_target,
            jax_label,
            jax_mask,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        np.testing.assert_allclose(
            float(returned_loss), float(torch_loss.detach()), rtol=2e-5, atol=5e-7
        )
        np.testing.assert_array_equal(
            np.asarray(returned_state), np.asarray(jax_next_state)
        )

        with torch.no_grad():
            for name, value in model.named_parameters():
                gradient = value.grad.detach()
                first[name].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second[name].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                corrected_first = first[name] / (1.0 - beta1**count)
                corrected_second = second[name] / (1.0 - beta2**count)
                normalized = corrected_first / (corrected_second.sqrt() + epsilon)
                value.add_(normalized + weight_decay * value, alpha=-learning_rate)

        _assert_mapping_close(updated, model.state_dict(), rtol=3e-4, atol=3e-6)
        for name in first:
            np.testing.assert_allclose(
                np.asarray(next_optimizer.first[name]),
                first[name].numpy(),
                rtol=5e-4,
                atol=3e-6,
                err_msg=f"first moment {name}",
            )
            np.testing.assert_allclose(
                np.asarray(next_optimizer.second[name]),
                second[name].numpy(),
                rtol=5e-4,
                atol=3e-6,
                err_msg=f"second moment {name}",
            )
            assert next_optimizer.first[name].dtype == jnp.float32
            assert next_optimizer.second[name].dtype == jnp.float32
        assert int(next_optimizer.count) == count
        assert all(value.dtype == jnp.float32 for value in updated.values())
        np.testing.assert_array_equal(np.asarray(updated["mean"]), model.mean.numpy())
        np.testing.assert_array_equal(np.asarray(updated["std"]), model.std.numpy())

        params, optimizer = updated, next_optimizer
        torch_state = torch_next_state.detach()
        jax_state = returned_state


def test_23_output_loss_ignores_auxiliary_channels() -> None:
    predictions = jax.random.normal(jax.random.PRNGKey(41), (2, 5, 23))
    targets = jax.random.normal(jax.random.PRNGKey(42), (2, 5, 2))
    labels = jax.random.normal(jax.random.PRNGKey(43), (2, 5, 2))
    mask = jnp.asarray([[1.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0, 1.0]])
    loss = weighted_fixed_target_mse(predictions, targets, labels, mask)
    changed_aux = predictions.at[..., 2:].set(1e4)
    changed_loss = weighted_fixed_target_mse(changed_aux, targets, labels, mask)
    np.testing.assert_array_equal(np.asarray(loss), np.asarray(changed_loss))

    prediction_gradient = jax.grad(
        lambda value: weighted_fixed_target_mse(value, targets, labels, mask)
    )(predictions)
    np.testing.assert_array_equal(np.asarray(prediction_gradient[..., 2:]), 0.0)


def test_bfloat16_forward_tracks_torch_autocast_with_weaker_tolerance() -> None:
    model = _model()
    params = _jax_params(model)
    generator = torch.Generator().manual_seed(45)
    values = torch.randn((2, 11, 7), generator=generator)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected, expected_state = model(values)
    actual, actual_state = forward_chunk(
        params, jnp.asarray(values.numpy()), precision="bf16"
    )

    # BF16 rounds recurrent activations and parameters, so this gate is looser
    # than the FP32 parity tolerances. It does not establish TPU parity.
    np.testing.assert_allclose(
        np.asarray(actual), expected.float().numpy(), rtol=3e-2, atol=1.5e-2
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        expected_state.float().numpy(),
        rtol=3e-2,
        atol=1.5e-2,
    )
