from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax", reason="JAX is supplied by the admitted TPU/GPU benchmark runtime")
jax.config.update("jax_default_matmul_precision", "highest")

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-04/staging/jax_tpu_e465/src"))

from wnn_jax.rnn import forward_chunk, load_npz_state, weighted_fixed_target_mse  # noqa: E402

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def _paired_model(tmp_path: Path) -> tuple[RNNModel, dict[str, jax.Array]]:
    torch.manual_seed(20261009)
    torch.set_num_threads(1)
    torch_model = RNNModel(n_in=211, hidden=256, layers=2, dropout=0.0, n_out=2)
    supplied_initial = os.environ.get("WNN_PARITY_INITIAL_STATE")
    if supplied_initial:
        with np.load(supplied_initial, allow_pickle=False) as archive:
            torch_model.load_state_dict(
                {name: torch.from_numpy(archive[name]) for name in archive.files}
            )
    state_path = tmp_path / "initial_state.npz"
    np.savez(
        state_path,
        **{
            name: tensor.detach().cpu().numpy() for name, tensor in torch_model.state_dict().items()
        },
    )
    return torch_model.eval(), load_npz_state(state_path, 211, 256, 2, 2)


def _features() -> np.ndarray:
    supplied = os.environ.get("WNN_PARITY_PROFILE_INPUTS")
    if supplied:
        with np.load(supplied, allow_pickle=False) as archive:
            return archive["inputs"][:2, :11].copy()
    raw = np.random.default_rng(20261009).normal(size=(2, 11, 112)).astype(np.float32)
    return np.concatenate((raw, event_features(raw)), axis=-1)


def test_211_input_gru256_forward_and_chunk_state_match_torch(tmp_path: Path) -> None:
    torch_model, jax_params = _paired_model(tmp_path)
    inputs = _features()
    torch_inputs = torch.from_numpy(inputs)

    with torch.no_grad():
        torch_full, torch_full_state = torch_model(torch_inputs)
        torch_first, torch_state = torch_model(torch_inputs[:, :5])
        torch_second, torch_final_state = torch_model(torch_inputs[:, 5:], torch_state)
    jax_full, jax_full_state = forward_chunk(jax_params, jax.numpy.asarray(inputs))
    jax_first, jax_state = forward_chunk(jax_params, jax.numpy.asarray(inputs[:, :5]))
    jax_second, jax_final_state = forward_chunk(
        jax_params, jax.numpy.asarray(inputs[:, 5:]), jax_state
    )

    np.testing.assert_allclose(np.asarray(jax_full), torch_full.numpy(), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        np.asarray(jax_full_state), torch_full_state.numpy(), rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        np.concatenate((np.asarray(jax_first), np.asarray(jax_second)), axis=1),
        torch.cat((torch_first, torch_second), dim=1).numpy(),
        rtol=3e-5,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax_final_state), torch_final_state.numpy(), rtol=3e-5, atol=3e-6
    )


def test_jax_checkpoint_reload_state_reset_and_prefix_causality(tmp_path: Path) -> None:
    torch_model, original_params = _paired_model(tmp_path)
    inputs = _features()
    checkpoint = tmp_path / "jax_state.npz"
    np.savez(
        checkpoint,
        **{name: np.asarray(value) for name, value in original_params.items()},
    )
    reloaded = load_npz_state(checkpoint, 211, 256, 2, 2)
    original, original_state = forward_chunk(original_params, jax.numpy.asarray(inputs))
    restored, restored_state = forward_chunk(reloaded, jax.numpy.asarray(inputs))
    np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))
    np.testing.assert_array_equal(np.asarray(original_state), np.asarray(restored_state))

    # A new sequence must start from zero state, not the preceding sequence's carry.
    first, _ = forward_chunk(reloaded, jax.numpy.asarray(inputs[:1]))
    second, _ = forward_chunk(reloaded, jax.numpy.asarray(inputs[1:]))
    paired, _ = forward_chunk(reloaded, jax.numpy.asarray(inputs))
    np.testing.assert_allclose(
        np.concatenate((np.asarray(first), np.asarray(second))),
        np.asarray(paired),
        rtol=3e-5,
        atol=3e-6,
    )
    explicit_reset, _ = forward_chunk(
        reloaded, jax.numpy.asarray(inputs), jax.numpy.zeros_like(original_state)
    )
    np.testing.assert_array_equal(np.asarray(explicit_reset), np.asarray(paired))

    changed_raw = _features()[:, :, :112].copy()
    changed_raw[:, 7:] += 0.5
    changed = np.concatenate((changed_raw, event_features(changed_raw)), axis=-1)
    changed_prediction, _ = forward_chunk(reloaded, jax.numpy.asarray(changed))
    np.testing.assert_allclose(
        np.asarray(original)[:, :7], np.asarray(changed_prediction)[:, :7], rtol=0.0, atol=0.0
    )

    # Torch remains the feature/model oracle for the exact cached-input path.
    with torch.no_grad():
        expected, _ = torch_model(torch.from_numpy(inputs))
    np.testing.assert_allclose(np.asarray(restored), expected.numpy(), rtol=3e-5, atol=3e-6)


def test_weighted_mse_matches_torch_on_synthetic_rows() -> None:
    from wnn.nn import weighted_mse_loss

    prediction = np.asarray([[[0.25, -0.5], [1.4, -1.8], [0.0, 0.3]]], dtype=np.float32)
    target = np.asarray([[[0.2, -0.4], [1.0, -1.2], [0.0, 0.3]]], dtype=np.float32)
    mask = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
    torch_prediction = torch.from_numpy(prediction)
    torch_target = torch.from_numpy(target)
    torch_mask = torch.from_numpy(mask)
    expected = weighted_mse_loss(torch_prediction, torch_target, torch_mask)
    observed = weighted_fixed_target_mse(
        jax.numpy.asarray(prediction),
        jax.numpy.asarray(target),
        jax.numpy.asarray(target),
        jax.numpy.asarray(mask),
    )
    np.testing.assert_allclose(np.asarray(observed), expected.item(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("loss_kind", ["wmse", "huber"])
def test_matched_loss_gradient_adamw_updates_and_detached_carry(
    tmp_path: Path, loss_kind: str
) -> None:
    """Match two optimizer clocks and chunk carry against separate Torch autograd."""
    from wnn_jax.rnn import adamw_init

    from wnn.nn import weighted_huber_loss, weighted_mse_loss

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jax_step import train_chunk_step, weighted_loss_terms

    model, params = _paired_model(tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    jax_optimizer = adamw_init(params)
    inputs = _features()
    target = np.random.default_rng(19).normal(size=(2, 11, 2)).astype(np.float32) * 3
    mask = np.ones((2, 11), dtype=np.float32)
    mask[:, :2] = 0
    torch_state = None
    jax_state = None
    for index, (start, stop, lr, beta1) in enumerate(((0, 5, 0.001, 0.95), (5, 11, 0.0004, 0.85))):
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[0]["betas"] = (beta1, 0.999)
        optimizer.zero_grad(set_to_none=True)
        tx = torch.from_numpy(inputs[:, start:stop])
        ty = torch.from_numpy(target[:, start:stop])
        tm = torch.from_numpy(mask[:, start:stop])
        prediction, next_torch_state = model(tx, torch_state)
        loss = (
            weighted_mse_loss(prediction, ty, tm)
            if loss_kind == "wmse"
            else weighted_huber_loss(prediction, ty, tm, delta=1.0)
        )
        loss.backward()
        expected_gradients = {
            name: tensor.grad.detach().numpy().copy() for name, tensor in model.named_parameters()
        }

        def independent_objective(
            current: dict[str, jax.Array],
            start: int = start,
            stop: int = stop,
            jax_state: jax.Array | None = jax_state,
        ) -> jax.Array:
            jp, _ = forward_chunk(current, jax.numpy.asarray(inputs[:, start:stop]), jax_state)
            numerator, denominator = weighted_loss_terms(
                jp,
                jax.numpy.asarray(target[:, start:stop]),
                jax.numpy.asarray(mask[:, start:stop]),
                loss_kind=loss_kind,
                huber_delta=1.0,
            )
            return numerator / jax.numpy.maximum(denominator, 1e-8)

        gradients = jax.grad(independent_objective)(params)
        for name, expected in expected_gradients.items():
            np.testing.assert_allclose(
                np.asarray(gradients[name]),
                expected,
                rtol=2e-3,
                atol=3e-6,
                err_msg=f"gradient {index} {name}",
            )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        result = train_chunk_step(
            params,
            jax_optimizer,
            jax.numpy.asarray(inputs[:, start:stop]),
            jax_state,
            jax.numpy.asarray(target[:, start:stop]),
            jax.numpy.asarray(mask[:, start:stop]),
            learning_rate=lr,
            beta1=beta1,
            loss_kind=loss_kind,
            precision="fp32",
        )
        params, jax_optimizer, jax_state, observed_loss, observed_norm, _, finite = result
        jax.block_until_ready(result)
        assert bool(finite)
        np.testing.assert_allclose(np.asarray(observed_loss), loss.item(), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(np.asarray(observed_norm), norm.item(), rtol=5e-4, atol=3e-6)
        for name, expected in model.state_dict().items():
            np.testing.assert_allclose(
                np.asarray(params[name]),
                expected.numpy(),
                rtol=5e-4,
                atol=2e-5,
                err_msg=f"AdamW {index} {name}",
            )
        np.testing.assert_allclose(
            np.asarray(jax_state), next_torch_state.detach().numpy(), rtol=5e-4, atol=2e-5
        )
        torch_state = next_torch_state.detach()


def test_resume_checkpoint_drops_replica_axis_and_preserves_optimizer(tmp_path: Path) -> None:
    """Use the real checkpoint writer/reader instead of a synthetic NPZ schema."""
    from wnn_jax.rnn import adamw_init

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tpu_worker import _load_checkpoint, _replicate, _save_resume_checkpoint

    _, params = _paired_model(tmp_path)
    optimizer = adamw_init(params)
    devices = jax.local_devices()
    _save_resume_checkpoint(
        tmp_path,
        _replicate(params, devices),
        _replicate(optimizer, devices),
        contract_sha="actual-fixture",
        loss_kind="wmse",
        precision="fp32",
        next_epoch=0,
        next_batch=1,
        updates_completed=10,
    )
    restored, restored_optimizer, metadata = _load_checkpoint(
        tmp_path / "training_checkpoint_latest.npz",
        tmp_path / "training_checkpoint_latest.json",
        "actual-fixture",
        "wmse",
    )
    assert metadata["updates_completed"] == 10
    for name, value in params.items():
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(value))
    assert np.asarray(restored_optimizer.count).item() == np.asarray(optimizer.count).item()
    for name, value in optimizer.first.items():
        np.testing.assert_array_equal(np.asarray(restored_optimizer.first[name]), np.asarray(value))
        np.testing.assert_array_equal(
            np.asarray(restored_optimizer.second[name]), np.asarray(optimizer.second[name])
        )
