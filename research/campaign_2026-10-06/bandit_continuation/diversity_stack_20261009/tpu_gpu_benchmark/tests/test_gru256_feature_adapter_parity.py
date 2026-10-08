from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax", reason="JAX is supplied by the admitted TPU/GPU benchmark runtime")

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
    state_path = tmp_path / "initial_state.npz"
    np.savez(
        state_path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in torch_model.state_dict().items()
        },
    )
    return torch_model.eval(), load_npz_state(state_path, 211, 256, 2, 2)


def _features() -> np.ndarray:
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
    np.testing.assert_allclose(np.concatenate((np.asarray(first), np.asarray(second))),
                               np.asarray(paired), rtol=0.0, atol=0.0)

    changed_raw = _features()[:, :, :112].copy()
    changed_raw[:, 7:] += 0.5
    changed = np.concatenate((changed_raw, event_features(changed_raw)), axis=-1)
    changed_prediction, _ = forward_chunk(reloaded, jax.numpy.asarray(changed))
    np.testing.assert_allclose(np.asarray(original)[:, :7], np.asarray(changed_prediction)[:, :7],
                               rtol=0.0, atol=0.0)

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
