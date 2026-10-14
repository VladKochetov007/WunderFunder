from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("wnn_native")

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def _weights(model: RNNModel) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for name, value in model.state_dict().items()
    }


def _raw(rows: int, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 112)).astype(np.float32)
    hold = rng.random(size=values.shape) < 0.55
    hold[0] = False
    for row in range(1, rows):
        values[row, hold[row]] = values[row - 1, hold[row]]
    return values


def _features(raw: np.ndarray) -> np.ndarray:
    return np.concatenate((raw, event_features(raw)), axis=-1)


def _python_sequence(model: RNNModel, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        prediction, _ = model(torch.from_numpy(features)[None])
    return prediction[0, :, :2].cpu().numpy()


@pytest.mark.parametrize("shape", [(8, 8, 6, 1), (12, 8, 6, 1), (12, 8, 6, 2)])
def test_native_full_callback_matches_python_chunks_causality_and_reset(shape) -> None:
    encoder, recurrent, readout, head_layers = shape
    torch.manual_seed(2202)
    model = RNNModel(
        211,
        hidden=recurrent,
        layers=2,
        encoder_width=encoder,
        recurrent_width=recurrent,
        readout_width=readout,
        head_layers=head_layers,
    ).eval()
    weights = _weights(model)
    raw = _raw(137)
    features = _features(raw)
    expected = _python_sequence(model, features)

    exact = native.Predictor(weights, precision="f32", fast_activations=False)
    actual = np.stack([exact.step(row) for row in raw])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    gru_stepper = native.GruStepper(weights)
    gru_output = np.stack([gru_stepper.step(row) for row in features])
    np.testing.assert_allclose(gru_output, expected, rtol=2e-5, atol=2e-5)
    first_hidden = exact.hidden_state.copy()
    exact.reset()
    replay = np.stack([exact.step(row) for row in raw])
    np.testing.assert_array_equal(replay, actual)
    np.testing.assert_array_equal(exact.hidden_state, first_hidden)

    cut = 53
    direct = native.ModelStepper(weights, fast_activations=False, precision="f32")
    direct_output = direct.run(features)
    chunked = native.ModelStepper(weights, fast_activations=False, precision="f32")
    first = chunked.run(features[:cut])
    second = chunked.run(features[cut:])
    np.testing.assert_array_equal(np.concatenate((first, second)), direct_output)

    changed = raw.copy()
    changed[cut + 1 :] += np.float32(30.0)
    causal = native.Predictor(weights, precision="f32", fast_activations=False)
    counterfactual = native.Predictor(weights, precision="f32", fast_activations=False)
    left = np.stack([causal.step(row) for row in raw])
    right = np.stack([counterfactual.step(row) for row in changed])
    np.testing.assert_array_equal(left[: cut + 1], right[: cut + 1])

    with torch.no_grad():
        full, _ = model(torch.from_numpy(features)[None])
        first_chunk, state = model(torch.from_numpy(features[:cut])[None])
        second_chunk, _ = model(torch.from_numpy(features[cut:])[None], state)
    np.testing.assert_allclose(
        torch.cat((first_chunk, second_chunk), dim=1), full, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "shape",
    [(12, 8, 6, 2), (8, 8, 6, 1), (8, 8, 20, 1)],
)
def test_native_q8vnni_packs_width_variants_and_reports_actual_storage(shape) -> None:
    encoder, recurrent, readout, head_layers = shape
    torch.manual_seed(2203)
    model = RNNModel(
        211,
        hidden=recurrent,
        layers=2,
        encoder_width=encoder,
        recurrent_width=recurrent,
        readout_width=readout,
        head_layers=head_layers,
    ).eval()
    weights = _weights(model)
    raw = _raw(137, seed=42)
    packed = native.Predictor(weights, precision="q8vnni_portable", fast_activations=False)
    assert packed.precision == "q8vnni_portable"
    prediction = np.stack([packed.step(row) for row in raw])
    assert prediction.shape == (137, 2)
    assert np.isfinite(prediction).all()
    reference_model = native.Predictor(weights, precision="f32", fast_activations=False)
    reference = np.stack([reference_model.step(row) for row in raw])
    np.testing.assert_allclose(prediction, reference, rtol=0.05, atol=0.05)

    codes = packed.quantization_codes()
    assert ("head.extra.weight.codes" in codes) == (head_layers == 2)
    expected_packed_bytes = 0
    for name in sorted(key.removesuffix(".codes") for key in codes if key.endswith(".codes")):
        matrix = codes[f"{name}.codes"]
        rows, columns = matrix.shape
        super_blocks = (rows + 63) // 64
        column_groups = (columns + 3) // 4
        padded_rows = super_blocks * 64
        expected_packed_bytes += super_blocks * column_groups * 256 + padded_rows * 8
    storage = packed.storage_bytes
    assert storage["dynamic_q8_weights"] == expected_packed_bytes
    assert storage["total_streamed_weights"] == packed.weight_bytes
    assert storage["dynamic_q8_weights"] < storage["total_streamed_weights"]

    packed.reset()
    replay = np.stack([packed.step(row) for row in raw])
    np.testing.assert_array_equal(replay, prediction)
