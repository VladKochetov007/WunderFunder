from __future__ import annotations

import numpy as np
import pytest
import torch
import wnn_native

from wnn.features import StreamingEventFeatures
from wnn.min_gru import MinGRUModel


def model_fixture() -> tuple[MinGRUModel, dict[str, np.ndarray]]:
    torch.manual_seed(9027)
    model = MinGRUModel(
        211,
        hidden=12,
        layers=2,
        encoder_width=12,
        recurrent_width=8,
        readout_width=6,
    ).eval()
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.1, 0.1, 211))
        model.std.copy_(torch.linspace(0.7, 1.3, 211))
    weights = {
        name: np.ascontiguousarray(value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }
    return model, weights


def engineered_rows(raw: np.ndarray) -> np.ndarray:
    features = StreamingEventFeatures()
    event = np.empty((raw.shape[0], 99), dtype=np.float32)
    for index, row in enumerate(raw):
        event[index] = features.update(row)
    return np.concatenate((raw, event), axis=1)


def test_native_model_stepper_matches_python_and_chunks_reset() -> None:
    model, weights = model_fixture()
    inputs = np.random.default_rng(324).normal(0.0, 0.15, (137, 211)).astype(np.float32)
    with torch.inference_mode():
        expected, expected_state = model(torch.from_numpy(inputs[None]))
    native = wnn_native.MinGruStepper(weights, fast_activations=False)
    actual = native.run(inputs)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=4e-5, atol=3e-6)
    np.testing.assert_allclose(
        native.hidden_state, expected_state[:, 0].numpy(), rtol=4e-5, atol=3e-6
    )

    chunked = wnn_native.MinGruStepper(weights, fast_activations=False)
    first = chunked.run(inputs[:99])
    second = chunked.run(inputs[99:])
    np.testing.assert_array_equal(np.concatenate((first, second)), actual)
    np.testing.assert_array_equal(chunked.hidden_state, native.hidden_state)
    chunked.reset()
    np.testing.assert_array_equal(chunked.run(inputs), actual)


def test_native_raw_callback_matches_python_causal_features_and_reset() -> None:
    model, weights = model_fixture()
    raw = np.random.default_rng(615).normal(0.0, 0.2, (2048, 112)).astype(np.float32)
    inputs = engineered_rows(raw)
    with torch.inference_mode():
        expected, expected_state = model(torch.from_numpy(inputs[None]))

    predictor = wnn_native.MinGruPredictor(weights, fast_activations=False)
    actual = predictor.run(raw)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=5e-5, atol=4e-6)
    np.testing.assert_allclose(
        predictor.hidden_state, expected_state[:, 0].numpy(), rtol=5e-5, atol=4e-6
    )

    direct = wnn_native.MinGruStepper(weights, fast_activations=False).run(inputs)
    np.testing.assert_allclose(actual, direct, rtol=0, atol=2e-7)

    row_callback = wnn_native.MinGruPredictor(weights, fast_activations=False)
    stepped = np.stack([row_callback.step(row) for row in raw[:64]])
    np.testing.assert_allclose(stepped, actual[:64], rtol=0, atol=2e-7)
    predictor.reset()
    np.testing.assert_array_equal(predictor.run(raw), actual)


def test_native_min_gru_fast_and_q8_modes_are_finite_and_resettable() -> None:
    _, weights = model_fixture()
    raw = np.random.default_rng(2718).normal(0.0, 0.2, (1024, 112)).astype(np.float32)
    exact = wnn_native.MinGruPredictor(
        weights, fast_activations=False, precision="f32"
    ).run(raw)
    for precision in ("f32", "q8vnni", "q8portable"):
        predictor = wnn_native.MinGruPredictor(
            weights, fast_activations=True, precision=precision
        )
        actual = predictor.run(raw)
        assert np.isfinite(actual).all()
        assert np.isfinite(predictor.hidden_state).all()
        assert predictor.precision == precision
        assert predictor.fast_activations
        if precision == "f32":
            assert np.max(np.abs(actual - exact)) < 1e-5
        predictor.reset()
        np.testing.assert_array_equal(predictor.run(raw), actual)


def test_native_min_gru_rejects_non_211_and_wrong_output_architecture() -> None:
    _, weights = model_fixture()
    with pytest.raises(ValueError, match="211"):
        wnn_native.MinGruStepper({**weights, "mean": np.zeros(210, np.float32)})
    invalid = dict(weights)
    invalid["head.2.weight"] = np.zeros((3, 6), np.float32)
    invalid["head.2.bias"] = np.zeros(3, np.float32)
    with pytest.raises(ValueError, match="head.2"):
        wnn_native.MinGruStepper(invalid)
