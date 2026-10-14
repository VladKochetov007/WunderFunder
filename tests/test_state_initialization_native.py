"""Export and callback parity checks for native causal state initialization."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.features import StreamingEventFeatures
from wnn.nn import RNNModel
from wnn.state_initialization import InitializedGRU, export_native_initializer

native = pytest.importorskip("wnn_native")


def _model(mode: str) -> InitializedGRU:
    torch.manual_seed(197)
    wrapper = InitializedGRU(RNNModel(211, hidden=8, layers=2), mode, residual_bound=0.7)
    with torch.no_grad():
        wrapper.backbone.mean.copy_(torch.linspace(-0.1, 0.1, 211))
        wrapper.backbone.std.copy_(torch.linspace(0.7, 1.3, 211))
        if wrapper.constant is not None:
            wrapper.constant.fill_(0.4)
        if wrapper.predictor is not None:
            wrapper.predictor[0].weight.normal_(std=0.025)
            wrapper.predictor[0].bias.normal_(std=0.02)
            wrapper.predictor[-1].weight.normal_(std=0.035)
            wrapper.predictor[-1].bias.fill_(0.2)
    wrapper.eval()
    return wrapper


def _weights(wrapper: InitializedGRU) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        for name, tensor in wrapper.backbone.state_dict().items()
    }


def _features(raw: np.ndarray) -> np.ndarray:
    stream = StreamingEventFeatures()
    return np.stack([np.concatenate((row, stream.update(row))) for row in raw], axis=0).astype(
        np.float32
    )


def _raw_rows(count: int = 137) -> np.ndarray:
    random = np.random.default_rng(5921)
    values = random.normal(0.0, 0.15, size=(count, 112)).astype(np.float32)
    values[:, 0] += np.arange(count, dtype=np.float32) * 0.002
    values[:, 11] -= np.arange(count, dtype=np.float32) * 0.001
    return np.ascontiguousarray(values)


def _native_predictor(
    wrapper: InitializedGRU,
    *,
    initializer: bool = True,
    fast_activations: bool = False,
    precision: str = "f32",
):
    configs = [export_native_initializer(wrapper)] if initializer else None
    return native.EnsemblePredictor(
        [_weights(wrapper)],
        [1.0],
        fast_activations=fast_activations,
        precision=precision,
        state_initializers=configs,
    )


def _native_run(model, raw: np.ndarray, seq_ix: int = 1) -> np.ndarray:
    outputs = []
    for row in raw:
        point = SimpleNamespace(seq_ix=seq_ix, state=row, need_prediction=True)
        prediction = model.predict(point)
        assert prediction is not None
        outputs.append(np.asarray(prediction, dtype=np.float32))
    return np.stack(outputs)


@pytest.mark.parametrize("mode", ["zero", "constant", "first_row", "prefix"])
def test_native_callback_matches_exact_python_wrapper_and_reset(
    mode: str, record_property: Callable[[str, object], None]
) -> None:
    wrapper = _model(mode)
    raw = _raw_rows()
    features = torch.from_numpy(_features(raw))[None]
    with torch.no_grad():
        full_reference, reference_state = wrapper(features)
        state = None
        chunks = []
        for begin, end in [(0, 1), (1, 37), (37, 98), (98, 99), (99, 100), (100, len(raw))]:
            output, state = wrapper(features[:, begin:end], state)
            chunks.append(output)
    torch.testing.assert_close(torch.cat(chunks, dim=1), full_reference, rtol=1e-5, atol=1e-6)

    native_model = _native_predictor(wrapper)
    actual = _native_run(native_model, raw)
    prediction_error = actual - full_reference[0].numpy()
    np.testing.assert_allclose(
        actual,
        full_reference[0].numpy(),
        rtol=3e-4,
        atol=3e-5,
        err_msg=f"native exact-activation mismatch for {mode}",
    )
    np.testing.assert_allclose(
        native_model.hidden_states[0],
        reference_state.hidden[:, 0].numpy(),
        rtol=4e-4,
        atol=4e-5,
    )
    hidden_error = native_model.hidden_states[0] - reference_state.hidden[:, 0].numpy()
    record_property("prediction_max_abs_error", float(np.max(np.abs(prediction_error))))
    record_property("prediction_rms_error", float(np.sqrt(np.mean(prediction_error**2))))
    record_property("hidden_max_abs_error", float(np.max(np.abs(hidden_error))))

    final_hidden = native_model.hidden_states[0].copy()
    replay = _native_run(native_model, raw, seq_ix=2)
    np.testing.assert_array_equal(replay, actual)
    np.testing.assert_array_equal(native_model.hidden_states[0], final_hidden)
    if mode == "zero":
        baseline = _native_predictor(wrapper, initializer=False)
        baseline_output = _native_run(baseline, raw)
        np.testing.assert_array_equal(baseline_output, actual)
        np.testing.assert_array_equal(baseline.hidden_states[0], final_hidden)
        record_property("zero_arm_baseline_max_abs_error", 0.0)


@pytest.mark.parametrize("mode", ["constant", "first_row", "prefix"])
@pytest.mark.parametrize("precision", ["q8vnni_portable", "q8x_portable", "q16x_portable"])
def test_quantized_state_transport_is_resettable_and_affects_the_intended_rows(
    mode: str, precision: str, record_property: Callable[[str, object], None]
) -> None:
    wrapper = _model(mode)
    raw = _raw_rows(125)
    initialized = _native_predictor(wrapper, precision=precision)
    baseline = _native_predictor(wrapper, initializer=False, precision=precision)
    actual = _native_run(initialized, raw)
    control = _native_run(baseline, raw)
    record_property(
        "quantized_callback_max_abs_delta_vs_zero", float(np.max(np.abs(actual - control)))
    )
    first_affected = 0 if mode in ("constant", "first_row") else 99
    assert np.any(actual[first_affected:] != control[first_affected:])
    final_hidden = initialized.hidden_states[0].copy()
    replay = _native_run(initialized, raw, seq_ix=2)
    np.testing.assert_array_equal(replay, actual)
    np.testing.assert_array_equal(initialized.hidden_states[0], final_hidden)
    assert np.isfinite(initialized.hidden_states[0]).all()


def test_prefix_is_causal_through_row_99_and_native_fast_activation_is_separate(
    record_property: Callable[[str, object], None],
) -> None:
    wrapper = _model("prefix")
    raw = _raw_rows(137)
    changed = raw.copy()
    changed[100:] += np.float32(40.0)
    reference = _native_predictor(wrapper)
    changed_model = _native_predictor(wrapper)
    original = _native_run(reference, raw)
    altered = _native_run(changed_model, changed)
    np.testing.assert_array_equal(original[:100], altered[:100])

    fast = _native_predictor(wrapper, fast_activations=True)
    fast_output = _native_run(fast, raw)
    assert np.isfinite(fast_output).all()
    fast_error = fast_output - original
    record_property("fast_vs_exact_max_abs_delta", float(np.max(np.abs(fast_error))))
    record_property("fast_vs_exact_rms_delta", float(np.sqrt(np.mean(fast_error**2))))
    assert np.max(np.abs(fast_error)) < 0.02


def test_native_initializer_export_preserves_float32_parameter_arrays() -> None:
    for mode in ("zero", "constant", "first_row", "prefix"):
        wrapper = _model(mode)
        config = export_native_initializer(wrapper)
        assert config["mode"] == mode
        for _key, value in config.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == np.float32
                assert value.flags.c_contiguous
        if wrapper.constant is not None:
            np.testing.assert_array_equal(
                config["constant"],
                wrapper.constant.detach().cpu().numpy()[:, 0, :],
            )
