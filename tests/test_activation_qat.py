from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch

from wnn.activation_qat import (
    ActivationQATModel,
    activation_qat_forward,
    quantize_activation_a8,
)
from wnn.nn import RNNModel
from wnn.quantization import quantize_model
from wnn.training_qat import configure_qat


def test_a8_codes_and_scales_match_native_scalar_rule() -> None:
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [2.5, 1.5, 0.5], [tiny, 0.0, 0.0]],
        dtype=np.float32,
    )
    codes, scales = quantize_activation_a8(torch.from_numpy(values))
    expected_codes = np.zeros(values.shape, dtype=np.int8)
    expected_scales = np.zeros(values.shape[:-1], dtype=np.float32)
    for row, vector in enumerate(values):
        maximum = np.max(np.abs(vector))
        if maximum > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                inverse = np.float32(127) / maximum
                products = vector * inverse
                clamped = np.fmin(np.fmax(products, -127), 127)
            expected_codes[row] = np.rint(clamped).astype(np.int8)
            expected_scales[row] = maximum / np.float32(127)
    np.testing.assert_array_equal(codes.numpy(), expected_codes)
    np.testing.assert_array_equal(scales.numpy(), expected_scales)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_a8_quantizer_rejects_nonfinite_boundary_inputs(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite float32"):
        quantize_activation_a8(torch.tensor([[0.5, invalid]], dtype=torch.float32))


def _chunked_forward(
    model: RNNModel, values: torch.Tensor, initial: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial
    outputs = []
    for start, end in ((0, 4), (4, values.shape[1])):
        prediction, state = activation_qat_forward(
            model, values[:, start:end], state, quantize_activations=False
        )
        outputs.append(prediction)
    return torch.cat(outputs, dim=1), state


def test_scan_forward_and_backward_match_fused_gru_with_chunked_initial_state() -> None:
    torch.manual_seed(67013)
    reference = RNNModel(4, hidden=5, layers=2).eval()
    scanned = copy.deepcopy(reference)
    reference_inputs = torch.randn(2, 11, 4, requires_grad=True)
    scanned_inputs = reference_inputs.detach().clone().requires_grad_(True)
    reference_state = torch.randn(2, 2, 5, requires_grad=True)
    scanned_state = reference_state.detach().clone().requires_grad_(True)

    expected, expected_state = reference(reference_inputs, reference_state)
    actual, actual_state = _chunked_forward(scanned, scanned_inputs, scanned_state)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)

    prediction_direction = torch.randn_like(expected)
    state_direction = torch.randn_like(expected_state)
    expected_loss = (expected * prediction_direction).sum() + (
        expected_state * state_direction
    ).sum()
    actual_loss = (actual * prediction_direction).sum() + (actual_state * state_direction).sum()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(scanned_inputs.grad, reference_inputs.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(scanned_state.grad, reference_state.grad, rtol=3e-5, atol=3e-6)
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), scanned.named_parameters(), strict=True
    ):
        assert expected_name == actual_name
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=5e-5, atol=5e-6
        )


def test_activation_quantization_keeps_recurrent_state_float32() -> None:
    torch.manual_seed(67014)
    model = RNNModel(4, hidden=5, layers=2).eval()
    inputs = torch.randn(2, 9, 4)
    predictions, state = activation_qat_forward(model, inputs)
    assert predictions.dtype == torch.float32
    assert state.shape == (2, 2, 5)
    assert state.dtype == torch.float32
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(state).all()


def test_activation_qat_model_wrapper_preserves_chunk_initial_state() -> None:
    torch.manual_seed(67016)
    model = RNNModel(4, hidden=5, layers=2).eval()
    inputs = torch.randn(2, 7, 4)
    initial = torch.randn(2, 2, 5)
    expected, expected_state = model(inputs, initial)
    actual, actual_state = ActivationQATModel(model, quantize_activations=False)(
        inputs, {}, False, initial
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)


def test_native_q8vnni_hidden_state_tracks_recurrent_a8_backend() -> None:
    library = os.environ.get("WNN_Q8_TEST_LIBRARY")
    if library is None:
        pytest.skip("explicitly built native library required for recurrent A8 comparison")
    sys.path.insert(0, library)
    native = pytest.importorskip("wnn_native")

    torch.manual_seed(67015)
    model = RNNModel(3, hidden=8, layers=2).eval()
    # Zero projection is exactly represented by the native I16 input path and
    # makes its output zero before GELU, isolating recurrent A8 effects.
    with torch.no_grad():
        model.inp.weight.zero_()
        model.inp.bias.zero_()
        for layer in range(model.gru.num_layers):
            hidden = model.gru.hidden_size
            getattr(model.gru, f"bias_ih_l{layer}")[: 2 * hidden].zero_()
            getattr(model.gru, f"bias_ih_l{layer}")[2 * hidden :].fill_(0.25)
            getattr(model.gru, f"bias_hh_l{layer}").zero_()
    weights = {
        name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()
    }
    _, q8_codes, _ = quantize_model(weights, "recurrent_head")
    configure_qat(model, "recurrent_head")
    inputs = torch.randn(1, 96, 3)
    with torch.no_grad():
        _, torch_state = activation_qat_forward(model, inputs)
        _, float_activation_state = activation_qat_forward(
            model, inputs, quantize_activations=False
        )

    stepper = native.ModelStepper(
        weights,
        fast_activations=False,
        precision="q8vnni_portable",
        q8_codes=q8_codes,
    )
    stepper.run(inputs[0].numpy())
    native_state = np.asarray(stepper.hidden_state)
    a8_state = torch_state[:, 0].numpy()
    float_state = float_activation_state[:, 0].numpy()
    native_a8_error = float(np.max(np.abs(native_state - a8_state)))
    native_float_error = float(np.max(np.abs(native_state - float_state)))
    activation_effect = float(np.max(np.abs(a8_state - float_state)))
    print(
        "native_q8vnni_recurrent_state_max_abs_error="
        f"{native_a8_error:.9g}; no_a8_error={native_float_error:.9g}; "
        f"a8_effect={activation_effect:.9g}"
    )
    assert activation_effect > 1e-5, "disabling recurrent A8 did not change this fixture"
    np.testing.assert_allclose(native_state, a8_state, rtol=0, atol=1e-4)
