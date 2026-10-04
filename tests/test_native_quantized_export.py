from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from wnn.quantization import quantize_model
from wnn.quantization_export import read_q8_export, write_q8_export

LIBRARY = os.environ.get("WNN_Q8_TEST_LIBRARY")
if LIBRARY is None:
    pytest.skip("explicit newly built q8 library required", allow_module_level=True)
sys.path.insert(0, LIBRARY)
native = pytest.importorskip("wnn_native")
torch = pytest.importorskip("torch")

from wnn.nn import RNNModel  # noqa: E402


@pytest.mark.parametrize("preset,precision", [("recurrent", "q8r"), ("recurrent_head", "q8x")])
def test_native_exact_code_import_export_and_portable_prediction(
    tmp_path: Path, preset: str, precision: str
) -> None:
    torch.manual_seed(811)
    model = RNNModel(211, 8, 2).eval()
    weights = {name: tensor.detach().numpy().copy() for name, tensor in model.state_dict().items()}
    _, expected, _ = quantize_model(weights, preset)
    raw = native.Predictor(weights, precision=precision)
    for name, value in raw.quantization_codes().items():
        np.testing.assert_array_equal(value.view(np.uint8), expected[name].view(np.uint8))
    # Preserve a deliberately non-maximal row and an arbitrary scale exactly;
    # recomputing absmax scales from the decoded matrix would change its bytes.
    expected["gru.weight_ih_l0.0.codes"][0] = 3
    expected["gru.weight_ih_l0.0.scales"][0] = np.array([0x3C123456], np.uint32).view(np.float32)[0]
    path = tmp_path / "weights.npz"
    write_q8_export(path, weights, preset, expected)
    represented, codes, _ = read_q8_export(path)
    simd = native.Predictor(represented, precision=precision, q8_codes=codes)
    portable = native.Predictor(represented, precision=f"{precision}_portable", q8_codes=codes)
    assert simd.weight_bytes < native.Predictor(represented).weight_bytes
    for name, value in simd.quantization_codes().items():
        np.testing.assert_array_equal(value.view(np.uint8), expected[name].view(np.uint8))
    features = np.random.default_rng(77).normal(size=(200, 112)).astype(np.float32)
    first = np.stack([simd.step(row) for row in features])
    np.testing.assert_array_equal(first, np.stack([portable.step(row) for row in features]))
    simd.reset()
    np.testing.assert_array_equal(first, np.stack([simd.step(row) for row in features]))
    with pytest.raises(ValueError, match="preset"):
        native.Predictor(represented, precision="f32", q8_codes=codes)
    with pytest.raises(ValueError):
        native.Predictor(represented, precision=precision, q8_codes={})
