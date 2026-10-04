from __future__ import annotations

import numpy as np
import pytest

from wnn.quantization import quantize_model
from wnn.quantization_export import read_q8_export, write_q8_export


@pytest.mark.parametrize("preset", ["recurrent", "recurrent_head"])
def test_q8_transport_preserves_codes_scale_bits_and_untouched_tensors(
    tmp_path, preset: str
) -> None:
    rng = np.random.default_rng(199)
    weights = {
        "inp.bias": np.zeros(5, np.float32),
        "inp.weight": rng.normal(size=(5, 211)).astype(np.float32),
        "gru.weight_ih_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "gru.weight_hh_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "head.0.weight": rng.normal(size=(5, 10)).astype(np.float32),
    }
    represented, encoded, _ = quantize_model(weights, preset)
    # A non-maximal row code demonstrates why import must preserve the original
    # scale instead of recomputing it from effective float32 weights.
    encoded["gru.weight_ih_l0.0.codes"][0] = np.array([3, -11, 5, 0, 8], np.int8)
    filename = tmp_path / "weights_q8.npz"
    write_q8_export(filename, represented, preset, encoded)
    restored, codes, got_preset = read_q8_export(filename)
    assert got_preset == preset
    for name in encoded:
        assert encoded[name].dtype == codes[name].dtype
        np.testing.assert_array_equal(encoded[name].view(np.uint8), codes[name].view(np.uint8))
    np.testing.assert_array_equal(restored["inp.weight"], weights["inp.weight"])
    np.testing.assert_array_equal(
        restored["gru.weight_ih_l0"][0],
        encoded["gru.weight_ih_l0.0.codes"][0].astype(np.float32)
        * encoded["gru.weight_ih_l0.0.scales"][0],
    )
    with pytest.raises(FileExistsError):
        write_q8_export(filename, weights, preset)
