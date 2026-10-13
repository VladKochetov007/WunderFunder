from __future__ import annotations

import numpy as np
import pytest

from wnn.quantization import quantize_model, represented_rows_q4
from wnn.quantization_export import (
    pack_q4_codes,
    pack_q8_codes,
    read_packed_quantized_export,
    read_q8_export,
    unpack_q4_codes,
    unpack_q8_codes,
    write_packed_quantized_export,
    write_q8_export,
)


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


@pytest.mark.parametrize("preset", ["recurrent", "recurrent_w4", "input_w4_recurrent_w8"])
def test_versioned_packed_archive_roundtrip_is_exact(tmp_path, preset: str) -> None:
    rng = np.random.default_rng(891)
    weights = {
        "inp.bias": np.zeros(5, np.float32),
        "inp.weight": rng.normal(size=(5, 211)).astype(np.float32),
        "gru.weight_ih_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "gru.weight_hh_l0": rng.normal(size=(15, 5)).astype(np.float32),
        "head.0.weight": rng.normal(size=(5, 10)).astype(np.float32),
    }
    effective, encoded, report = quantize_model(weights, preset)
    # Non-maximal codes prove that reading never derives new scales from floats.
    key = "gru.weight_ih_l0.0.codes"
    encoded[key][0] = np.array([3, -6, 5, 0, 2], np.int8)
    scale_key = "gru.weight_ih_l0.0.scales"
    effective["gru.weight_ih_l0"] = encoded[key].astype(np.float32) * encoded[scale_key][:, None]
    if preset in ("recurrent_w4", "input_w4_recurrent_w8"):
        effective["gru.weight_ih_l0"] = represented_rows_q4(encoded[key], encoded[scale_key])
    archive = tmp_path / "quantized.npz"
    write_packed_quantized_export(archive, effective, preset, encoded)
    restored, exact, metadata = read_packed_quantized_export(archive)
    assert metadata["version"] == 2
    assert metadata["preset"] == preset
    assert report["preset"] == preset
    for name in encoded:
        assert exact[name].dtype == encoded[name].dtype
        np.testing.assert_array_equal(exact[name].view(np.uint8), encoded[name].view(np.uint8))
    for name in effective:
        np.testing.assert_array_equal(restored[name], effective[name])


def test_packed_code_helpers_define_nibble_order_padding_and_invalids() -> None:
    q4 = np.array([[1, -2, 7], [-7, 0, 3]], np.int8)
    packed = pack_q4_codes(q4)
    np.testing.assert_array_equal(packed, [0xE1, 0x97, 0x30])
    np.testing.assert_array_equal(unpack_q4_codes(packed, q4.shape), q4)
    with pytest.raises(ValueError, match="pad"):
        unpack_q4_codes(np.array([0x01, 0x81], np.uint8), (1, 3))
    with pytest.raises(ValueError, match="-8"):
        unpack_q4_codes(np.array([0x08], np.uint8), (1, 1))
    with pytest.raises(ValueError, match=r"\[-7, 7\]"):
        pack_q4_codes(np.array([[-8]], np.int8))
    q8 = np.array([[-127, -3, 0, 8, 126]], np.int8)
    np.testing.assert_array_equal(unpack_q8_codes(pack_q8_codes(q8), q8.shape), q8)
    with pytest.raises(ValueError, match="-128"):
        pack_q8_codes(np.array([[-128]], np.int8))
