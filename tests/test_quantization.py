from __future__ import annotations

import numpy as np
import pytest

from wnn.quantization import (
    component_panel,
    merge_frozen_heads,
    quantize_model,
    quantize_rows,
    quantize_rows_q4,
    represented_rows,
    represented_rows_q4,
)


def test_component_panel_is_fixed_balanced_and_order_independent() -> None:
    mapping = {group: group // 3 for group in range(300)}
    # One large component must not dominate the sequence panel.
    mapping.update({group: 0 for group in range(300, 500)})
    selected = component_panel(tuple(mapping), mapping)
    assert len(selected) == len({mapping[group] for group in selected}) == 64
    assert component_panel(tuple(reversed(mapping)), mapping) == selected
    with pytest.raises(ValueError, match="distinct"):
        component_panel(tuple(range(63)), {group: group for group in range(63)})


def fixture_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(73)
    return {
        "inp.bias": np.ones(3, np.float32),
        "inp.weight": rng.normal(size=(3, 5)).astype(np.float32),
        "gru.weight_ih_l0": rng.normal(size=(9, 3)).astype(np.float32),
        "gru.weight_hh_l0": rng.normal(size=(9, 3)).astype(np.float32),
        "head.0.weight": rng.normal(size=(3, 6)).astype(np.float32),
        "head.0.bias": rng.normal(size=3).astype(np.float32),
        "head.2.weight": rng.normal(size=(2, 3)).astype(np.float32),
        "head.2.bias": rng.normal(size=2).astype(np.float32),
    }


def test_native_tie_rounding_fixture_and_subnormal() -> None:
    weights = np.array([[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]], np.float32)
    codes, scales = quantize_rows(weights)
    np.testing.assert_array_equal(codes, [[127, 0, 2, 2, 0, -2, -2]])
    np.testing.assert_array_equal(represented_rows(codes, scales), codes.astype(np.float32))
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = np.array([[tiny, 0], [0, 0]], np.float32)
    np.testing.assert_array_equal(represented_rows(*quantize_rows(values)), values)


def test_q4_ties_subnormals_zero_rows_and_invalid_codes() -> None:
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = np.array(
        [[7, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -7], [tiny, 0, 0, 0, 0, 0, 0, 0], [0] * 8],
        np.float32,
    )
    codes, scales = quantize_rows_q4(values)
    np.testing.assert_array_equal(codes[0], [7, 0, 2, 2, 0, -2, -2, -7])
    np.testing.assert_array_equal(represented_rows_q4(codes, scales)[1:], values[1:])
    np.testing.assert_array_equal(scales[2:], [0])
    with pytest.raises(ValueError, match="Q4"):
        represented_rows_q4(np.array([[-8]], np.int8), np.array([1], np.float32))
    with pytest.raises(ValueError, match="Q4"):
        represented_rows_q4(np.array([[1]], np.int8), np.array([0], np.float32))


@pytest.mark.parametrize("preset", ["recurrent", "recurrent_head"])
def test_exact_serialized_representation_and_unchanged_fields(tmp_path, preset: str) -> None:
    original = fixture_weights()
    represented, encoded, report = quantize_model(original, preset)
    np.savez(tmp_path / "codes.npz", **encoded)
    with np.load(tmp_path / "codes.npz", allow_pickle=False) as archive:
        for name, details in report["selected"].items():
            reconstructed = np.concatenate(
                [
                    represented_rows(archive[f"{name}.{i}.codes"], archive[f"{name}.{i}.scales"])
                    for i in range(details["blocks"])
                ],
                axis=1,
            )
            np.testing.assert_array_equal(represented[name], reconstructed)
    for name in report["unquantized"]:
        np.testing.assert_array_equal(represented[name], original[name])
    for name in report["selected"]:
        assert not np.array_equal(represented[name], original[name])


@pytest.mark.parametrize(
    ("preset", "expected_bits"),
    [
        ("recurrent_w4", {"gru.weight_ih_l0": 4, "gru.weight_hh_l0": 4, "head.0.weight": 8}),
        (
            "input_w4_recurrent_w8",
            {"gru.weight_ih_l0": 4, "gru.weight_hh_l0": 8, "head.0.weight": 8},
        ),
    ],
)
def test_versioned_w4_policies_select_expected_matrices(
    preset: str, expected_bits: dict[str, int]
) -> None:
    original = fixture_weights()
    represented, encoded, report = quantize_model(original, preset)
    assert report["version"] == 2
    for name, bits in expected_bits.items():
        assert report["selected"][name]["block_bits"] == (
            [8, 8] if name == "head.0.weight" else [bits]
        )
        assert np.all(np.abs(encoded[f"{name}.0.codes"]) <= (7 if bits == 4 else 127))
    for name in report["unquantized"]:
        np.testing.assert_array_equal(represented[name], original[name])


def test_merging_checks_frozen_trunk_and_preserves_equal_head_prediction() -> None:
    parent = fixture_weights()
    tuned = {name: value.copy() for name, value in parent.items()}
    tuned["head.2.bias"] += np.float32(0.7)
    merged = merge_frozen_heads(parent, tuned)
    x = np.arange(3, dtype=np.float32)
    expected = (
        parent["head.2.weight"] @ x
        + parent["head.2.bias"]
        + tuned["head.2.weight"] @ x
        + tuned["head.2.bias"]
    ) * 0.5
    np.testing.assert_allclose(
        merged["head.2.weight"] @ np.tile(x, 2) + merged["head.2.bias"], expected, atol=1e-6
    )
    tuned["inp.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="frozen trunk"):
        merge_frozen_heads(parent, tuned)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_matrices_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite matrix"):
        quantize_rows(np.array([[value]], np.float32))


def test_impossible_representation_rejected() -> None:
    with pytest.raises(ValueError, match="representation"):
        represented_rows(np.array([[-128]], np.int8), np.ones(1, np.float32))
    with pytest.raises(ValueError, match="representation"):
        represented_rows(np.array([[1]], np.int8), np.zeros(1, np.float32))


@pytest.mark.parametrize("preset", ["recurrent", "recurrent_head"])
def test_represented_merged_model_native_matches_torch(preset: str) -> None:
    import importlib.util
    import os
    import sys
    from pathlib import Path

    torch = pytest.importorskip("torch")
    library = os.environ.get("WNN_Q8_TEST_LIBRARY")
    if not library:
        pytest.skip("explicit native test library required")
    sys.path.insert(0, library)
    native = pytest.importorskip("wnn_native")
    from wnn.nn import RNNModel

    path = Path(__file__).resolve().parents[1] / "scripts/probe_q8_model.py"
    loader = importlib.util.spec_from_file_location("q8_probe", path)
    assert loader is not None and loader.loader is not None
    probe = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(probe)
    torch.manual_seed(177)
    model = RNNModel(211, 8, 2).eval()
    parent = {name: tensor.detach().numpy().copy() for name, tensor in model.state_dict().items()}
    tuned = {name: value.copy() for name, value in parent.items()}
    tuned["head.0.weight"] *= np.float32(1.3)
    weights, _, _ = quantize_model(merge_frozen_heads(parent, tuned), preset)
    features = np.cumsum(np.random.default_rng(17).normal(size=(160, 112)), axis=0).astype(
        np.float32
    )
    expected = probe.torch_reference(weights, features)
    predictor = native.Predictor(weights, fast_activations=False, precision="f32")
    observed = np.stack([predictor.step(row) for row in features])
    np.testing.assert_allclose(observed, expected, atol=3e-5, rtol=1e-5)
    predictor.reset()
    np.testing.assert_array_equal(predictor.step(features[0]), observed[0])
