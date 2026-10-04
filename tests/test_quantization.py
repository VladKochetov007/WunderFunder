from __future__ import annotations

import numpy as np
import pytest

from wnn.quantization import merge_frozen_heads, quantize_model, quantize_rows, represented_rows


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
