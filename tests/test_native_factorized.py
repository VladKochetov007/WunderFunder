from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.nn import RNNModel


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("WNN_FACTOR_TEST_LIBRARY")
    if not path:
        pytest.skip("set WNN_FACTOR_TEST_LIBRARY to a built extension directory")
    spec = importlib.util.spec_from_file_location("wnn_native", Path(path) / "wnn_native.abi3.so")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_model(hidden: int, rank: int) -> tuple[RNNModel, dict, dict]:
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    model = RNNModel(11, hidden, 2, n_out=2).eval()
    rng = np.random.default_rng(271828)
    factors = {}
    with torch.no_grad():
        for layer in range(2):
            for kind in ("ih", "hh"):
                name = f"weight_{kind}_l{layer}"
                left = rng.normal(0, 0.25 / np.sqrt(rank), (3 * hidden, rank)).astype(np.float32)
                right = rng.normal(0, 1 / np.sqrt(hidden), (rank, hidden)).astype(np.float32)
                factors[f"gru.{name}"] = {"left": left, "right": right}
                getattr(model.gru, name).copy_(torch.from_numpy(left) @ torch.from_numpy(right))
    weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    return model, weights, factors


@pytest.mark.parametrize("hidden,rank,rows", [(17, 5, 20000), (33, 9, 513), (384, 128, 64)])
def test_factor_native_matches_effective_dense_and_preserves_import(native, hidden, rank, rows):
    model, weights, factors = fixture_model(hidden, rank)
    x = np.random.default_rng(123).normal(0, 0.25, (rows, 11)).astype(np.float32)
    predictor = native.ModelStepper(weights, precision="factor_f32", factors=factors)
    scalar = native.ModelStepper(weights, precision="factor_f32_portable", factors=factors)
    actual = predictor.run(x)
    assert np.array_equal(actual, scalar.run(x))
    predictor.reset()
    assert np.array_equal(actual, predictor.run(x))
    expected, state = [], None
    with torch.inference_mode():
        for first in range(0, rows, 512):
            value, state = model(torch.from_numpy(x[first : first + 512])[None], state)
            expected.append(value[0].numpy())
    assert np.abs(actual - np.concatenate(expected)).max() <= 3e-4
    exported = predictor.factorization_tensors()
    assert exported.keys() == factors.keys()
    for name, pair in factors.items():
        for side, value in pair.items():
            assert exported[name][side].dtype == np.float32
            assert exported[name][side].tobytes() == value.tobytes()
    storage = predictor.storage_bytes
    assert storage["factor_weights"] == sum(
        value.nbytes for pair in factors.values() for value in pair.values()
    )
    assert storage["import_only_dense_recurrent"] == 4 * 3 * hidden * hidden * 4
    assert storage["recurrent_state"] == 2 * hidden * 4
    assert storage["scratch"] >= rank * 4
    assert storage["total_streamed_weights"] == predictor.weight_bytes
    assert storage["factor_weights"] <= storage["total_streamed_weights"]
    assert predictor.weight_bytes < native.ModelStepper(weights).weight_bytes
    # Dense import placeholders affect neither the persistent representation nor outputs.
    changed = {
        name: np.zeros_like(value) if name in factors else value for name, value in weights.items()
    }
    imported = native.ModelStepper(changed, precision="factor_f32", factors=factors)
    assert np.array_equal(actual, imported.run(x))


def test_factor_import_rejects_incomplete_wrong_dtype_or_mixed_precision(native):
    _, weights, factors = fixture_model(17, 5)
    with pytest.raises(ValueError, match="requires all GRU"):
        native.ModelStepper(weights, precision="factor_f32")
    with pytest.raises(ValueError, match="require factor_f32"):
        native.ModelStepper(weights, factors=factors)
    incomplete = dict(factors)
    incomplete.pop("gru.weight_ih_l0")
    with pytest.raises(ValueError, match="requires all GRU"):
        native.ModelStepper(weights, precision="factor_f32", factors=incomplete)
    invalid = {name: dict(pair) for name, pair in factors.items()}
    invalid["gru.weight_ih_l0"]["left"] = invalid["gru.weight_ih_l0"]["left"].astype(np.float64)
    with pytest.raises(TypeError):
        native.ModelStepper(weights, precision="factor_f32", factors=invalid)
    with pytest.raises(ValueError, match="q8 codes/scales"):
        native.ModelStepper(weights, precision="factor_f32", factors=factors, q8_codes={})
