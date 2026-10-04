from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_trainer_resume_integration import _repository, training
from wnn import data
from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel
from wnn.sentinel import (
    SENTINEL,
    SENTINEL_ARMS,
    SENTINEL_TOLERANCE,
    SentinelPreprocessor,
    canonical_hash,
    initialize_sentinel_checkpoint,
    missing_numpy,
)
from wnn.sentinel_calibration import calibrate, fit_safe_statistics, validate_fit


def _checkpoint() -> dict:
    torch.manual_seed(81)
    model = RNNModel(112 + N_EVENT_FEATURES, hidden=4, layers=1)
    model.mean.copy_(torch.randn_like(model.mean))
    model.std.copy_(torch.rand_like(model.std) + 0.5)
    return {
        "model": model.state_dict(),
        "n_in": 112 + N_EVENT_FEATURES,
        "args": {"hidden": 4, "layers": 1, "dropout": 0},
    }


def _calibration(checkpoint: dict) -> dict:
    return {
        "parent_sha256": "a" * 64,
        "parent_mean": checkpoint["model"]["mean"][:112].tolist(),
        "parent_std": checkpoint["model"]["std"][:112].tolist(),
        "safe_mean": [0.2] * 112,
        "safe_std": [0.7] * 112,
        "sentinel": float(SENTINEL),
        "absolute_tolerance": float(SENTINEL_TOLERANCE),
    }


@pytest.mark.parametrize("values,indicators", SENTINEL_ARMS)
def test_numpy_torch_streaming_and_frozen_normalization(values: str, indicators: str) -> None:
    parent = _checkpoint()
    child = initialize_sentinel_checkpoint(parent, _calibration(parent), values, indicators)
    preprocessing = SentinelPreprocessor(child["sentinel_preprocessing"])
    rng = np.random.default_rng(81)
    inputs = rng.normal(size=(20000, parent["n_in"])).astype(np.float32)
    inputs[::19, :5] = SENTINEL
    expected = preprocessing.numpy(inputs)
    actual = preprocessing.torch(torch.from_numpy(inputs)).numpy()
    np.testing.assert_array_equal(actual, expected)
    streamed = np.stack([preprocessing.update(row) for row in inputs[:113]])
    np.testing.assert_array_equal(streamed, expected[:113])
    preprocessing.reset()
    np.testing.assert_array_equal(preprocessing.update(inputs[0]), expected[0])
    altered = inputs.copy()
    altered[99:] *= -3
    np.testing.assert_array_equal(preprocessing.numpy(altered)[:99], expected[:99])
    np.testing.assert_array_equal(expected[:, 112 : parent["n_in"]], inputs[:, 112:])
    for name in ("mean", "std"):
        torch.testing.assert_close(
            child["model"][name][: parent["n_in"]], parent["model"][name], rtol=0, atol=0
        )
    normalized = (
        expected[:, :112] - np.array(preprocessing.constants["parent_mean"])
    ) / preprocessing.constants["parent_std"]
    missing = missing_numpy(inputs[:, :112])
    if values != "unchanged":
        assert (normalized[missing] == 0).all()
    if indicators == "missing":
        np.testing.assert_array_equal(expected[:, -112:], missing.astype(np.float32))
    elif indicators == "current":
        np.testing.assert_array_equal(expected[:, -112:], normalized)


def test_indicators_zero_initialization_preserves_parent_prediction_exactly() -> None:
    parent = _checkpoint()
    original = training.make_model(parent["args"], parent["n_in"], 2).eval()
    original.load_state_dict(parent["model"])
    inputs = torch.randn(1, 20000, parent["n_in"])
    inputs[:, ::17, 111] = float(SENTINEL)
    with torch.no_grad():
        expected, state = original(inputs)
        for indicators in ("none", "missing", "current"):
            child = initialize_sentinel_checkpoint(
                parent, _calibration(parent), "unchanged", indicators
            )
            model = training.make_model(child["args"], child["n_in"], 2).eval()
            model.load_state_dict(child["model"])
            preprocessing = SentinelPreprocessor(child["sentinel_preprocessing"])
            actual, hidden = model(preprocessing.torch(inputs))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(hidden, state, rtol=0, atol=0)
    assert parent["n_in"] == 211


def test_safe_statistics_exclude_missing_and_define_empty_singleton_fallbacks() -> None:
    raw = np.full((4, 112), SENTINEL, np.float32)
    raw[:, 0] = [1, 2, 3, 4]
    raw[0, 1] = 9
    result = fit_safe_statistics([raw[:2], raw[2:]])
    assert result["safe_mean"][:3] == [2.5, 9, 0]
    assert result["safe_std"][:3] == pytest.approx([np.std([1, 2, 3, 4], ddof=1), 1, 1])
    assert result["observed_count"][:3] == [4, 1, 0]
    near = np.array([SENTINEL, SENTINEL + np.float32(0.00009), SENTINEL + np.float32(0.00012)])
    np.testing.assert_array_equal(missing_numpy(near), [True, True, False])


def test_frozen_propensity_and_history_use_original_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _checkpoint()
    child = initialize_sentinel_checkpoint(parent, _calibration(parent), "safe", "missing")
    raw = torch.randn(2, 12, 112)
    raw[:, 3, 110] = float(SENTINEL)
    monkeypatch.setattr(training, "RESEARCH_FEATURES", "none")
    monkeypatch.setattr(training, "PRIVILEGED_MODE", "none")
    monkeypatch.setattr(training, "ONLINE_NORM", False)
    monkeypatch.setattr(training, "RAW_ONLY", False)
    monkeypatch.setattr(training, "INPUT_PROPENSITY", None)
    monkeypatch.setattr(
        training, "SENTINEL_PREPROCESSOR", SentinelPreprocessor(child["sentinel_preprocessing"])
    )
    expected = torch.cat((raw, training.event_features_torch(raw)), dim=-1)

    def propensity(model: object, inputs: torch.Tensor) -> torch.Tensor:
        torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
        return torch.full(inputs.shape[:2], 0.5)

    monkeypatch.setattr(training, "propensity", propensity)
    inputs = training.build_inputs(raw)
    torch.testing.assert_close(inputs[..., 112:211], expected[..., 112:], rtol=0, atol=0)
    args = argparse.Namespace(
        mask_weight=False,
        prop_cache=False,
        prop_power=2,
        prop_eps=0.03,
        class_weight="",
        valid_true_mask=-1,
        since_weight=0,
        episode_weight=0,
    )
    valid = torch.ones(raw.shape[:2], dtype=torch.bool)
    training.main_row_weights(
        inputs, torch.zeros(2, 12, 2), valid, valid, object(), None, args, raw
    )
    with pytest.raises(ValueError, match="original raw"):
        training.main_row_weights(inputs, torch.zeros(2, 12, 2), valid, valid, object(), None, args)


def test_calibration_membership_components_content_and_native_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path, with_sentinels=True)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    parent = _checkpoint()
    parent["provenance"] = {"fit_row_groups": {"train": [], "valid": [0, 1, 2, 3]}}
    path = tmp_path / "parent.pt"
    torch.save(parent, path)
    _, calibration = calibrate(path, tmp_path, sequence_count=3, row_stride=2)
    assert len(calibration["selected_sequences"]) == 3
    assert {entry["row_group"] for entry in calibration["selected_sequences"]} <= {0, 1, 2, 3}
    assert 4 not in calibration["fit_components"]
    validate_fit(calibration, tmp_path, parent["provenance"]["fit_row_groups"], None)
    with pytest.raises(ValueError, match="outside current"):
        validate_fit(calibration, tmp_path, {"train": [], "valid": []}, None)
    child = initialize_sentinel_checkpoint(parent, calibration, "safe", "missing")
    damaged = copy.deepcopy(child["sentinel_preprocessing"])
    damaged["calibration"]["safe_std"][0] = 0
    with pytest.raises(ValueError, match="hash differs"):
        SentinelPreprocessor(damaged)
    damaged["calibration_sha256"] = canonical_hash(damaged["calibration"])
    with pytest.raises(ValueError, match="positive"):
        SentinelPreprocessor(damaged)
    run = tmp_path / "artifacts/runs/test"
    run.mkdir(parents=True)
    torch.save(child, run / "model.pt")
    package = importlib.import_module("package_submission")
    monkeypatch.setattr(package, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="native port"):
        package.export("test", tmp_path / "weights.npz")
    with pytest.raises(ValueError, match="native port"):
        package.merged_heads(["test", "test"])
    with (tmp_path / "datasets/valid.parquet").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="dataset content"):
        validate_fit(calibration, tmp_path, parent["provenance"]["fit_row_groups"], None)
    with (tmp_path / "cache/chains.npz").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="component graph"):
        validate_fit(calibration, tmp_path, parent["provenance"]["fit_row_groups"], None)
