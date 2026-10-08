from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))

from wnn.features import N_EVENT_FEATURES, event_features, event_features_torch  # noqa: E402


def test_numpy_and_trainer_event_features_have_same_211_wide_input() -> None:
    raw = np.random.default_rng(20261009).normal(size=(3, 23, 112)).astype(np.float32)
    expected = event_features(raw)
    observed = event_features_torch(torch.from_numpy(raw)).numpy()
    assert expected.shape == (3, 23, N_EVENT_FEATURES)
    assert N_EVENT_FEATURES == 99
    np.testing.assert_array_equal(expected, observed)
    model_inputs = np.concatenate((raw, expected), axis=-1)
    assert model_inputs.shape == (3, 23, 211)


def test_event_features_do_not_depend_on_future_rows() -> None:
    raw = np.random.default_rng(20261010).normal(size=(1, 23, 112)).astype(np.float32)
    changed = raw.copy()
    changed[:, 12:] += np.float32(0.4)
    original_features = event_features(raw)
    changed_features = event_features(changed)
    np.testing.assert_array_equal(original_features[:, :12], changed_features[:, :12])
