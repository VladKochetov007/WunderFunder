from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from feature_cache import frozen_state  # noqa: E402
from feature_reset import LAGS, panel_groups, reset_prediction  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402


@pytest.mark.parametrize("mode,width", [("none", 211), ("volume28", 239)])
def test_reset_preserves_warmup_features_weights_and_exact_boundaries(
    mode: str, width: int
) -> None:
    torch.manual_seed(9)
    model = RNNModel(width, hidden=4, layers=1).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    raw = np.random.default_rng(19).normal(size=(2300, 112)).astype(np.float32)
    baseline = reset_prediction(model, raw, mode, "cpu", None, 1.0)
    _, expected = frozen_state(model, raw, mode, "cpu")
    np.testing.assert_array_equal(baseline, expected)
    for lag in LAGS:
        retained = reset_prediction(model, raw, mode, "cpu", lag, 1.0)
        reset = reset_prediction(model, raw, mode, "cpu", lag, 0.0)
        half = reset_prediction(model, raw, mode, "cpu", lag, 0.5)
        boundary = 99 + lag
        np.testing.assert_allclose(retained, baseline, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(reset[:boundary], baseline[:boundary], atol=2e-7, rtol=2e-6)
        assert np.max(np.abs(reset[boundary] - baseline[boundary])) > 1e-6
        assert np.max(np.abs(half[boundary] - baseline[boundary])) > 1e-6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    altered = raw.copy()
    altered[600:] *= 10
    expected_prefix = reset_prediction(model, raw, mode, "cpu", 250, 0.0)[:600]
    actual_prefix = reset_prediction(model, altered, mode, "cpu", 250, 0.0)[:600]
    np.testing.assert_array_equal(actual_prefix, expected_prefix)


def test_panel_is_label_independent_bounded_and_keeps_components_complete() -> None:
    groups = {component: [2 * component, 2 * component + 1] for component in range(40)}
    selected = panel_groups(groups)
    assert len(selected) == 32
    assert selected == panel_groups(dict(reversed(list(groups.items()))))
    assert all(
        (2 * component in selected) == (2 * component + 1 in selected) for component in groups
    )
    with pytest.raises(ValueError, match="four complete"):
        panel_groups({key: list(range(20)) for key in range(4)}, group_limit=32)
    with pytest.raises(ValueError, match="prespecified"):
        reset_prediction(RNNModel(211, 4), np.zeros((100, 112), np.float32), "none", "cpu", 12, 0)
