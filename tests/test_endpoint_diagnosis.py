from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def test_streamed_diagnostic_moments_match_dense_weighted_calculation() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/diagnose_native_h512.py"
    )
    spec = importlib.util.spec_from_file_location("endpoint_diagnostic_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.array([[2.0, -3.0], [4.0, 5.0], [-1.0, 7.0], [9.0, 1.0]])
    weights = np.array([0.1, 2.0, 0.0, 1.5])
    raw, paired = np.zeros(5), np.zeros(6)
    for selection in (slice(0, 1), slice(1, 4)):
        raw = module.raw_update(raw, values[selection])
        paired = module.pair(paired, *values[selection].T, weights[selection])
    np.testing.assert_allclose(module.raw_summary(raw)["mean"], values.mean(axis=0))
    np.testing.assert_allclose(module.raw_summary(raw)["std"], values.std(axis=0))
    center = np.average(values, axis=0, weights=weights)
    covariance = np.average(np.prod(values - center, axis=1), weights=weights)
    np.testing.assert_allclose(module.summary(paired)["covariance"], covariance)
