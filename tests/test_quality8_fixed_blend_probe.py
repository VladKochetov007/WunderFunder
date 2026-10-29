from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    """Load the standalone report-only adapter without invoking its CLI."""
    path = Path(__file__).parents[1] / "scripts/probe_quality8_fixed_blend.py"
    spec = importlib.util.spec_from_file_location("quality8_fixed_blend_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_blend_must_precede_official_clipping(probe: ModuleType) -> None:
    """Large opposite raw outputs distinguish the two clipping orders."""
    reference = np.array([[8.0, -8.0]], dtype=np.float32)
    candidate = np.array([[-4.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(probe.fixed_blend(reference, candidate, 0.2), [[5.6, -5.6]])


def test_correlation_matches_independent_weighted_formula(probe: ModuleType) -> None:
    """Check centered correlation against scalar arithmetic with unequal weights."""
    left, right, weight = [1.0, 3.0, 7.0], [5.0, 2.0, 8.0], [1.0, 2.0, 4.0]
    total = math.fsum(weight)
    a = math.fsum(x * w for x, w in zip(left, weight, strict=True)) / total
    b = math.fsum(x * w for x, w in zip(right, weight, strict=True)) / total
    numerator = math.fsum(
        w * (x - a) * (y - b) for x, y, w in zip(left, right, weight, strict=True)
    )
    va = math.fsum(w * (x - a) ** 2 for x, w in zip(left, weight, strict=True))
    vb = math.fsum(w * (x - b) ** 2 for x, w in zip(right, weight, strict=True))
    actual = probe.weighted_correlation(np.array(left), np.array(right), np.array(weight))
    assert actual == pytest.approx(numerator / math.sqrt(va * vb), abs=1e-14)


def test_unaligned_or_nonfinite_blend_is_rejected(probe: ModuleType) -> None:
    """A score must not be produced for misaligned or corrupt prediction arrays."""
    with pytest.raises(ValueError):
        probe.fixed_blend(np.zeros((2, 2)), np.zeros((1, 2)), 0.2)
    with pytest.raises(ValueError):
        probe.fixed_blend(np.zeros((2, 2)), np.full((2, 2), np.nan), 0.2)
