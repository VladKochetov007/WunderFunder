from __future__ import annotations

import numpy as np
import pytest

from scripts.independent_heterogeneous_inputs import ROLES
from scripts.independent_heterogeneous_numeric import MIXTURES, TwoPassPearson, compose


def test_recursive_mixtures_keep_raw_values_until_after_combination() -> None:
    names = [*ROLES, "galam"]
    values = {name: np.array([[3, -4], [-1, 2]], dtype=np.float32) for name in names}
    values["w18"] = np.array([[12, -9], [-8, 7]], dtype=np.float32)
    result = compose(values)
    assert len(result) == 25 and list(MIXTURES)[0] == "TL3"
    raw = np.float32(0.34) * values["w18"] + np.float32(0.33) * values["tiedlead_s5"]
    raw += np.float32(0.33) * values["tiedlead_s6"]
    np.testing.assert_array_equal(result["TL3"], raw)
    assert result["TL3"][0, 0] > 2
    expected = np.float32(0.9) * raw + np.float32(0.1) * values["galam"]
    np.testing.assert_array_equal(result["TL3_plus_GALAM"], expected)
    early_clip = np.float32(0.9) * np.clip(raw, -2, 2) + np.float32(0.1) * values["galam"]
    assert not np.array_equal(expected, early_clip)


def test_independent_two_pass_matches_explicit_centered_weighted_formula() -> None:
    y = np.array([[-3, 0.4], [-0.1, -0.2], [0.7, 1.3], [4, -0.7]], dtype=np.float32)
    p = np.array([[-4, 0.3], [0.2, -0.8], [0.5, 1.4], [3, -0.9]], dtype=np.float32)
    accumulator = TwoPassPearson(["candidate"])
    for index in (0, 1):
        accumulator.add(y[:2], {"candidate": p[:2]}, index)
        accumulator.add(y[2:], {"candidate": p[2:]}, index)
    result = accumulator.result()["candidate"]
    expected = []
    for k in range(2):
        target = np.clip(y[:, k], -2, 2).astype(np.float64)
        prediction = np.clip(p[:, k], -2, 2).astype(np.float64)
        weights = np.abs(target)
        a = target - np.average(target, weights=weights)
        b = prediction - np.average(prediction, weights=weights)
        expected.append(
            np.sum(weights * a * b) / np.sqrt(np.sum(weights * a * a) * np.sum(weights * b * b))
        )
    assert result["t0"] == pytest.approx(expected[0], abs=1e-14)
    assert result["t1"] == pytest.approx(expected[1], abs=1e-14)
    assert result["wp"] == pytest.approx(sum(expected) / 2, abs=1e-14)
    assert result["selected_rows"] == 4


def test_zero_weight_and_constant_predictions_preserve_fallback_and_counts() -> None:
    y = np.array([[0, -1], [0, 1]], dtype=np.float32)
    p = np.ones((2, 2), dtype=np.float32)
    accumulator = TwoPassPearson(["candidate"])
    for index in (0, 1):
        accumulator.add(y, {"candidate": p}, index)
    assert accumulator.result()["candidate"] == {
        "t0": 0.0,
        "t1": 0.0,
        "wp": 0.0,
        "selected_rows": 2,
    }


def test_selected_nonfinite_values_and_incomplete_second_pass_are_rejected() -> None:
    accumulator = TwoPassPearson(["candidate"])
    y = np.array([[1, -1]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite before clipping"):
        accumulator.add(y, {"candidate": np.array([[np.inf, -1]], dtype=np.float32)}, 0)
    with pytest.raises(ValueError, match="complete nonempty passes"):
        TwoPassPearson(["candidate"]).result()
