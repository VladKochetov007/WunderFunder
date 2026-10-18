"""Tiny invariants for the bounded R2 comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-19"))

from idea_screens_r1.feature_views import build_view  # noqa: E402

from idea_screens_r2.ridge_cv import (  # noqa: E402
    RidgeModel,
    _collect_fit_rows,
    _component_folds,
    _outer_score,
)
from idea_screens_r2.views import build_views  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402


def test_fit_row_collection_ignores_outer_partition_labels() -> None:
    rng = np.random.default_rng(19020)
    fit_x = rng.normal(size=(20_000, 112)).astype(np.float32)
    fit_y = rng.normal(size=(20_000, 2)).astype(np.float32)
    fit_mask = np.zeros(20_000, dtype=bool)
    fit_mask[0] = True
    fit = {
        "role": "fit",
        "x": fit_x,
        "y": fit_y,
        "scored": fit_mask,
        "component": 14,
    }
    # If the collector inspects any outer array or label, this deliberately invalid record fails.
    outer = {"role": "eval", "x": np.zeros((1, 1)), "y": np.full((1, 1), 1e20)}

    views, targets, components, sequence_count = _collect_fit_rows([fit, outer])

    assert sequence_count == 1 and len(targets) == 1
    np.testing.assert_array_equal(targets, fit_y[:1])
    np.testing.assert_array_equal(components, np.array([14], dtype=np.int32))
    assert all(len(matrix) == 1 for matrix in views.values())


def test_seeded_component_folds_are_a_partition() -> None:
    components = list(range(100, 114))
    folds = _component_folds(components)
    assert [len(fold) for fold in folds] == [5, 5, 4]
    assert set.union(*folds) == set(components)
    assert all(not folds[left] & folds[right] for left in range(3) for right in range(left))
    assert folds == _component_folds(components)


def test_signed_and_absolute_ema_views_are_exact_causal_r1_parts() -> None:
    rng = np.random.default_rng(52)
    raw = rng.normal(size=(140, 112)).astype(np.float32)
    full, _ = build_view(raw, "raw_net_absinnovation_ema")
    views = build_views(raw)
    np.testing.assert_array_equal(views["raw_signed_delta_ema_hl32"], full[:, :224])
    expected_absolute = np.concatenate((raw, full[:, 224:336]), axis=1)
    np.testing.assert_array_equal(views["raw_absolute_innovation_ema_hl32"], expected_absolute)
    np.testing.assert_array_equal(views["raw_both_ema_parts_hl32"], full)

    altered = raw.copy()
    altered[90:] += 20.0
    changed, _ = build_view(altered, "raw_net_absinnovation_ema")
    np.testing.assert_array_equal(full[:90], changed[:90])


def test_outer_score_matches_canonical_global_scorer_and_drop_one() -> None:
    target_a = np.array([[1.0, -1.0], [-2.0, 2.0]], dtype=np.float32)
    target_b = np.array([[2.0, 1.0], [-1.0, -2.0]], dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    sequences = []
    for target, component, source in (
        (target_a, 1, "s1"),
        (target_b, 1, "s2"),
        (-target_a, 2, "s3"),
        (-target_b, 2, "s4"),
    ):
        features = np.zeros((2, 112), dtype=np.float32)
        features[:, :2] = target
        sequences.append(
            {
                "role": "eval",
                "x": features,
                "y": target,
                "scored": mask,
                "component": component,
                "source": source,
                "group": source,
            }
        )
    coefficients = np.zeros((112, 2), dtype=np.float64)
    coefficients[0, 0], coefficients[1, 1] = 0.7, -0.4
    model = RidgeModel(np.zeros(112), np.ones(112), coefficients, np.zeros(2), (0.0, 0.0))

    result = _outer_score(sequences, "raw", {"selected": model})["selected"]

    targets = np.concatenate([sequence["y"] for sequence in sequences])
    prediction = np.concatenate(
        [sequence["x"][:, :2] * np.array([0.7, -0.4]) for sequence in sequences]
    )
    expected = global_wp(targets, prediction, np.ones(8, dtype=bool))
    assert abs(result["official_metric"]["wp"] - expected["wp"]) < 1e-12
    assert len(result["component_moments"]) == 2
    assert all(row["selected_rows"] == 4 for row in result["component_moments"])
    assert all(row["official_metric"]["selected_rows"] == 4 for row in result["drop_one_component"])
