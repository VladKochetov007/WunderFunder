from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from wnn.gmetric import reference_global_wp

ROOT = Path(__file__).resolve().parents[1]
GRADER = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
    / "grade_fixed_embedding_pair.py"
)


def test_fixed_pair_matches_direct_metric_with_reordered_rows_and_raw_clipping() -> None:
    spec = importlib.util.spec_from_file_location("e540_fixed_pair_test", GRADER)
    assert spec is not None and spec.loader is not None
    grade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grade)
    scorer = grade.load_module("e540_scorer_test", grade.SCORER)
    rng = np.random.default_rng(20261008)
    labels = rng.normal(size=(3, 130, 2)).astype(np.float32)
    candidate = rng.normal(size=(3, 130, 2)).astype(np.float32) * 5.0
    control = rng.normal(size=(3, 130, 2)).astype(np.float32) * 4.0
    mask = rng.random((3, 130)) > 0.2
    selected = mask & (np.arange(130)[None, :] >= 99)
    # Candidate and control exports deliberately use distinct physical row orders.
    candidate_order = [2, 0, 1]
    control_order = [1, 2, 0]
    aligned = [
        (("valid", group), candidate_order.index(group), control_order.index(group), group)
        for group in range(3)
    ]
    rows = grade.score_pair(
        {"y": labels, "mask": mask},
        candidate[candidate_order],
        control[control_order],
        aligned,
        np.arange(3),
        scorer,
        [0.0, 0.5, 1.0],
        {"replicates": 20, "seed": 271828},
    )
    assert [row["weight_on_ple"] for row in rows] == [0.0, 0.5, 1.0]
    for row in rows:
        weight = row["weight_on_ple"]
        raw = (1.0 - weight) * control + weight * candidate
        expected = reference_global_wp(labels, raw, selected)
        assert np.isclose(row["score"]["wp"], expected, atol=1e-12, rtol=0)
    premature = 0.5 * np.clip(control, -2, 2) + 0.5 * np.clip(candidate, -2, 2)
    assert not np.isclose(rows[1]["score"]["wp"], reference_global_wp(labels, premature, selected))
