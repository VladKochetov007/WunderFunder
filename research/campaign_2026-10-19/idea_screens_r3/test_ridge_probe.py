"""Tiny synthetic invariants for the fixed R3 component/ridge analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-19"))

from idea_screens_r1.feature_views import build_view  # noqa: E402

from idea_screens_r3.analysis_fit import fit_component_sequences  # noqa: E402
from idea_screens_r3.analysis_views import CASE_NAMES, build_case_views  # noqa: E402
from idea_screens_r3.probe_helpers import (  # noqa: E402
    ALPHAS,
    component_folds,
    metric_without,
    raw_sums_by_group,
    select_alpha,
)
from wnn.gmetric import GlobalAccumulator  # noqa: E402


def test_feature_blocks_are_exact_and_causal() -> None:
    rng = np.random.default_rng(19_025)
    raw = rng.normal(size=(160, 112)).astype(np.float32)
    views = build_case_views(raw)
    assert tuple(views) == CASE_NAMES
    assert [views[name].shape[1] for name in CASE_NAMES] == [
        52,
        60,
        104,
        112,
        224,
        224,
        336,
        112,
        60,
    ]
    np.testing.assert_array_equal(views["full64_i0_52"], raw[:, :52])
    np.testing.assert_array_equal(views["full64_i0_tail60"][:, 52:], raw[:, 104:112])
    np.testing.assert_array_equal(views["full64_books104"], raw[:, :104])
    ema, _ = build_view(raw, "raw_net_absinnovation_ema")
    np.testing.assert_array_equal(views["full64_raw_both_ema_parts_hl32"], ema)
    np.testing.assert_array_equal(views["full64_raw_signed_ema_hl32"][:, 112:], ema[:, 112:224])
    np.testing.assert_array_equal(
        views["full64_raw_absolute_innovation_ema_hl32"][:, 112:], ema[:, 224:336]
    )

    changed = raw.copy()
    changed[100:] += np.float32(5.0)
    changed_views = build_case_views(changed)
    for name in CASE_NAMES:
        np.testing.assert_array_equal(views[name][:100], changed_views[name][:100])
    fresh = build_case_views(np.full((160, 112), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(fresh["full64_raw_both_ema_parts_hl32"][0, 112:], 0.0)


def test_component_folds_are_nested_budget_partitions() -> None:
    full = component_folds(list(range(64)))
    assert [len(fold) for fold in full] == [22, 21, 21]
    assert full == component_folds(list(reversed(range(64))))
    low = component_folds(list(range(32)))
    assert [len(fold) for fold in low] == [11, 11, 10]
    assert set.union(*low) == set(range(32))


def test_fit_metadata_resolution_never_reads_outer_arrays() -> None:
    class EvaluationBomb(dict):
        def __getitem__(self, key):
            if key in {"x", "y", "scored"}:
                raise AssertionError("fit metadata resolution touched an evaluation array")
            return super().__getitem__(key)

    fit = [{"role": "fit", "component": comp} for comp in range(64)]
    outer = EvaluationBomb(role="eval", component=500, x=None, y=None, scored=None)
    selected = fit_component_sequences(fit + [outer], list(range(64)))
    assert set(selected) == set(range(64))


def test_drop_one_uses_canonical_pooled_moments_not_component_average() -> None:
    rng = np.random.default_rng(1_902_500)
    lengths = (7, 50, 100)
    part_accumulators = {}
    arrays = {}
    for component, length in enumerate(lengths):
        y = rng.normal(size=(1, length, 2)).astype(np.float32)
        noise = rng.normal(size=y.shape).astype(np.float32)
        prediction = y * (1.0 if component == 0 else -0.5) + noise * (
            0.01 if component == 0 else 0.6
        )
        mask = np.ones((1, length), dtype=bool)
        accumulator = GlobalAccumulator()
        accumulator.add(y, prediction, mask, group=component)
        part_accumulators[component] = accumulator
        arrays[component] = (y, prediction, mask)

    dropped = metric_without(part_accumulators, 0)
    direct = GlobalAccumulator()
    for component in (1, 2):
        direct.add(*arrays[component])
    np.testing.assert_allclose(
        [dropped["t0"], dropped["t1"], dropped["wp"]],
        [direct.result()["t0"], direct.result()["t1"], direct.result()["wp"]],
        rtol=1e-13,
        atol=1e-13,
    )
    assert set(raw_sums_by_group(part_accumulators[0])) == {"0"}

    pooled = {alpha: GlobalAccumulator() for alpha in ALPHAS}
    same_y, same_p, same_mask = arrays[1]
    for accumulator in pooled.values():
        accumulator.add(same_y, same_p, same_mask)
    assert select_alpha(pooled) == max(ALPHAS)
