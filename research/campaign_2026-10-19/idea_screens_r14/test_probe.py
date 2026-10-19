"""Synthetic invariants and metadata-only source-consumer checks for R14."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research" / "campaign_2026-10-19"
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(ROOT / "src"))

from idea_screens_r3.probe_helpers import metric_without, safe_result  # noqa: E402

from idea_screens_r14.history_features import (  # noqa: E402
    RAW_WIDTH,
    build_a_features,
    coordinate_map,
    direct_map_check,
    fit_raw_normalizer,
)
from idea_screens_r14.linear_math import (  # noqa: E402
    build_systems,
    mapped_systems,
    solve_optimum,
    steepest_descent,
)
from idea_screens_r14.result_payload import _mean_defined  # noqa: E402
from wnn.gmetric import GlobalAccumulator, global_wp  # noqa: E402

R14 = CAMPAIGN / "idea_screens_r14"


def _objective(system: object, beta: np.ndarray) -> float:
    """Evaluate the normalized sufficient-statistic objective for one toy fit."""
    return float(
        (beta @ system.hessian @ beta - 2 * beta @ system.rhs + system.label_square_sum)
        / system.weight_sum
    )


def test_history_is_causal_and_resets_per_sequence() -> None:
    raw = np.zeros((64, RAW_WIDTH), dtype=np.float32)
    raw[:, 0] = np.arange(len(raw), dtype=np.float32)
    altered = raw.copy()
    altered[32:, 0] += 10_000
    mean, scale = np.zeros(RAW_WIDTH), np.ones(RAW_WIDTH)
    first = build_a_features(raw, mean, scale)
    second = build_a_features(altered, mean, scale)
    assert np.array_equal(first[:32], second[:32])
    assert np.array_equal(first[0, RAW_WIDTH:-1], np.zeros(336))
    assert np.array_equal(build_a_features(raw[:16], mean, scale)[0], first[0])
    assert not np.array_equal(first[1, RAW_WIDTH:-1], first[0, RAW_WIDTH:-1])


def test_fit_normalizer_uses_only_fit_scored_stride_rows_and_floors_constants() -> None:
    fit_raw = np.broadcast_to(np.full((1, RAW_WIDTH), 2.5, dtype=np.float32), (20_000, RAW_WIDTH))
    eval_raw = np.broadcast_to(np.full((1, RAW_WIDTH), -80, dtype=np.float32), (20_000, RAW_WIDTH))
    scored = np.arange(20_000) >= 99
    sequences = [
        {"role": "fit", "component": component, "x": fit_raw, "scored": scored}
        for component in range(64)
    ]
    sequences.append({"role": "eval", "component": 100, "x": eval_raw, "scored": scored})
    mean, scale, rows = fit_raw_normalizer(sequences)
    assert rows == 31_808
    assert np.allclose(mean, 2.5)
    assert np.array_equal(scale, np.full(RAW_WIDTH, 1e-6))


def test_analytic_map_preserves_rows_objective_and_optimum() -> None:
    rng = np.random.default_rng(1401)
    mean, scale = np.zeros(RAW_WIDTH), np.ones(RAW_WIDTH)
    raw = rng.normal(size=(24, RAW_WIDTH)).astype(np.float32)
    design_a = build_a_features(raw, mean, scale)
    gram, whitener, transform, error = coordinate_map()
    design_b = design_a @ transform.T
    beta_b = rng.normal(scale=0.01, size=(449, 2))
    assert error < 1e-12
    assert np.max(np.abs(whitener @ gram @ whitener.T - np.eye(3))) < 1e-12
    mapped_prediction = design_a @ (transform.T @ beta_b[:, 0])
    native_prediction = design_b @ beta_b[:, 0]
    direct_error = direct_map_check(design_a, beta_b[:, 0], transform, whitener)
    assert direct_error == pytest.approx(np.max(np.abs(native_prediction - mapped_prediction)))
    assert direct_error < 1e-10
    targets = rng.normal(size=(len(raw), 2)).astype(np.float64)
    systems_a = build_systems(design_a, targets)
    systems_b = mapped_systems(systems_a, transform)
    for target in range(2):
        mapped_beta = transform.T @ beta_b[:, target]
        prediction_a = design_a @ mapped_beta
        prediction_b = design_b @ beta_b[:, target]
        assert np.allclose(prediction_a, prediction_b, rtol=1e-12, atol=1e-12)
        assert _objective(systems_a[target], mapped_beta) == pytest.approx(
            _objective(systems_b[target], beta_b[:, target]), abs=1e-10
        )
        optimum_a = solve_optimum(systems_a[target])
        optimum_b = transform.T @ solve_optimum(systems_b[target])
        assert np.linalg.norm(optimum_a - optimum_b) / np.linalg.norm(optimum_a) < 1e-7


def test_steepest_descent_reports_initial_energy_and_bounded_progress() -> None:
    rng = np.random.default_rng(1402)
    design = rng.normal(scale=0.03, size=(32, 449))
    design[:, -1] = 1.0
    targets = rng.normal(size=(32, 2))
    trajectory = steepest_descent(build_systems(design, targets)[0])
    assert len(trajectory.iterations) == 128
    assert [row["iteration"] for row in trajectory.checkpoints] == [0, 1, 2, 4, 8, 16, 32, 64, 128]
    assert trajectory.initial_gap == pytest.approx(trajectory.star_beta_h_beta)
    assert trajectory.initial_relative_gap == pytest.approx(1.0)
    assert _mean_defined([trajectory.checkpoints[-1]["relative_gap"]]) == pytest.approx(
        trajectory.checkpoints[-1]["relative_gap"]
    )
    assert _mean_defined([None, 0.5]) is None
    assert trajectory.checkpoints[0]["delta_h_delta"] == pytest.approx(trajectory.initial_gap)
    assert trajectory.checkpoints[-1]["relative_gap"] <= 1.0
    assert all(row["finite"] for row in trajectory.iterations)
    assert trajectory.nonzero_steps + trajectory.no_op_steps == 128


def test_canonical_scorer_component_merge_matches_direct_score() -> None:
    targets = np.asarray([[1.0, -0.5], [4.0, 0.25], [-1.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    predictions = np.asarray([[0.4, -0.3], [3.0, 0.0], [-0.5, 0.2], [20.0, 0.7]], dtype=np.float32)
    mask = np.asarray([True, True, True, False])
    accumulator = GlobalAccumulator()
    for component, row in ((10, slice(0, 2)), (11, slice(2, 4))):
        accumulator.add(targets[row], predictions[row], mask[row], group=component)
    pooled, direct = safe_result(accumulator), global_wp(targets, predictions, mask)
    assert pooled["selected_rows"] == direct["selected_rows"] == 3
    for metric in ("t0", "t1", "wp"):
        assert pooled[metric] == pytest.approx(direct[metric], abs=1e-12)
    parts = {}
    for component, row in ((10, slice(0, 2)), (11, slice(2, 4))):
        part = GlobalAccumulator()
        part.add(targets[row], predictions[row], mask[row])
        parts[component] = part
    assert metric_without(parts, 11)["selected_rows"] == 2
    assert metric_without(parts, 10)["selected_rows"] == 1


def test_actual_producer_fixtures_are_metadata_only_and_pinned() -> None:
    fixtures = {
        "REMOTE_SAMPLE.json": "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64",
        "REMOTE_MEMBERSHIP.json": "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217",
    }
    root = CAMPAIGN / "idea_screens_r3" / "records" / "sample_v2" / "terminal_capture"
    sample = json.loads((root / "REMOTE_SAMPLE.json").read_bytes())
    membership = json.loads((root / "REMOTE_MEMBERSHIP.json").read_bytes())
    assert sample["schema"] == "idea_screens_r3.sample.v2"
    assert membership["schema"] == "idea_screens_r3.membership.v2"
    assert len(sample["records"]) == len(membership["records"]) == 96
    assert len(sample["selection"]["fit_full64_component_order"]) == 64
    assert len(sample["selection"]["eval32_component_order"]) == 32
    for name, expected in fixtures.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected


def test_python_m_entrypoint_reaches_only_help_path() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CAMPAIGN)
    result = subprocess.run(
        [sys.executable, "-m", "idea_screens_r14.run_probe", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--expected-preparation-manifest-sha256" in result.stdout
    assert "R14 with python -m" not in result.stderr


def test_cold_import_smoke_keeps_payload_readers_untouched(tmp_path: Path) -> None:
    manifest_path = R14 / "PREPARATION_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("source closure smoke runs after the final manifest is generated")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    script = """
import importlib, sys
import numpy as np
import pyarrow.parquet as pq
def forbidden(*args, **kwargs):
    raise AssertionError('payload reader called during source-only smoke')
np.load = forbidden
np.memmap = forbidden
pq.ParquetFile = forbidden
pq.read_table = forbidden
module = importlib.import_module('idea_screens_r14.run_probe')
inventory = module.smoke_imports(__import__('pathlib').Path(sys.argv[1]), __import__('pathlib').Path(sys.argv[2]), __import__('pathlib').Path(sys.argv[3]), sys.argv[4])
assert len(inventory) == 20
print('source-only closure PASS; 20 modules; payload sentinels untouched')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(CAMPAIGN), str(ROOT / "src")))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(CAMPAIGN),
            str(ROOT),
            str(manifest_path),
            manifest_sha,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "payload sentinels untouched" in result.stdout
