"""Small synthetic invariants for the fixed R16 input-only screen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from idea_screens_r10.panel_io import InputBinding, input_file_receipts, validate_metadata_objects

from idea_screens_r16.estimator import (
    _score_pair,
    edge_is_stable,
    fit_input_transform,
    fit_source_residualizer,
)
from idea_screens_r16.linear_ops import (
    clip_pca_roundoff,
    fit_ridge_with_intercept,
    symmetric_gram_pseudoinverse,
)
from idea_screens_r16.panel import selected_innovations
from idea_screens_r16.run_probe import _cli_module

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research" / "campaign_2026-10-19"
FIXTURES = CAMPAIGN / "idea_screens_r3" / "records" / "sample_v2" / "terminal_capture"


def test_selected_innovations_use_only_fixed_past_and_current_rows() -> None:
    """The response is one-step and predictors never include the current row."""
    x = np.arange(20 * 112, dtype=np.float32).reshape(20, 112)
    scored = np.zeros(20, dtype=bool)
    scored[[4, 8]] = True
    past, delta, rows = selected_innovations(x, scored, stride=4)
    np.testing.assert_array_equal(rows, [4, 8])
    np.testing.assert_array_equal(past, x[[3, 7]])
    np.testing.assert_array_equal(delta, x[[4, 8]] - x[[3, 7]])
    future_changed = x.copy()
    future_changed[9:] += 1_000_000
    later_past, later_delta, later_rows = selected_innovations(future_changed, scored, stride=4)
    np.testing.assert_array_equal(later_rows, rows)
    np.testing.assert_array_equal(later_past, past)
    np.testing.assert_array_equal(later_delta, delta)


def test_fit_transform_uses_training_rows_only() -> None:
    """Changing held inputs cannot alter any fitted scaler or PCA parameter."""
    rng = np.random.default_rng(1601)
    train = rng.normal(size=(48, 112))
    held = rng.normal(size=(12, 112))
    train_view, held_view, state = fit_input_transform(train, held)
    changed_train_view, changed_held_view, changed_state = fit_input_transform(train, held + 1e6)
    assert np.isfinite(train_view).all() and np.isfinite(held_view).all()
    for key in ("input_mean", "input_scale", "pca_center", "pca_basis", "pca_eigenvalues"):
        np.testing.assert_array_equal(state[key], changed_state[key])
    np.testing.assert_array_equal(train_view, changed_train_view)
    assert not np.array_equal(held_view, changed_held_view)


def test_symmetric_gram_pseudoinverse_uses_frozen_cutoff() -> None:
    """Small positive modes under the relative cutoff are treated as zero."""
    gram = np.diag([4.0, 4e-11, 0.0])
    pinv, report = symmetric_gram_pseudoinverse(gram)
    assert report["rank"] == 1
    np.testing.assert_allclose(pinv, np.diag([0.25, 0.0, 0.0]))
    with pytest.raises(FloatingPointError, match="negative eigenvalue"):
        symmetric_gram_pseudoinverse(np.diag([1.0, -1e-8]))


def test_pca_keeps_small_positive_modes_below_gram_cutoff() -> None:
    """The residualizer Gram cutoff does not truncate positive PCA modes."""
    values, tolerance = clip_pca_roundoff(np.array([1.0, 1e-12, -1e-12]))
    assert tolerance == 1e-10
    np.testing.assert_array_equal(values, [1.0, 1e-12, 0.0])


def test_one_binding_per_component_yields_two_input_file_receipts() -> None:
    """R10 returns64 bindings, each for one x and one scored-mask file."""
    bindings = []
    for component in range(64):
        record = {
            "role": "fit",
            "source": "train",
            "group": component,
            "component": component,
            "seq_ix": component,
        }
        bindings.append(
            InputBinding(
                record,
                Path(f"train_{component}_x.npy"),
                "a" * 64,
                100,
                Path(f"train_{component}_scored.npy"),
                "b" * 64,
                200,
            )
        )
    receipts = input_file_receipts(bindings)
    assert len(bindings) == 64
    assert len(receipts) == 128
    assert [item["key"] for item in receipts[:2]] == ["x", "scored"]


def test_residualizer_fits_only_control_span_and_reports_scale() -> None:
    """The source residual is finite after fixed Z projection and RMS scaling."""
    rng = np.random.default_rng(1602)
    z = rng.normal(size=(60, 5))
    source = z @ rng.normal(size=(5, 3)) + 0.05 * rng.normal(size=(60, 3))
    held_z = rng.normal(size=(20, 5))
    held_source = held_z @ rng.normal(size=(5, 3))
    train_r, held_r, gamma, diagnostics = fit_source_residualizer(z, source, held_z, held_source)
    assert gamma.shape == (5, 3)
    assert train_r.shape == source.shape and held_r.shape == held_source.shape
    assert np.isfinite(train_r).all() and np.isfinite(held_r).all()
    assert diagnostics["rank"] >= 1
    assert diagnostics["source_residual_scale_floor_count"] == 0


def test_ridge_intercept_is_unpenalized() -> None:
    """A fitted affine relation is recovered without regularizing its mean."""
    z = np.linspace(-3.0, 3.0, 101, dtype=np.float64)[:, None]
    y = np.full_like(z, 7.0)
    beta, mean, response_mean = fit_ridge_with_intercept(z, y)
    prediction = response_mean + (z - mean) @ beta
    np.testing.assert_allclose(prediction, y, atol=1e-12)
    np.testing.assert_allclose(response_mean, [7.0])


def test_component_sse_sums_and_undefined_denominator_fail_closed() -> None:
    """All component rows are retained and a zero baseline cannot pass the gate."""
    target = np.array([[1.0], [2.0], [3.0], [4.0]])
    base = np.zeros_like(target)
    corrected = target * 0.5
    report = _score_pair(target, base, corrected, {10: (0, 2), 20: (2, 4)}, "i0")
    assert report["baseline_sse"] == sum(c["baseline_sse"] for c in report["components"])
    assert report["corrected_sse"] == sum(c["corrected_sse"] for c in report["components"])
    assert edge_is_stable(
        {"relative_sse_reduction": 0.5, "component_median_relative_sse_reduction": 0.5},
        {"relative_sse_reduction": 0.5, "component_median_relative_sse_reduction": 0.5},
    )
    undefined = _score_pair(
        np.zeros((4, 1)),
        np.zeros((4, 1)),
        np.zeros((4, 1)),
        {10: (0, 2), 20: (2, 4)},
        "i0",
    )
    assert undefined["relative_sse_reduction"] is None
    assert undefined["component_median_relative_sse_reduction"] is None
    assert not edge_is_stable(undefined, undefined)
    assert not edge_is_stable(
        {"relative_sse_reduction": float("nan"), "component_median_relative_sse_reduction": 1.0},
        {"relative_sse_reduction": 1.0, "component_median_relative_sse_reduction": 1.0},
    )
    with pytest.raises(FloatingPointError, match="SSE terms overflowed"):
        _score_pair(
            np.array([[1e200]]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            {10: (0, 1)},
            "i0",
        )


def test_actual_r3_metadata_fixture_matches_fixed_fit_membership() -> None:
    """The exact producer fixture validates without opening any array payload."""
    sample = json.loads((FIXTURES / "REMOTE_SAMPLE.json").read_bytes())
    membership = json.loads((FIXTURES / "REMOTE_MEMBERSHIP.json").read_bytes())
    panel = validate_metadata_objects(sample, membership)
    assert len(panel.fit_component_order) == 64
    assert len(panel.eval_component_order) == 32
    assert len(panel.fit_records) == 64


def test_cli_module_accepts_exact_python_m_identity_only() -> None:
    """The cold CLI accepts its exact __main__ spec and rejects another entrypoint."""
    module = ModuleType("__main__")
    module.__spec__ = ModuleSpec("idea_screens_r16.run_probe", loader=None)
    module.__file__ = str(CAMPAIGN / "idea_screens_r16" / "run_probe.py")
    assert _cli_module({"__main__": module}) is module
    module.__spec__ = ModuleSpec("wrong.module", loader=None)
    with pytest.raises(RuntimeError, match="python -m"):
        _cli_module({"__main__": module})


def test_source_only_import_smoke_and_help_read_no_payloads() -> None:
    """Fresh source imports match the pinned closure while NPY entrypoints are sentinels."""
    script = r"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
def forbidden(*args, **kwargs):
    raise AssertionError("panel payload reader reached during import smoke")
np.load = forbidden
np.memmap = forbidden
from idea_screens_r16.run_probe import smoke_imports
campaign = Path(os.environ["R16_CAMPAIGN_ROOT"])
manifest = campaign / "idea_screens_r16" / "PREPARATION_MANIFEST.json"
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
inventory = smoke_imports(campaign, manifest, digest)
assert "torch" not in sys.modules and "pyarrow" not in sys.modules
assert len(inventory) == 9
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CAMPAIGN)
    env["R16_CAMPAIGN_ROOT"] = str(CAMPAIGN)
    subprocess.run([sys.executable, "-B", "-c", script], env=env, check=True, timeout=30)
    help_run = subprocess.run(
        [sys.executable, "-B", "-m", "idea_screens_r16.run_probe", "--help"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "--expected-preparation-manifest-sha256" in help_run.stdout
