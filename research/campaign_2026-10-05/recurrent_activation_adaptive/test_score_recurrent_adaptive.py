"""Synthetic-only tests for the independent E479 scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import score_recurrent_adaptive as grader


def test_independent_synthetic_model_and_bootstrap() -> None:
    """Exercise tensor alpha, parameter layout, chronology, and all contrasts."""
    receipt = grader.self_check()
    assert receipt["real_data_opened"] is False
    assert receipt["real_checkpoint_loaded"] is False
    assert receipt["arms"] == 16
    assert receipt["contrasts"] == 28


def test_export_creation_is_atomic_and_no_overwrite() -> None:
    """The export writer creates fresh files and refuses an existing destination."""
    arrays = {"x": np.arange(5, dtype=np.int64)}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "paired.npz"
        grader.atomic_save_npz(path, arrays)
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["x"], arrays["x"])
        try:
            grader.atomic_save_npz(path, arrays)
        except ValueError:
            pass
        else:
            raise AssertionError("writer did not reject overwrite")


def test_receipt_and_normalizer_guards_fail_closed() -> None:
    """Synthetic receipts exercise unit, full spec, and exact normalization guards."""
    attempt = {"unit_properties": {"Result": "success", "ExecMainStatus": 0}}
    grader.verify_unit_success(attempt)
    for bad in (
        {"unit_properties": {"Result": "failed", "ExecMainStatus": 0}},
        {"unit_properties": {"Result": "success", "ExecMainStatus": 1}},
        {},
    ):
        try:
            grader.verify_unit_success(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid systemd unit receipt was accepted")

    seal_path = grader.HERE / "e479_sixteen_fit_v2.sealed.json"
    if not seal_path.is_file():
        seal_path = grader.HERE / "e479_sixteen_fit_v1.sealed.json"
    sealed = json.loads(seal_path.read_text())
    assert grader.validate_spec(sealed) == sealed
    broken = {**sealed, "kind": "not-a-runner-kind"}
    try:
        grader.validate_spec(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid sealed spec was accepted")

    mean = np.linspace(-1, 1, grader.INPUTS, dtype=np.float32)
    scale = np.linspace(0.1, 2, grader.INPUTS, dtype=np.float32)
    grader.verify_normalizer_arrays(mean.copy(), scale.copy(), mean, scale)
    changed = mean.copy()
    changed[3] += 0.01
    try:
        grader.verify_normalizer_arrays(changed, scale, mean, scale)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched fit-only normalizer was accepted")


def test_output_paths_resolve_and_refuse_existing_files() -> None:
    """Preflight checks catch both JSON and NPZ output collisions."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        target = root / "nested" / "RESULT.json"
        resolved, npz = grader.require_fresh_outputs(target)
        assert resolved.is_absolute() and npz.is_absolute()
        target.parent.mkdir(parents=True)
        npz.touch()
        try:
            grader.require_fresh_outputs(target)
        except ValueError:
            pass
        else:
            raise AssertionError("existing NPZ output was not rejected")
