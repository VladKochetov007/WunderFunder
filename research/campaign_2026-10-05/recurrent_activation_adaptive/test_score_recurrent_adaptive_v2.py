"""Synthetic guards for E479 grader v2's fit manifest metadata contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import score_recurrent_adaptive_v2 as grader


def test_common_batch_chunk_are_manifest_scoped() -> None:
    """Actual actor shape omits per-arm batch/chunk fields by design."""
    report = {"batch": 4, "chunk_rows": 256}
    grader.validate_common_cadence(report)
    grader.validate_optional_arm_cadence(
        {"seed": 271828, "optimizer": "adamw", "condition": "fixed0"},
        (271828, "adamw", "fixed0"),
    )
    grader.validate_optional_arm_cadence(
        {"batch": 4, "chunk_rows": 256}, (271828, "adamw", "fixed0")
    )
    for wrong in ({"batch": 8, "chunk_rows": 256}, {"batch": 4, "chunk_rows": 128}):
        try:
            grader.validate_optional_arm_cadence(wrong, (271828, "adamw", "fixed0"))
        except ValueError:
            pass
        else:
            raise AssertionError("contradictory per-arm cadence was accepted")
    for wrong in ({"batch": 8, "chunk_rows": 256}, {"batch": 4, "chunk_rows": 128}, {}):
        try:
            grader.validate_common_cadence(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("missing or wrong manifest cadence was accepted")


def test_normalizer_identity_mismatch_fails_closed() -> None:
    """A manifest cannot silently substitute another fit normalizer."""
    mean = np.linspace(-1, 1, grader.INPUTS, dtype=np.float32)
    scale = np.linspace(0.1, 2, grader.INPUTS, dtype=np.float32)
    grader.verify_normalizer_arrays(mean.copy(), scale.copy(), mean, scale)
    changed = scale.copy()
    changed[7] *= 1.01
    try:
        grader.verify_normalizer_arrays(mean, changed, mean, scale)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched normalizer was accepted")


def test_output_guard_precedes_any_scoring_work() -> None:
    """Both output destinations are checked before receipt or data validation."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "eda" / "RESULT.json"
        resolved, npz = grader.require_fresh_outputs(output)
        assert resolved.is_absolute() and npz.is_absolute()
        npz.parent.mkdir(parents=True)
        output.touch()
        try:
            grader.require_fresh_outputs(output)
        except ValueError:
            pass
        else:
            raise AssertionError("existing result path was not rejected")
