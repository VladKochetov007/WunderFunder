from __future__ import annotations

from pathlib import Path

import pytest

from scripts.l4_teacher_capacity_20261104.grade_stage2_cache import _teacher_fit_output


def test_transformer_r4_output_leaf_fit_is_bound_by_run_and_frozen_argv(
    tmp_path: Path,
) -> None:
    run = "l4tc_transformer_d384_l8_h8_s1001_20261104"
    output = tmp_path / "transformer" / "s1001" / "fit"
    cell = {
        "run_name": run,
        "fit_output": str(output),
        "fit_argv": ["python", "train_sequence.py", "--output", str(output)],
    }

    assert _teacher_fit_output("causal_transformer", run, cell, {"paths": {}}) == output


def test_transformer_output_must_match_frozen_argv(tmp_path: Path) -> None:
    run = "l4tc_transformer_d384_l8_h8_s1001_20261104"
    output = tmp_path / "transformer" / "s1001" / "fit"
    cell = {
        "run_name": run,
        "fit_output": str(output),
        "fit_argv": ["python", "train_sequence.py", "--output", str(output / "other")],
    }

    with pytest.raises(ValueError, match="output must be absolute and match"):
        _teacher_fit_output("causal_transformer", run, cell, {"paths": {}})
