"""CPU checks for actual GRU and Transformer teacher-panel output schemas."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from scripts.l4_teacher_capacity_20261104.continue_students import teacher_fit_output

ROOT = Path(__file__).resolve().parents[1]


def _panel(name: str) -> dict:
    return json.loads(
        (ROOT / "scripts/l4_teacher_capacity_20261104" / name).read_text(encoding="utf-8")
    )


def test_native_gru_template_derives_output_without_changing_panel() -> None:
    panel = _panel("GRU_TEACHER_PANEL_R1.json")
    cell = panel["cells"][0]
    assert "fit_output" not in cell
    before = dict(cell)
    run = cell["run_name"]
    assert teacher_fit_output(panel, "wide_gru", run, cell) == (
        PurePosixPath(panel["paths"]["project_root"]) / "artifacts" / "runs" / run
    )
    assert cell == before


def test_transformer_template_requires_and_matches_explicit_output() -> None:
    panel = _panel("TRANSFORMER_TEACHER_PANEL_R1.json")
    cell = dict(panel["cells"][0])
    run = "l4tc_transformer_d384_l8_h8_s1001_20261104"
    cell["run_name"] = run
    cell["fit_output"] = str(
        PurePosixPath(panel["paths"]["runtime_root"]) / "transformer" / "s1001" / "fit"
    )
    cell["fit_argv"] = list(cell["fit_argv"])
    cell["fit_argv"][cell["fit_argv"].index("--output") + 1] = cell["fit_output"]
    assert PurePosixPath(cell["fit_output"]).name == "fit"
    assert teacher_fit_output(panel, "causal_transformer", run, cell) == PurePosixPath(
        cell["fit_output"]
    )
    missing = dict(cell)
    missing.pop("fit_output")
    with pytest.raises(ValueError, match="explicitly bind fit_output"):
        teacher_fit_output(panel, "causal_transformer", run, missing)
    with pytest.raises(ValueError, match="fit_output and frozen --output differ"):
        teacher_fit_output(panel, "causal_transformer", run + "_wrong", cell)


def test_gru_output_derivation_rejects_changed_name_or_schema() -> None:
    panel = _panel("GRU_TEACHER_PANEL_R1.json")
    cell = panel["cells"][0]
    with pytest.raises(ValueError, match="run/name/project identity"):
        teacher_fit_output(panel, "wide_gru", cell["run_name"] + "_other", cell)
    wrong_panel = dict(panel, schema="wnn.other.v1")
    with pytest.raises(ValueError, match="limited to the frozen GRU panel schema"):
        teacher_fit_output(wrong_panel, "wide_gru", cell["run_name"], cell)
