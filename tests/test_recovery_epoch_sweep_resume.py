"""Prepared resume checks for R5; leave unrun outside the receiving GPU host."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_epoch_sweep_fixtures import _make_fit, _make_sweep

PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))

from epoch_sweep_resume import prepare_sweep_output  # noqa: E402


def test_resume_reuses_only_hash_verified_completed_epochs(tmp_path: Path) -> None:
    fit = _make_fit(tmp_path, 0)
    output = _make_sweep(tmp_path, fit, [0.1] * 30)
    plan = json.loads((output / "plan.json").read_text())
    state = prepare_sweep_output(output, plan, fit, fit["binding"], resume=True)
    assert len(state["records"]) == 30
    assert state["sweep"]["status"] == "complete"

    with (output / "epochs/epoch-000009/score.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="receipt changed"):
        prepare_sweep_output(output, plan, fit, fit["binding"], resume=True)


def test_resume_rejects_partial_epoch_and_plan_drift(tmp_path: Path) -> None:
    fit = _make_fit(tmp_path, 0)
    output = _make_sweep(tmp_path, fit, [0.1] * 30)
    plan = json.loads((output / "plan.json").read_text())
    (output / "epochs/epoch-000004/epoch.json").unlink()
    with pytest.raises(ValueError, match="partial"):
        prepare_sweep_output(output, plan, fit, fit["binding"], resume=True)

    other_root = tmp_path / "other"
    other_fit = _make_fit(other_root, 0)
    output = _make_sweep(other_root, other_fit, [0.1] * 30)
    plan = json.loads((output / "plan.json").read_text())
    plan["data_root"] = "/changed/data"
    with pytest.raises(ValueError, match="resume plan differs"):
        prepare_sweep_output(
            output,
            plan,
            other_fit,
            other_fit["binding"],
            resume=True,
        )
