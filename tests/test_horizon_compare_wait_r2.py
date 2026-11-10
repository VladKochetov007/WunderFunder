"""CPU checks for ownership and fixed-endpoint admission in the one-shot repair."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/transformer-optimizer-horizon-20261108/receiving/wait_compare_r2.py"
)
SPEC = importlib.util.spec_from_file_location("horizon_waiter_r2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WAITER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAITER)


def _stat(root: Path, state: str, birth: int) -> None:
    folder = root / str(WAITER.RECEIVER_PID)
    folder.mkdir(parents=True, exist_ok=True)
    fields = [state, *("0" for _ in range(18)), str(birth)]
    (folder / "stat").write_text(
        f"{WAITER.RECEIVER_PID} (receiver with spaces) " + " ".join(fields)
    )


def test_waiter_never_grades_while_the_original_identity_is_running(tmp_path: Path) -> None:
    """PID reuse and zombie exit differ from the exact live receiver identity."""
    assert WAITER._receiver_observation(tmp_path) == {"observed_status": "absent"}
    _stat(tmp_path, "S", WAITER.RECEIVER_BIRTH)
    assert WAITER._receiver_observation(tmp_path) is None
    _stat(tmp_path, "Z", WAITER.RECEIVER_BIRTH)
    assert WAITER._receiver_observation(tmp_path)["observed_status"] == "exited_zombie"
    _stat(tmp_path, "S", WAITER.RECEIVER_BIRTH + 1)
    assert WAITER._receiver_observation(tmp_path)["observed_status"] == "pid_reused_different_birth"


def test_receiver_exit_does_not_admit_partial_fits_to_comparison(tmp_path: Path) -> None:
    """All six full endpoints are required; a stopped fit must remain incomplete."""
    plan = {"runtime": str(tmp_path), "cells": [{"cell": str(i)} for i in range(6)]}
    for cell in plan["cells"]:
        folder = tmp_path / "runs" / cell["cell"] / "fit"
        folder.mkdir(parents=True)
        (folder / "terminal_status.json").write_text(
            json.dumps({"status": "completed", "completed_epochs": 4, "global_updates": 3280})
        )
        (folder / "score.json").write_text(
            json.dumps({"sequences": 1873, "score": {"selected_rows": 3528273}})
        )
    WAITER._completed_cells(plan)
    terminal = tmp_path / "runs/5/fit/terminal_status.json"
    terminal.write_text(
        json.dumps({"status": "stopped", "completed_epochs": 3, "global_updates": 2460})
    )
    with pytest.raises(ValueError, match="incomplete fixed endpoint"):
        WAITER._completed_cells(plan)


def test_exit_receipt_preserves_existing_evidence(tmp_path: Path) -> None:
    """Reobserving an exit cannot replace a previous or differently owned proof."""
    proof = WAITER._exit_receipt(tmp_path, {"observed_status": "absent"})
    original = proof.read_bytes()
    WAITER._exit_receipt(tmp_path, {"observed_status": "pid_reused_different_birth"})
    assert proof.read_bytes() == original
    wrong = json.loads(original)
    wrong["receiver_pid"] += 1
    proof.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="another identity"):
        WAITER._exit_receipt(tmp_path, {"observed_status": "absent"})
