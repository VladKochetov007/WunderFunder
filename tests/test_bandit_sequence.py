"""Exercise sequencing decisions without training or launching a GPU process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bandit_sequence import run_sequence  # noqa: E402


def prepare(tmp_path: Path, status: str, exit_status: int) -> tuple[Path, Path]:
    """Create one observed predecessor and one unlaunched successor."""
    (tmp_path / "attempts").mkdir()
    (tmp_path / "attempts/first.json").write_text(
        json.dumps({"status": status, "exit_status": exit_status})
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "runtime": str(tmp_path),
                "steps": [{"attempt": "first"}, {"attempt": "second"}],
            }
        )
    )
    return plan, tmp_path / "receipt.json"


def test_failure_prevents_next_admission(tmp_path: Path) -> None:
    """A failed predecessor cannot be silently treated as completed work."""
    plan, receipt = prepare(tmp_path, "failed", 1)
    with pytest.raises(RuntimeError, match="first needs review"):
        run_sequence(plan, receipt)
    assert json.loads(receipt.read_text())["status"] == "needs_review"
    assert not (tmp_path / "attempts/second.json").exists()


def test_drain_preserves_success_and_blocks_successor(tmp_path: Path) -> None:
    """Closing admissions does not invalidate or restart a healthy predecessor."""
    plan, receipt = prepare(tmp_path, "finished", 0)
    (tmp_path / "STOP_ADMISSION.json").write_text("{}")
    run_sequence(plan, receipt)
    assert json.loads(receipt.read_text())["status"] == "drained"
    assert not (tmp_path / "attempts/second.json").exists()


def test_finished_with_failure_exit_is_not_success(tmp_path: Path) -> None:
    """Require both the terminal label and a successful process exit."""
    plan, receipt = prepare(tmp_path, "finished", 124)
    with pytest.raises(RuntimeError, match="first needs review"):
        run_sequence(plan, receipt)
