from __future__ import annotations

import pytest

from scripts.kaggle_qat_kernel_r1 import execute_preflight_barrier


def test_candidate_phase_runs_only_after_preflight_verification() -> None:
    """Keep profile/fit dispatch behind the separate runner and receipt barrier."""
    events: list[str] = []

    def run_preflight() -> dict[str, str]:
        events.append("preflight_runner_complete")
        return {"terminal": "complete"}

    def verify_preflight(result: dict[str, str]) -> None:
        assert result["terminal"] == "complete"
        events.append("input_receipt_pass")

    def run_candidates() -> dict[str, str]:
        events.append("candidate_runner_start")
        return {"terminal": "complete"}

    result = execute_preflight_barrier(run_preflight, verify_preflight, run_candidates)

    assert result == {"terminal": "complete"}
    assert events == ["preflight_runner_complete", "input_receipt_pass", "candidate_runner_start"]


def test_candidate_phase_does_not_run_when_preflight_fails() -> None:
    """Preserve a fail-closed barrier if either terminal check rejects preflight."""
    events: list[str] = []

    def run_preflight() -> dict[str, str]:
        events.append("preflight_runner_complete")
        return {"terminal": "complete"}

    def reject_receipt(result: dict[str, str]) -> None:
        assert result["terminal"] == "complete"
        events.append("input_receipt_failed")
        raise ValueError("input receipt did not pass")

    def run_candidates() -> dict[str, str]:
        events.append("candidate_runner_start")
        return {"terminal": "complete"}

    with pytest.raises(ValueError, match="input receipt did not pass"):
        execute_preflight_barrier(run_preflight, reject_receipt, run_candidates)

    assert events == ["preflight_runner_complete", "input_receipt_failed"]
