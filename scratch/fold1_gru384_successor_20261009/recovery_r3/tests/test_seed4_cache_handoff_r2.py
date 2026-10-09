"""The seed4 cache handoff requires one exact successful R3 predecessor."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "launch_seed4_after_r3_r2.py"
SPEC = importlib.util.spec_from_file_location("seed4_after_r3_r2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def predecessor() -> tuple[dict, dict, dict]:
    audit = {
        "schema": "wnn.fold1_gru384_resume_terminal_audit.r3",
        "status": "recovery_fit_and_output_passed",
        "attempt_id": "fold1-gru384-s3-resume-r3",
        "authority_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "attempt_sha256": "c" * 64,
        "output_audit": {"training_state_sha256": "d" * 64},
    }
    attempt = {
        "id": audit["attempt_id"],
        "kind": "gpu",
        "status": "finished",
        "exit_status": 0,
        "unit": "wnn-bandit-fold1-gru384-s3-resume-r3.service",
        "contract_sha256": audit["authority_sha256"],
    }
    unit = {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
    return audit, attempt, unit


def validate(values: tuple[dict, dict, dict]) -> None:
    audit, attempt, unit = values
    WRAPPER.require_successful_predecessor(
        audit,
        attempt,
        unit,
        attempt_id="fold1-gru384-s3-resume-r3",
        authority_sha256="a" * 64,
        request_sha256="b" * 64,
        attempt_sha256="c" * 64,
        unit_name="wnn-bandit-fold1-gru384-s3-resume-r3.service",
        recovered_state_sha256="d" * 64,
    )


def test_missing_or_failed_r3_predecessor_fails_closed() -> None:
    audit, attempt, unit = predecessor()
    with pytest.raises(ValueError, match="successful R3 terminal audit"):
        WRAPPER.require_successful_predecessor(
            None,
            attempt,
            unit,
            attempt_id=audit["attempt_id"],
            authority_sha256=audit["authority_sha256"],
            request_sha256=audit["request_sha256"],
            attempt_sha256=audit["attempt_sha256"],
            unit_name=attempt["unit"],
            recovered_state_sha256=audit["output_audit"]["training_state_sha256"],
        )
    audit["status"] = "recovery_fit_failed"
    with pytest.raises(ValueError, match="successful R3 terminal audit"):
        validate((audit, attempt, unit))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("attempt", "exact successful admitted predecessor"),
        ("unit", "successful terminal unit"),
        ("audit_hash", "exact predecessor attempt and state"),
        ("state_hash", "exact predecessor attempt and state"),
    ],
)
def test_mismatched_predecessor_identity_is_rejected(target: str, message: str) -> None:
    audit, attempt, unit = predecessor()
    if target == "attempt":
        attempt["status"] = "running"
    elif target == "unit":
        unit["ExecMainStatus"] = "1"
    elif target == "audit_hash":
        audit["attempt_sha256"] = "e" * 64
    else:
        audit["output_audit"]["training_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match=message):
        validate((audit, attempt, unit))


def test_exact_successful_r3_predecessor_passes() -> None:
    validate(predecessor())


def test_actual_attempt_parent_limits_use_kernel_page_rounding() -> None:
    root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    attempt = json.loads(
        (
            root
            / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/fold1-gru384-s3-resume-r3.json"
        ).read_text()
    )
    group, limits = WRAPPER.campaign_parent_cgroup(attempt, Path("/sys/fs/cgroup"), 5)
    page = os.sysconf("SC_PAGE_SIZE")
    assert group.name == "wnn-campaign.slice"
    assert limits["memory_high"] == 18_000_000_000 // page * page
    assert limits["memory_max"] == 20_000_000_000 // page * page


def test_actual_packet_nests_resume_output_path() -> None:
    packet = json.loads((SOURCE.parent / "RECOVERY_GATE_PACKET_R3.json").read_text())
    request = json.loads((SOURCE.parent / "RESUME_REQUEST_R3.json").read_text())
    assert packet["resume"]["output_path"] == request["resume_output_path"]


def test_file_cache_reclaim_is_bounded_and_targets_parent_headroom() -> None:
    requested = WRAPPER.bounded_reclaim_bytes(10_000_000_000, 9_000_000_000)
    assert requested == 1_500_000_000
    assert requested <= WRAPPER.MAX_RECLAIM_BYTES
    assert requested <= 9_000_000_000 - WRAPPER.PRESERVED_FILE_CACHE_BYTES
    assert 10_000_000_000 - requested == WRAPPER.HANDOFF_TARGET_BYTES

    assert WRAPPER.bounded_reclaim_bytes(8_000_000_000, 20_000_000_000) == 0
    assert WRAPPER.bounded_reclaim_bytes(10_000_000_000, 5_000_000_000) == 0
    assert WRAPPER.bounded_reclaim_bytes(20_000_000_000, 20_000_000_000) == (
        WRAPPER.MAX_RECLAIM_BYTES
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        WRAPPER.bounded_reclaim_bytes(-1, 10_000_000_000)
