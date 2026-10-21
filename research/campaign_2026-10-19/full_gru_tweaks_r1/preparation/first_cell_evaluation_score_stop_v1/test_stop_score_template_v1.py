from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import STOP_SCORE_TEMPLATE_V1 as stop

INVOCATION = "0123456789abcdef0123456789abcdef"


def _receipt(**overrides: object) -> bytes:
    value: dict[str, object] = {
        "schema": stop.ACCEPTANCE_SCHEMA,
        "status": stop.ACCEPTANCE_STATUS,
        "unit": stop.SCORE_UNIT,
        "invocation_id": INVOCATION,
        "stop_status": "PENDING_EXACT_STOP",
        "native_terminal": {
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
        },
        "result": {
            "path": "/captured/RESULT.json",
            "bytes": 8000,
            "sha256": "a" * 64,
            "schema": stop.RESULT_SCHEMA,
            "status": stop.RESULT_STATUS,
        },
    }
    value.update(overrides)
    return json.dumps(value, sort_keys=True).encode()


def test_root_receipt_joins_strict_native_exit_and_compact_result() -> None:
    raw = _receipt()
    digest, result = stop._validate_receipt(
        raw, len(raw), hashlib.sha256(raw).hexdigest(), INVOCATION
    )
    assert digest == hashlib.sha256(raw).hexdigest()
    assert result["schema"] == stop.RESULT_SCHEMA


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "wrong"},
        {"status": "PENDING"},
        {"unit": "computer-vision-homework-fg19-first-cell-export-v3.service"},
        {"stop_status": "STOPPED"},
        {
            "native_terminal": {
                "active_state": "active",
                "sub_state": "exited",
                "main_pid": True,
                "exec_main_code": 1,
                "exec_main_status": 0,
            }
        },
        {
            "result": {
                "path": "/captured/RESULT.json",
                "bytes": 8,
                "sha256": "a" * 64,
                "schema": "wrong",
                "status": stop.RESULT_STATUS,
            }
        },
    ],
)
def test_wrong_acceptance_identity_or_result_is_rejected(change: dict[str, object]) -> None:
    raw = _receipt(**change)
    with pytest.raises(ValueError):
        stop._validate_receipt(raw, len(raw), hashlib.sha256(raw).hexdigest(), INVOCATION)


def test_acceptance_invocation_and_bytes_pin_are_strict() -> None:
    raw = _receipt()
    with pytest.raises(ValueError):
        stop._validate_receipt(raw, len(raw), hashlib.sha256(raw).hexdigest(), "f" * 32)
    with pytest.raises(ValueError):
        stop._validate_receipt(raw, len(raw) + 1, hashlib.sha256(raw).hexdigest(), INVOCATION)


def test_stop_state_predicates_accept_only_exact_success_and_absence() -> None:
    before = {
        "Id": stop.SCORE_UNIT,
        "InvocationID": INVOCATION,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    stop._require_before(before, INVOCATION)
    stop._require_after(
        {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "InvocationID": "",
        }
    )
    with pytest.raises(ValueError):
        stop._require_before({**before, "Id": "other.service"}, INVOCATION)
    with pytest.raises(RuntimeError):
        stop._require_after({**before, "ActiveState": "active"})


def test_unit_scope_and_cgroup_host_root_fallback_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        stop._show("computer-vision-homework-fg19-first-cell-export-v3.service")
    with pytest.raises(ValueError):
        stop._checked_cgroup("/", tmp_path)
    with pytest.raises(ValueError):
        stop._checked_cgroup("/user.slice/foreign.slice", tmp_path)
    group = tmp_path / "user.slice" / "wnn-campaign.slice"
    group.mkdir(parents=True)
    assert stop._checked_cgroup("/user.slice/wnn-campaign.slice", tmp_path) == group
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "user.slice", target_is_directory=True)
    with pytest.raises(ValueError):
        stop._checked_cgroup("/alias/wnn-campaign.slice", tmp_path)


def test_bounded_cgroup_scan_reports_population_and_empty_tree(tmp_path: Path) -> None:
    root = tmp_path / "user.slice" / "wnn-campaign.slice"
    child = root / "score.service"
    child.mkdir(parents=True)
    (root / "cgroup.procs").write_text("", encoding="ascii")
    (child / "cgroup.procs").write_text("", encoding="ascii")
    assert stop._scan_populated(root) == {}
    (child / "cgroup.procs").write_text("1234\n", encoding="ascii")
    assert stop._scan_populated(root) == {str(child): "1234"}
