from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "controller.py"
SPEC = importlib.util.spec_from_file_location("vast_teacher_tick_guard_controller", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


def _unavailable_case(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_response: str,
    expected_reason: str,
) -> None:
    ownership = {
        "attempt_id": "vast-teachers-r1-lane-b-20261010",
        "instance_id": 55080658,
        "state": "created",
    }
    calls: list[tuple[str, ...]] = []

    def fake_vast(*args: str, **kwargs: object) -> str:
        calls.append(args)
        return provider_response

    monkeypatch.setattr(controller, "LANE", "b")
    monkeypatch.setattr(controller, "receipt", lambda: ownership)
    monkeypatch.setattr(controller, "vast", fake_vast)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unavailable lane-B preflight must exit before pull/spend/remote work")

    for name in (
        "cmd_pull",
        "spend",
        "probe",
        "ssh",
        "cmd_feed",
        "cmd_collect",
        "cmd_destroy",
    ):
        monkeypatch.setattr(controller, name, forbidden)

    before = ownership.copy()
    controller.cmd_tick(Namespace(no_pull=False, cap=70.0))

    result = json.loads(capsys.readouterr().out)
    assert calls == [("show", "instance", "55080658")]
    assert result["reason"] == expected_reason
    assert result["hours"] is None
    assert result["cost"] is None
    assert result["drained"] is False
    assert result["collection_complete"] is False
    assert result["collection_required"] is True
    assert result["teardown_verified"] is False
    assert result["provider_mutation"] is False
    assert result["remote_access_attempted"] is False
    assert ownership == before


def test_null_instances_envelope_exits_before_tick_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _unavailable_case(
        monkeypatch,
        capsys,
        '{"instances": null}',
        "provider_returned_no_instance_rows",
    )


def test_flat_stopped_row_exits_before_tick_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    row = {
        "id": 55080658,
        "actual_status": "stopped",
        "start_date": 1_760_000_000,
        "dph_total": 0.5,
    }
    _unavailable_case(
        monkeypatch,
        capsys,
        json.dumps(row),
        "provider_owned_instance_not_running",
    )


def test_mismatched_running_row_exits_before_tick_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    row = {
        "id": 55080659,
        "actual_status": "running",
        "start_date": 1_760_000_000,
        "dph_total": 0.5,
    }
    _unavailable_case(
        monkeypatch,
        capsys,
        json.dumps({"instances": [row]}),
        "provider_owned_instance_row_missing_or_ambiguous",
    )


def test_matching_running_envelope_preserves_tick_path_and_spend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ownership = {
        "attempt_id": "vast-teachers-r1-lane-b-20261010",
        "instance_id": 55080658,
        "state": "created",
    }
    calls: list[tuple[str, ...]] = []
    row = {
        "id": "55080658",
        "actual_status": "running",
        "start_date": 1_000,
        "dph_total": 0.5,
    }
    monkeypatch.setattr(controller, "LANE", "b")
    monkeypatch.setattr(controller, "receipt", lambda: ownership)
    monkeypatch.setattr(
        controller,
        "vast",
        lambda *args, **kwargs: calls.append(args) or json.dumps({"instances": [row]}),
    )
    pulls: list[bool] = []
    monkeypatch.setattr(controller, "cmd_pull", lambda _: pulls.append(True))
    monkeypatch.setattr(controller, "spend", lambda: pytest.fail("lane B must reuse preflight row"))
    monkeypatch.setattr(controller.time, "time", lambda: 5_500)
    monkeypatch.setattr(controller, "probe", lambda: {})
    monkeypatch.setattr(controller, "_remote_state", lambda _: (1, 0, 1, 0, False))
    monkeypatch.setattr(controller, "ready_items", lambda _: [])
    monkeypatch.setattr(controller, "_remote_quiescence", lambda _: (False, "fixture busy"))

    controller.cmd_tick(Namespace(no_pull=False, no_auto_collect=True, cap=70.0))

    result = json.loads(capsys.readouterr().out.splitlines()[0])
    assert calls == [("show", "instance", "55080658")]
    assert pulls == [True]
    assert result["hours"] == pytest.approx(1.25)
    assert result["cost"] == pytest.approx(0.62)
    assert result["drained"] is False
