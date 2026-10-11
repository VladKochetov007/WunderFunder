from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import research_monitor_wakeup as wakeup

THREAD_ID = "01a12a30-1630-77e1-b9e2-28cd0aa662ea"
PROMPT = "Resume the research session from its latest handoff."
QUEUE_ID = "01a12a59-64f0-7123-814e-1b256814ad2a"


def make_config(tmp_path: Path) -> tuple[Path, wakeup.MonitorConfig]:
    cwd = tmp_path / "workspace"
    cwd.mkdir(parents=True)
    prompt_path = tmp_path / "wake.md"
    prompt_path.write_text(PROMPT, encoding="utf-8")
    codex_path = tmp_path / "codex"
    codex_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex_path.chmod(0o700)
    state_dir = tmp_path / "state"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": wakeup.CONFIG_SCHEMA,
                "thread_id": THREAD_ID,
                "cwd": str(cwd),
                "prompt_path": str(prompt_path),
                "codex_path": str(codex_path),
                "state_dir": str(state_dir),
            }
        ),
        encoding="utf-8",
    )
    return config_path, wakeup.load_config(config_path)


def queued_receipt(accepted_at: datetime) -> dict[str, object]:
    return {
        "schema": wakeup.QUEUE_SCHEMA,
        "thread_id": THREAD_ID,
        "accepted_utc": wakeup.utc_text(accepted_at),
    }


def completion_receipt(completed_at: datetime) -> dict[str, object]:
    return {"completed_utc": wakeup.utc_text(completed_at)}


def write_receipt(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dry_run_prints_metadata_without_mutating_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, config = make_config(tmp_path)

    assert wakeup.main(["--config", str(config_path), "--dry-run"]) == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "dry_run"
    assert output["thread_id"] == THREAD_ID
    assert output["prompt_bytes"] == len(PROMPT.encode("utf-8"))
    assert PROMPT not in captured.out
    assert not config.state_dir.exists()


def test_queue_uses_exact_thread_message_and_safe_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = make_config(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []
    fixed_time = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)
    monkeypatch.setattr(wakeup, "utc_now", lambda: fixed_time)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        output = f"Queued message {QUEUE_ID} for thread {THREAD_ID}.\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(wakeup.subprocess, "run", fake_run)

    assert wakeup.run_wakeup(config, PROMPT) == 0

    assert len(calls) == 1
    command, options = calls[0]
    assert command == [
        str(config.codex_path),
        "queue",
        "--thread",
        THREAD_ID,
        "--message",
        PROMPT,
    ]
    assert options["cwd"] == config.cwd
    assert options["timeout"] == 90
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["env"] == {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
    }
    receipt = json.loads((config.state_dir / "LAST_QUEUED.json").read_text())
    event = json.loads((config.state_dir / "events.jsonl").read_text().splitlines()[0])
    assert receipt["accepted_utc"] == wakeup.utc_text(fixed_time)
    assert receipt["queue_id"] == QUEUE_ID
    assert receipt["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert event["exit_code"] == 0
    assert event["stdout_summary"]["bytes"] == len(
        f"Queued message {QUEUE_ID} for thread {THREAD_ID}.\n"
    )
    assert PROMPT not in json.dumps(receipt)
    assert PROMPT not in json.dumps(event)


def test_recent_unacknowledged_wakeup_is_coalesced_and_two_hour_boundary_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = make_config(tmp_path)
    queued_at = datetime(2026, 10, 11, 6, 0, tzinfo=UTC)
    write_receipt(config.state_dir / "LAST_QUEUED.json", queued_receipt(queued_at))
    now = [queued_at + timedelta(hours=2) - timedelta(microseconds=1)]
    monkeypatch.setattr(wakeup, "utc_now", lambda: now[0])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "accepted", "")

    monkeypatch.setattr(wakeup.subprocess, "run", fake_run)

    assert wakeup.run_wakeup(config, PROMPT) == 0
    assert calls == []
    skipped = json.loads((config.state_dir / "events.jsonl").read_text().splitlines()[0])
    assert skipped["status"] == "skipped_pending"
    assert skipped["last_wake_age_seconds"] == 7199.999999

    now[0] = queued_at + timedelta(hours=2)
    assert wakeup.run_wakeup(config, PROMPT) == 0
    assert len(calls) == 1


def test_only_completion_ack_newer_than_queue_releases_pending_wakeup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = make_config(tmp_path)
    queued_at = datetime(2026, 10, 11, 8, 0, tzinfo=UTC)
    now = [queued_at + timedelta(seconds=10)]
    write_receipt(config.state_dir / "LAST_QUEUED.json", queued_receipt(queued_at))
    write_receipt(config.state_dir / "LAST_COMPLETED.json", completion_receipt(queued_at))
    monkeypatch.setattr(wakeup, "utc_now", lambda: now[0])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(wakeup.subprocess, "run", fake_run)

    assert wakeup.run_wakeup(config, PROMPT) == 0
    assert calls == []
    now[0] = queued_at + timedelta(seconds=11)
    write_receipt(
        config.state_dir / "LAST_COMPLETED.json",
        completion_receipt(queued_at + timedelta(microseconds=1)),
    )
    assert wakeup.run_wakeup(config, PROMPT) == 0
    assert len(calls) == 1


def test_disabled_marker_and_overlapping_lock_suppress_cli_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, disabled_config = make_config(tmp_path / "disabled")
    disabled_config.state_dir.mkdir()
    (disabled_config.state_dir / "DISABLED").touch()
    _, locked_config = make_config(tmp_path / "locked")
    locked_config.state_dir.mkdir()
    lock_fd = os.open(locked_config.state_dir / "wakeup.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(
        wakeup.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("disabled or overlapping wakeup queued"),
    )
    try:
        assert wakeup.run_wakeup(disabled_config, PROMPT) == 0
        assert wakeup.run_wakeup(locked_config, PROMPT) == 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    statuses = {
        json.loads((item.state_dir / "events.jsonl").read_text().splitlines()[0])["status"]
        for item in (disabled_config, locked_config)
    }
    assert statuses == {"skipped_disabled", "skipped_overlap"}


def test_nonzero_queue_exit_is_logged_without_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = make_config(tmp_path)
    monkeypatch.setattr(
        wakeup.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 7, "", "denied"),
    )

    assert wakeup.run_wakeup(config, PROMPT) == 7

    assert not (config.state_dir / "LAST_QUEUED.json").exists()
    event = json.loads((config.state_dir / "events.jsonl").read_text().splitlines()[0])
    assert event["status"] == "queue_failed"
    assert event["exit_code"] == 7
    assert event["stderr_summary"]["bytes"] == len("denied")


@pytest.mark.parametrize(
    "override",
    [
        {"thread_id": "not-a-uuid"},
        {"unexpected": "field"},
    ],
)
def test_config_rejects_invalid_thread_and_undocumented_fields(
    tmp_path: Path, override: dict[str, str]
) -> None:
    config_path, _ = make_config(tmp_path)
    value = json.loads(config_path.read_text())
    value.update(override)
    config_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        wakeup.load_config(config_path)
