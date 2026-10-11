from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from scripts import watchdog


def _run_job_diagnostics(run_root: Path) -> dict[str, object]:
    environment = os.environ.copy()
    environment["WNN_WATCHDOG_RUN_ROOT"] = str(run_root)
    result = subprocess.run(
        [sys.executable, "-c", watchdog.REMOTE_JOB_DIAGNOSTICS],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
        env=environment,
    )
    rows = [
        line.removeprefix(watchdog.REMOTE_DIAGNOSTICS_MARKER)
        for line in result.stdout.splitlines()
        if line.startswith(watchdog.REMOTE_DIAGNOSTICS_MARKER)
    ]
    assert len(rows) == 1
    return json.loads(rows[0])


def _write_job(run_root: Path, job: dict[str, object]) -> None:
    running = run_root / "refill" / "running"
    running.mkdir(parents=True, exist_ok=True)
    (running / "job.json").write_text(json.dumps(job))


def test_pipeline_stall_uses_exact_arm_log_when_another_pipeline_log_is_fresh(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_job(
        run_root,
        {
            "name": "w19_2_vt_f1_dsq_t256",
            "steps": [
                {
                    "label": "pipeline",
                    "argv": [
                        "python",
                        "/root/wnn/scripts/distill_pipeline.py",
                        "--arm",
                        "t256",
                    ],
                }
            ],
        },
    )
    logs = run_root / "logs"
    logs.mkdir()
    exact_log = logs / "pipeline_t256.log"
    exact_log.write_text("old pipeline progress\n")
    unrelated_log = logs / "pipeline_t384.log"
    unrelated_log.write_text("recent unrelated progress\n")
    stale_time = time.time() - 30 * 60
    os.utime(exact_log, (stale_time, stale_time))

    diagnostics = _run_job_diagnostics(run_root)
    job = diagnostics["running_jobs"][0]

    assert job["arm"] == "t256"
    assert job["log"] == "pipeline_t256.log"
    assert job["state"] == "present"
    assert job["age_min"] >= 30


def test_pipeline_missing_log_is_not_hidden_by_another_fresh_pipeline_log(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_job(
        run_root,
        {
            "name": "w19_3_vt_f1_dsq_t384",
            "steps": [
                {
                    "label": "pipeline",
                    "argv": [
                        "python",
                        "/root/wnn/scripts/distill_pipeline.py",
                        "--arm=t384",
                    ],
                }
            ],
        },
    )
    logs = run_root / "logs"
    logs.mkdir()
    (logs / "pipeline_t256.log").write_text("recent unrelated progress\n")

    diagnostics = _run_job_diagnostics(run_root)
    job = diagnostics["running_jobs"][0]

    assert job["arm"] == "t384"
    assert job["log"] == "pipeline_t384.log"
    assert job["state"] == "missing"
    assert job["age_min"] is None


def test_fit_and_export_logs_require_the_running_jobs_exact_name(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_job(
        run_root,
        {
            "name": "w19_4_vt_f1_mu3",
            "steps": [
                {
                    "label": "fit",
                    "argv": ["python", "scripts/train_rnn.py", "--name", "vt_f1_mu3"],
                },
                {
                    "label": "export",
                    "argv": ["python", "scripts/predict_run.py", "--name", "vt_f1_mu3"],
                },
            ],
        },
    )
    logs = run_root / "logs"
    logs.mkdir()
    exact_log = logs / "w19_4_vt_f1_mu3.fit.log"
    exact_log.write_text("old fit progress\n")
    (logs / "other_job.fit.log").write_text("recent unrelated progress\n")
    stale_time = time.time() - 25 * 60
    os.utime(exact_log, (stale_time, stale_time))

    diagnostics = _run_job_diagnostics(run_root)
    job = diagnostics["running_jobs"][0]

    assert job["run_name"] == "vt_f1_mu3"
    assert job["log"] == "w19_4_vt_f1_mu3.fit.log"
    assert job["state"] == "present"
    assert job["age_min"] >= 25


def test_check_vast_only_monitors_lane_b(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller = ModuleType("controller")
    observed: list[str] = []
    monkeypatch.setitem(sys.modules, "controller", controller)
    monkeypatch.setattr(watchdog, "VAST", tmp_path)
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    monkeypatch.setattr(
        watchdog,
        "check_vast_lane",
        lambda problems, state, module, lane: observed.append(lane),
    )

    watchdog.check_vast(watchdog.Problems({}), {})

    assert observed == ["b"]
    assert "VT_LANE" not in os.environ


def test_kaggle_check_uses_current_kernel_and_archives_retired_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retired = watchdog.RETIRED_KAGGLE_KERNELS[0]
    old_key = f"kaggle_kernel_bad:{retired}"
    state: dict[str, object] = {
        "kaggle_checked": time.time(),
        "kaggle_targets": list(watchdog.RETIRED_KAGGLE_KERNELS),
        "kaggle_status": {retired: "ERROR"},
        "kaggle_problems": {old_key: ["CRIT", "historical error"]},
    }
    calls: list[list[str]] = []

    def fake_sh(args: list[str], timeout: int = 60) -> str:
        calls.append(args)
        return "Kernel status: RUNNING"

    monkeypatch.setattr(watchdog, "sh", fake_sh)
    problems = watchdog.Problems({})

    watchdog.check_kaggle(problems, state)

    assert watchdog.KAGGLE_KERNELS == ("vladkochetov07/wnn-kaggle-long-20261010-a",)
    assert [call[-1] for call in calls] == list(watchdog.KAGGLE_KERNELS)
    assert state["kaggle_retired"][retired] == {
        "last_status": "ERROR",
        "last_problem": ["CRIT", "historical error"],
    }
    assert old_key not in state["kaggle_problems"]
    assert not problems.messages


def test_alert_reader_displays_current_and_legacy_rows_and_keeps_cursor(
    tmp_path: Path,
) -> None:
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "alerts_unread.sh"
    shutil.copy2(Path(watchdog.__file__).with_name("alerts_unread.sh"), script)
    alerts = tmp_path / "research/campaign_2026-10-06/bandit_continuation/watchdog/ALERTS.jsonl"
    alerts.parent.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/watchdog_alerts_mixed.jsonl"
    shutil.copy2(fixture, alerts)

    first = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10, check=True
    )
    assert "2026-10-11T12:00:00 ALERT WARN vast_b_job_stalled:w19_fit" in first.stdout
    assert "2026-10-10T18:30:00 RECOVERED INFO local_gpu_idle_with_queue" in first.stdout
    cursor = tmp_path / ".runtime/alerts_cursor"
    assert cursor.read_text().strip() == "2"

    second = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10, check=True
    )
    assert second.stdout == ""
    assert cursor.read_text().strip() == "2"

    with alerts.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "utc": "2026-10-11T12:05:00+00:00",
                    "event": "RECOVERED",
                    "severity": "INFO",
                    "key": "vast_b_job_stalled:w19_fit",
                    "message": "recovered: job log resumed",
                }
            )
            + "\n"
        )
    third = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10, check=True
    )

    assert "2026-10-11T12:05:00 RECOVERED INFO vast_b_job_stalled:w19_fit" in third.stdout
    assert cursor.read_text().strip() == "3"
