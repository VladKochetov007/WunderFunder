from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/campaign_job.py"
spec = importlib.util.spec_from_file_location("independent_campaign_launcher", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
RESULTS: list[dict] = []


def check(name: str, action: object) -> None:
    """Preserve independent launcher checks without launching any GPU process."""
    started = time.perf_counter()
    try:
        detail = action()
        result = {"name": name, "status": "pass", "detail": detail}
    except Exception as error:
        result = {"name": name, "status": "fail", "error": repr(error)}
    result["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.append(result)
    print(json.dumps(result), flush=True)


def timeout_validation() -> dict:
    accepted = []
    for value in ("nan", "inf", "-inf"):
        argv = [
            str(SCRIPT),
            "launch",
            "--name",
            "isolated-invalid-timeout",
            "--kind",
            "gpu",
            "--timeout",
            value,
            "--",
            "/bin/true",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(launcher, "launch", side_effect=accepted.append),
        ):
            with redirect_stderr(io.StringIO()):
                try:
                    launcher.main()
                except SystemExit as error:
                    assert error.code == 2
    assert not accepted, [str(args.timeout) for args in accepted]
    return {"nonfinite_timeout_rejected": True}


def telemetry_timeout() -> dict:
    completed = subprocess.CompletedProcess([], 0, stdout="1000\n", stderr="")
    with patch.object(launcher.subprocess, "run", return_value=completed) as run:
        assert launcher.device_memory() == 1000 * 1024**2
    kwargs = run.call_args.kwargs
    assert kwargs.get("timeout", 0) > 0, kwargs
    return {"timeout_seconds": kwargs["timeout"]}


def durable_runtime_backstop() -> dict:
    args = argparse.Namespace(
        name="independent-launch-inspection",
        timeout=120.0,
        kind="gpu",
        phase="controls",
        command=["/bin/true"],
    )
    with patch.object(launcher.subprocess, "run") as run:
        launcher.launch(args)
    command = run.call_args.args[0]
    assert "RuntimeMaxSec=128" in command, command
    assert "TimeoutStopSec=2s" in command, command
    assert "--setenv=WNN_RUNTIME_MAX=125" in command, command
    assert "--setenv=WNN_SCOPE_UNIT=wnn-top1-independent-launch-inspection-scope" in command, (
        command
    )
    return {"systemd_command": command}


def accounting() -> dict:
    finished = {"status": "finished", "elapsed_seconds": 31.2, "kind": "gpu"}
    failed = {"status": "failed", "elapsed_seconds": 20.1, "kind": "gpu"}
    cpu = {"status": "finished", "elapsed_seconds": 50.0, "kind": "cpu"}
    with patch.object(launcher, "records", return_value=[finished, failed, cpu]):
        result = launcher.budget()
    assert abs(result["gpu_seconds_used"] - 51.3) < 1e-12
    stale = {
        "status": "running",
        "elapsed_seconds": 1.0,
        "timeout_seconds": 120.0,
        "started_unix": 1.0,
        "unit": "nonexistent-unit",
        "kind": "gpu",
    }
    with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)):
        assert launcher.charged_seconds(stale) >= 128.0
    return {
        "finished_and_failed_counted": True,
        "cpu_excluded": True,
        "stale_gpu_record_charged_full_allowance": True,
    }


def stale_active_scope_refused() -> dict:
    """A dead monitor must not permit a second job while its scope runs."""
    args = argparse.Namespace(
        name="independent-refusal",
        timeout=120.0,
        kind="gpu",
        phase="controls",
        unit="unused",
        command=["/bin/true"],
    )
    prior = [{"kind": "gpu", "scope_unit": "previous.scope", "status": "failed"}]
    with (
        patch.object(launcher.fcntl, "flock"),
        patch.object(launcher, "records", return_value=prior),
        patch.object(launcher, "unit_active", return_value=True),
        patch.object(launcher, "write_record") as write,
    ):
        try:
            launcher.run_job(args)
        except RuntimeError as error:
            assert "previous GPU scope remains active" in str(error)
        else:
            raise AssertionError("a live previous GPU scope was accepted")
    write.assert_not_called()
    return {"prior_scope_checked_before_new_job_evidence": True}


def scope_stop_confirmation() -> dict:
    """Stop must fail closed if the bounded operation leaves a live scope."""
    with (
        patch.object(launcher, "unit_active", side_effect=[True, False]),
        patch.object(launcher.subprocess, "run") as run,
    ):
        launcher.stop_scope("independent.scope")
        assert run.call_args.kwargs["timeout"] <= 10
    with (
        patch.object(launcher, "unit_active", return_value=True),
        patch.object(launcher.subprocess, "run"),
    ):
        try:
            launcher.stop_scope("independent.scope")
        except RuntimeError as error:
            assert "remains active" in str(error)
        else:
            raise AssertionError("live scope accepted after stop")
    return {"scope_stop_confirmed_and_live_refused": True}


def common_slice() -> dict:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "wnn-campaign.slice",
            "-p",
            "CPUQuotaPerSecUSec",
            "-p",
            "MemoryMax",
            "-p",
            "MemorySwapMax",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    properties = dict(line.split("=", 1) for line in completed.stdout.splitlines())
    assert properties["CPUQuotaPerSecUSec"] == "12s"
    assert int(properties["MemoryMax"]) == 36 * 1024**3
    assert properties["MemorySwapMax"] == "0"
    return properties


def main() -> None:
    started = time.perf_counter()
    original_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    for name, action in (
        ("cumulative_failed_job_accounting", accounting),
        ("common_slice_hard_caps", common_slice),
        ("nonfinite_timeout_validation", timeout_validation),
        ("bounded_telemetry", telemetry_timeout),
        ("stale_active_scope_refused", stale_active_scope_refused),
        ("scope_stop_confirmation", scope_stop_confirmation),
        ("durable_systemd_runtime_backstop", durable_runtime_backstop),
    ):
        check(name, action)
    report = {
        "status": "pass" if all(row["status"] == "pass" for row in RESULTS) else "fail",
        "checks": RESULTS,
        "elapsed_seconds": time.perf_counter() - started,
        "script_sha256": original_hash,
        "script_sha256_after": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "scope": "Mocked service launch and telemetry; no GPU process or live job evidence modified",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
