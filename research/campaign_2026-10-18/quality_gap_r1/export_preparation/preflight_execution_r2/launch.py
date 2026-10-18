"""Run one externally bounded R2 CPU loader preflight through the B SSH alias."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

from bindings import (
    B_BASE,
    B_PROJECT_ROOT,
    B_PYTHON,
    B_TMP,
    EXECUTION_REL,
    MEMBERSHIP_REL,
    OLD_SPLIT_REL,
    OVERLAY_REL,
    REGISTRY_REL,
    verify_local_prerequisites,
)
from local_common import (
    get_record_dir,
    parse_remote_json,
    read_json,
    run_ssh,
    utc_now,
    verify_remote_file_sha256,
    write_bytes_exclusive,
    write_json_exclusive,
)

CPU_LIMIT_SECONDS = 900
SSH_WAIT_SECONDS = 930


def _verify_remote_stage(alias: str, attempt_id: str) -> dict[str, Any]:
    """Require the exact staged root and an absent output receipt."""
    helper = B_PROJECT_ROOT / EXECUTION_REL / "remote_stage.py"
    record = get_record_dir(attempt_id)
    binding = read_json(record / "STAGE_BINDING_R2.json")
    verify_remote_file_sha256(
        alias,
        helper,
        binding["remote"]["stage_helper"]["sha256"],
    )
    result = run_ssh(
        alias,
        [
            str(B_PYTHON),
            "-B",
            str(helper),
            "--mode",
            "verify",
            "--attempt-id",
            attempt_id,
            "--root",
            str(B_PROJECT_ROOT),
        ],
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "B stage verification failed: "
            + result.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    receipt = parse_remote_json(result.stdout)
    if receipt.get("status") != "VERIFIED_READY_FOR_CPU_PREFLIGHT":
        raise ValueError("B stage is not ready for the CPU preflight")
    return receipt


def _preflight_command(attempt_id: str) -> list[str]:
    """Return the fixed, resource-capped B command for the R2 loader gate."""
    project = str(B_PROJECT_ROOT)
    overlay = B_PROJECT_ROOT / OVERLAY_REL
    frozen = B_PROJECT_ROOT / "frozen_source_snapshot"
    registry = B_PROJECT_ROOT / REGISTRY_REL
    membership = B_PROJECT_ROOT / MEMBERSHIP_REL
    split = B_PROJECT_ROOT / OLD_SPLIT_REL
    script = overlay / "scripts/preflight_compatibility.py"
    output = overlay / "COMPATIBILITY_RECEIPT_R2.json"
    unit_suffix = attempt_id.removeprefix("preflight-r2-")
    unit = f"computer-vision-homework-quality-gap-f0-preflight-r2-{unit_suffix}.service"
    pythonpath = f"{overlay / 'src'}:{frozen / 'src'}"
    return [
        "/usr/bin/flock",
        "-n",
        "--conflict-exit-code=75",
        str(B_BASE / "quality-gap-r1/export-preparation-r2/preflight.lock"),
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={unit}",
        "--slice=wnn-campaign.slice",
        "--property=CPUQuota=50%",
        "--property=MemoryHigh=1300000000",
        "--property=MemoryMax=1500000000",
        "--property=MemorySwapMax=0",
        f"--property=RuntimeMaxSec={CPU_LIMIT_SECONDS}",
        "--property=TimeoutStopSec=10",
        "--property=KillMode=control-group",
        f"--working-directory={project}",
        "--setenv=CUDA_VISIBLE_DEVICES=",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=WNN_STARTUP_DIAGNOSTICS=1",
        f"--setenv=TMPDIR={B_TMP}",
        f"--setenv=WNN_PROJECT_ROOT={project}",
        f"--setenv=FROZEN_SOURCE_SNAPSHOT={frozen}",
        f"--setenv=WNN_PORTABLE_DATA_MANIFEST={registry}",
        f"--setenv=WNN_F0_MEMBERSHIP_RECEIPT={membership}",
        f"--setenv=PYTHONPATH={pythonpath}",
        str(B_PYTHON),
        "-B",
        str(script),
        "--project-root",
        project,
        "--split-manifest",
        str(split),
        "--membership-receipt",
        str(membership),
        "--output",
        str(output),
    ]


def launch(attempt_id: str, alias: str) -> Path:
    """Start one capped CPU preflight and preserve its raw stdout and stderr."""
    record_dir = get_record_dir(attempt_id)
    stage = read_json(record_dir / "STAGE.json")
    if stage.get("status") != "STAGED" or stage.get("attempt_id") != attempt_id:
        raise ValueError("this attempt has no successful create-only stage")
    local_pins = verify_local_prerequisites()
    remote_stage = _verify_remote_stage(alias, attempt_id)
    command = _preflight_command(attempt_id)
    start_record = {
        "schema": "quality_gap.preflight_launch_start.r2",
        "status": "LAUNCH_PREPARED",
        "attempt_id": attempt_id,
        "ssh_alias": alias,
        "created_utc": utc_now(),
        "remote_stage_status": remote_stage["status"],
        "reviewed_overlay_manifest_sha256": local_pins["overlay_manifest"]["sha256"],
        "reviewed_manifest_digest_matches": True,
        "remote_command_argv": command,
        "cpu_quota_percent": 50,
        "memory_high_bytes": 1300000000,
        "memory_max_bytes": 1500000000,
        "memory_swap_max_bytes": 0,
        "runtime_max_seconds": CPU_LIMIT_SECONDS,
        "local_ssh_wait_seconds": SSH_WAIT_SECONDS,
        "full_source_bytes_approx": 37100000000,
        "measured_full_hash_rate_bytes_per_second": None,
        "minimum_rate_to_finish_within_cap_bytes_per_second": 37100000000 / CPU_LIMIT_SECONDS,
        "runtime_estimate": "No full-source scan timing is available; 900 seconds is a hard stop, not a forecast.",
        "model_or_target_array_load": False,
        "gpu_or_training": False,
    }
    write_json_exclusive(record_dir / "LAUNCH_STARTED.json", start_record)
    started = time.monotonic()
    finished_utc: str | None = None
    returncode: int | None = None
    timed_out = False
    stdout = b""
    stderr = b""
    error: str | None = None
    try:
        result = run_ssh(alias, command, timeout=SSH_WAIT_SECONDS)
        returncode = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        error = "local SSH wait timed out; the B systemd unit remains bounded by RuntimeMaxSec"
    except Exception as exc:
        error = str(exc)
    finished_utc = utc_now()
    write_bytes_exclusive(record_dir / "PREFLIGHT_STDOUT.raw", stdout)
    write_bytes_exclusive(record_dir / "PREFLIGHT_STDERR.raw", stderr)
    summary: dict[str, Any] = {}
    with contextlib.suppress(ValueError):
        summary = parse_remote_json(stdout)
    execution = {
        "schema": "quality_gap.preflight_execution.r2",
        "status": "SSH_WAIT_TIMEOUT" if timed_out else "RETURNED",
        "attempt_id": attempt_id,
        "ssh_alias": alias,
        "started_utc": start_record["created_utc"],
        "finished_utc": finished_utc,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "preflight_summary": summary,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "error": error,
        "root_admission": "NOT_ADMITTED",
    }
    write_json_exclusive(record_dir / "EXECUTION.json", execution)
    if timed_out or error or returncode != 0:
        raise RuntimeError(error or f"preflight command exited with status {returncode}")
    if summary.get("status") != "PASS_CPU_LOADER_PREFLIGHT":
        raise ValueError("preflight process returned zero without the expected PASS status")
    return record_dir


def main() -> int:
    """Invoke one staged, externally bounded B-side CPU preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-alias", default="beleriand")
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    result = launch(args.attempt_id, args.ssh_alias)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
