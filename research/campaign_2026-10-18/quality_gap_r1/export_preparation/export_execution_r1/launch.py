"""Launch one root-admitted F0 export through the durable B resource guard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bindings import (
    B_GUARD,
    B_GUARD_STATE,
    B_HOME,
    B_LIMITED,
    B_PROJECT_ROOT,
    B_PYTHON,
    B_TMP,
    EXECUTION_DIR,
    RESOURCE_ENVELOPE,
    read_json,
    require_outputs_absent,
    sha256_file,
    write_json_exclusive,
)
from prepare import build_contract


def _same_contract(actual: dict[str, Any], saved: dict[str, Any]) -> None:
    saved_without_path = dict(saved)
    saved_without_path.pop("contract_path", None)
    if actual != saved_without_path:
        raise ValueError("live staged inputs no longer match the prepared export contract")


def worker_command(
    contract_path: Path, contract_sha256: str, contract: dict[str, Any]
) -> list[str]:
    """Construct the exact child command guarded and later matched at capture."""
    project_root = Path(contract["project_root"])
    overlay = (
        project_root
        / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2"
    )
    frozen = project_root / "frozen_source_snapshot"
    env_values = {
        "WNN_PROJECT_ROOT": str(project_root),
        "WNN_PORTABLE_DATA_MANIFEST": str(
            project_root / "research/campaign_2026-10-04/portable_inputs.json"
        ),
        "WNN_F0_COMPATIBILITY_RECEIPT": str(
            project_root
            / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/COMPATIBILITY_RECEIPT_R2.json"
        ),
        "WNN_F0_MEMBERSHIP_RECEIPT": str(
            project_root
            / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
        ),
        "PYTHONPATH": f"{overlay / 'src'}:{frozen / 'src'}",
        "WNN_GPU_MEMORY_BYTES": str(RESOURCE_ENVELOPE["allocator_bytes"]),
        "WNN_GPU_MEMORY_RESERVE_BYTES": str(
            RESOURCE_ENVELOPE["gpu_free_memory_safety_reserve_bytes"]
        ),
        "WNN_LOADER_WORKERS": "0",
        "WNN_LOADER_RAM_GB": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env = ["/usr/bin/env", *(f"{key}={value}" for key, value in env_values.items())]
    worker = [
        str(B_PYTHON),
        "-B",
        str(EXECUTION_DIR / "worker.py"),
        "--contract",
        str(contract_path),
        "--contract-sha256",
        contract_sha256,
        "--predictor",
        str(project_root / "frozen_source_snapshot/scripts/predict_run.py"),
    ]
    for name in contract["runtime"]["export_utility_code"]:
        worker.extend(["--sealed-source", str(EXECUTION_DIR / name)])
    return [
        *env,
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{RESOURCE_ENVELOPE['runtime_limit_seconds']}s",
        *worker,
    ]


def guard_command(contract_path: Path, contract_sha256: str, contract: dict[str, Any]) -> list[str]:
    """Wrap the predictor worker in limited and bandit_resources GPU admission."""
    admission = contract["root_admission"]
    attempt = contract["attempt_id"]
    run_child = worker_command(contract_path, contract_sha256, contract)
    return [
        str(B_LIMITED),
        str(B_PYTHON),
        str(B_GUARD),
        "--root",
        str(B_PROJECT_ROOT),
        "--state",
        str(B_GUARD_STATE),
        "launch",
        "--attempt",
        attempt,
        "--kind",
        "gpu",
        "--cpus",
        "1",
        "--reserve-bytes",
        str(RESOURCE_ENVELOPE["reserve_bytes"]),
        "--profile",
        str(RESOURCE_ENVELOPE["profile"]),
        "--contract",
        str(contract_path),
        "--gpu-device",
        str(admission["gpu_uuid"]),
        "--python",
        str(B_PYTHON),
        "--admission-timeout",
        str(RESOURCE_ENVELOPE["admission_timeout_seconds"]),
        "--",
        *run_child,
    ]


def _parse_guard_output(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("bandit_resources did not return its JSON launch receipt") from error
    if not isinstance(value, dict):
        raise ValueError("bandit_resources launch receipt is not a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    saved = read_json(contract_path)
    attempt = str(saved.get("attempt_id", ""))
    admission_path = Path(saved["root_admission"]["path"])
    actual = build_contract(attempt, admission_path, require_fresh_admission=True)
    _same_contract(actual, saved)
    project_root = Path(saved["project_root"])
    require_outputs_absent(project_root)
    for path in (B_LIMITED, B_GUARD, B_PYTHON):
        if not path.is_file():
            raise FileNotFoundError(f"required executable is unavailable: {path}")
    if not os.access(B_LIMITED, os.X_OK) or not os.access(B_PYTHON, os.X_OK):
        raise PermissionError("frozen limited helper or B Python interpreter is not executable")
    runtime = saved["runtime"]
    if sha256_file(B_LIMITED) != runtime["limited_sha256"]:
        raise ValueError("scripts/limited changed after preparation")
    if sha256_file(B_GUARD) != runtime["bandit_resources_sha256"]:
        raise ValueError("bandit_resources.py changed after preparation")
    contract_sha256 = sha256_file(contract_path)
    launch_dir = EXECUTION_DIR / "records" / attempt
    launch_dir.mkdir(parents=True, exist_ok=True)
    started_path = launch_dir / "LAUNCH_STARTED.json"
    write_json_exclusive(
        started_path,
        {
            "schema": "quality_gap.strong_control_export_launch_start.r1",
            "status": "GUARD_LAUNCH_REQUESTED",
            "attempt_id": attempt,
            "contract": str(contract_path),
            "contract_sha256": contract_sha256,
            "created_utc": datetime.now(UTC).isoformat(),
            "gpu_uuid": saved["gpu_uuid"],
            "command": guard_command(contract_path, contract_sha256, saved),
        },
    )
    environment = os.environ.copy()
    environment.update(
        {
            "WNN_SLICE": "wnn-campaign.slice",
            "WNN_CPU_QUOTA": "10%",
            "WNN_THREADS": "1",
            "WNN_MEM_MAX": "512M",
            "WNN_RUNTIME_MAX": "120",
            "WNN_TMP_ROOT": str(B_TMP),
            "WNN_GPU_MEMORY_RESERVE_BYTES": str(
                RESOURCE_ENVELOPE["gpu_free_memory_safety_reserve_bytes"]
            ),
        }
    )
    try:
        result = subprocess.run(
            guard_command(contract_path, contract_sha256, saved),
            cwd=B_HOME,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        parsed = _parse_guard_output(result.stdout) if result.returncode == 0 else None
        payload = {
            "schema": "quality_gap.strong_control_export_guard_launch.r1",
            "status": "RESERVED"
            if result.returncode == 0
            else "LAUNCH_COMMAND_FAILED_RETAIN_EVIDENCE",
            "attempt_id": attempt,
            "contract_sha256": contract_sha256,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "guard_result": parsed,
            "command": guard_command(contract_path, contract_sha256, saved),
        }
    except (OSError, subprocess.SubprocessError) as error:
        payload = {
            "schema": "quality_gap.strong_control_export_guard_launch.r1",
            "status": "LAUNCH_UNCERTAIN_RECONCILE_GUARD_STATE",
            "attempt_id": attempt,
            "contract_sha256": contract_sha256,
            "error": f"{type(error).__name__}: {error}",
            "command": guard_command(contract_path, contract_sha256, saved),
        }
        write_json_exclusive(launch_dir / "GUARD_LAUNCH.json", payload)
        raise RuntimeError(
            "guard launch outcome is uncertain; retain evidence and reconcile"
        ) from error
    write_json_exclusive(launch_dir / "GUARD_LAUNCH.json", payload)
    if payload["status"] != "RESERVED":
        raise RuntimeError(
            "guard launch failed; retain launch evidence and inspect its attempt record"
        )
    if parsed.get("id") != attempt or parsed.get("status") != "reserved":
        raise RuntimeError("guard launch receipt does not match the requested reservation")
    if parsed.get("unit") != f"wnn-bandit-{attempt}.service":
        raise RuntimeError("guard launch unit name differs from the verified controller")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
