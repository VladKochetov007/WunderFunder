from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_command(
    contract: dict[str, object], helper: Path
) -> tuple[str, list[str], dict[str, str]]:
    argv = contract["argv"]
    python_index = argv.index(
        "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python"
    )
    env_index = argv.index("/usr/bin/env")
    environment = os.environ.copy()
    for item in argv[env_index + 1 : python_index]:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"contract environment entry is malformed: {item}")
        environment[key] = value
    environment.update(
        {
            "WNN_SLICE": "wnn-campaign.slice",
            "WNN_CPU_QUOTA": "50%",
            "WNN_THREADS": "1",
            "WNN_MEM_MAX": "700000000",
            "WNN_RUNTIME_MAX": "120s",
        }
    )
    command = list(argv[python_index + 1 :])
    if not command:
        raise ValueError("contract has no helper command")
    command[0] = str(helper)
    return argv[python_index], command, environment


def _run(
    python: str, command: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python, *command],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve(strict=True)
    packet = args.packet.resolve(strict=True)
    manifest_path = packet / "EXPORT_ROOT_BINDING_MANIFEST_R3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    helper = packet / "run_predict_root_bound_r3.py"
    verifier = Path(__file__).resolve(strict=True)
    if _sha256(verifier) != manifest["validation"]["script_sha256"]:
        raise ValueError("preflight verifier differs from the frozen manifest")
    if _sha256(helper) != manifest["sources"]["helper_sha256"]:
        raise ValueError("runtime helper differs from the frozen manifest")

    preflight_results: dict[str, dict[str, object]] = {}
    commands: dict[str, list[str]] = {}
    python_path = ""
    environment: dict[str, str] = {}
    for item in manifest["exports"]:
        stage = item["stage"]
        contract_path = packet / item["contract_path"]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if _sha256(contract_path) != item["contract_sha256"]:
            raise ValueError(f"{stage} contract differs from the frozen manifest")
        project_index = contract["argv"].index("--project-root")
        if Path(contract["argv"][project_index + 1]).resolve(strict=True) != project_root:
            raise ValueError(f"{stage} contract points at a different project root")
        python_path, command, environment = _contract_command(contract, helper)
        if command[-2:] != ["--expected-checkpoint-sha256", "PENDING_ROOT_TERMINAL_SHA256"]:
            raise ValueError(f"{stage} contract checkpoint SHA is no longer pending")
        command.append("--preflight-only")
        result = _run(python_path, command, environment)
        if result.returncode != 0:
            raise RuntimeError(f"{stage} preflight failed: {result.stderr[-1000:]}")
        output = json.loads(result.stdout)
        if output.get("status") != "preflight_pass_no_export":
            raise ValueError(f"{stage} preflight did not return the expected status")
        preflight_results[stage] = output
        commands[stage] = command

    parent_contract = json.loads(
        (packet / "contracts" / "oldz4-r4-f1-s3-parent-export-rootbind-r3.json").read_text(
            encoding="utf-8"
        )
    )
    python_path, fail_closed_command, environment = _contract_command(parent_contract, helper)
    fail_closed = _run(python_path, fail_closed_command, environment)
    expected_failure = "checkpoint expected SHA-256 must be a lowercase 64-character SHA-256"
    if fail_closed.returncode == 0 or expected_failure not in fail_closed.stderr:
        raise RuntimeError("pending checkpoint SHA did not fail closed before export")

    receipt = {
        "schema": "wnn.oldz4-f1-root-bound-export-preflight.r3",
        "packet_id": manifest["packet_id"],
        "prepared_utc": datetime.now(UTC).isoformat(),
        "status": "actual_source_input_and_root_binding_preflights_pass; exports_not_run",
        "execution": {
            "fit": False,
            "prediction_export": False,
            "predictor_entrypoint_invoked": False,
            "full_valid_content_read": False,
            "checkpoint_sha_checked": False,
            "provider_action": False,
        },
        "resource_boundary": {
            "slice": "wnn-campaign.slice",
            "cpu_quota": "50%",
            "threads": 1,
            "memory_max_bytes": 700000000,
            "runtime_max": "120s per command",
        },
        "runtime_helper": {
            "path": str(helper),
            "sha256": _sha256(helper),
        },
        "preflights": preflight_results,
        "commands": commands,
        "fail_closed_pending_checkpoint_sha": {
            "exit_code": fail_closed.returncode,
            "expected_error": expected_failure,
            "stderr_last_line": fail_closed.stderr.strip().splitlines()[-1],
            "predictor_invoked": False,
        },
        "terminal_checkpoint_hashes": {
            "parent_cp3_sha256": None,
            "student_cp1_sha256": None,
            "binding_owner": "root after exact fit terminal and checkpoint/provenance/state validation",
        },
        "superseded_preflight_receipt": {
            "path": "ROOT_BINDING_PREFLIGHT_R3_ATTEMPT1_SUPERSEDED.json",
            "sha256": _sha256(packet / "ROOT_BINDING_PREFLIGHT_R3_ATTEMPT1_SUPERSEDED.json"),
            "reason": "Attempt 1 echoed the pending checkpoint-SHA sentinel in a field labeled checkpoint_sha256. The final helper reports null in preflight mode because that checkpoint was not checked; no fit or export ran in either attempt.",
        },
        "source_and_input_bindings": {
            "trainer_sha256": manifest["sources"]["trainer_sha256"],
            "predictor_sha256": manifest["sources"]["predictor_sha256"],
            "source_bindings_sha256": manifest["sources"]["source_bindings_sha256"],
            "input_bindings_sha256": manifest["input"]["input_bindings_sha256"],
            "split_manifest_sha256": manifest["input"]["split_manifest_sha256"],
        },
    }
    receipt_path = args.receipt.resolve(strict=False)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps({"receipt": str(receipt_path), "status": receipt["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
