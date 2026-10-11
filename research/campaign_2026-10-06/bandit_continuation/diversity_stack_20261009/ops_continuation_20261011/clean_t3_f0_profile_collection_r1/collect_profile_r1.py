from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# This collection folder is six directories below the repository root.
ROOT = HERE.parents[5]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CLOUD = MAIN / "research/campaign_2026-10-06/cloud_preflight"
sys.path.insert(0, str(CLOUD))

import collect_terminal_outputs as terminal  # noqa: E402
import poll_provider  # noqa: E402

KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner"
DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1"
ARCHIVE_SHA256 = "0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f"
PINS = {
    "ROOT_PROFILE_DISPATCH_ADMISSION_R1.json": "1368c5283b532dbee61567edcf50b595db3dca0f2d9a238d6cc2afdb10a33948",
    "ROOT_PROFILE_PUSH_R1.json": "19eb640c6e8425398a7244c0244154ca23c338dc6151c30f47a1d8c4ce9d82f1",
    "ROOT_PRIVATE_DATASET_READY_READBACK_R1.json": "8c2e4c846584bdb228da40a9c843e9aade4a6b8b354b30a3f3c5fea7484f9f81",
    "ROOT_PROFILE_PROVIDER_POLL_R1.json": "851e97e446c6afa6ab4ef8ac6305c76bd81a75d40afd9f884560fb30ad5c6b5d",
}


def sha256(path: Path) -> str:
    """Hash a file in bounded blocks for the receipt bindings."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_root_bindings() -> dict[str, dict[str, Any]]:
    """Fail closed unless copied root admission, push and READY receipts match."""
    bindings: dict[str, dict[str, Any]] = {}
    for name, expected in PINS.items():
        path = HERE / "root_bindings" / name
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"root receipt hash mismatch: {name}")
        bindings[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": observed}
    admission = json.loads(
        (HERE / "root_bindings" / "ROOT_PROFILE_DISPATCH_ADMISSION_R1.json").read_text()
    )
    push = json.loads((HERE / "root_bindings" / "ROOT_PROFILE_PUSH_R1.json").read_text())
    ready = json.loads(
        (HERE / "root_bindings" / "ROOT_PRIVATE_DATASET_READY_READBACK_R1.json").read_text()
    )
    if (
        admission.get("scope") != "two T4 profile jobs only, no fullfit"
        or admission.get("archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("root admission scope/archive binding mismatch")
    if (
        push.get("kernel_id") != KERNEL
        or push.get("returncode") != 0
        or push.get("profile_only") is not True
    ):
        raise ValueError("root push receipt does not bind the admitted profile kernel")
    if (
        push.get("dataset_version") != 1
        or push.get("readback_sha256")
        != bindings["ROOT_PRIVATE_DATASET_READY_READBACK_R1.json"]["sha256"]
    ):
        raise ValueError("push receipt does not bind exact READY readback")
    if ready.get("dataset") != DATASET or ready.get("kernel_push") is not False:
        raise ValueError("READY receipt dataset or scope mismatch")
    expected_remote = {
        "bundle.wnn": (217267, ARCHIVE_SHA256),
        "bundle-manifest.json": (
            24149,
            "8529006db1c07b9935aef4c1a4dc63e18a08b0dc1aeef129f3e8bfe4a8c6d960",
        ),
    }
    observed_remote = {
        row["name"]: (row["size_bytes"], row["sha256"])
        for row in ready.get("verified_remote_files", [])
    }
    if observed_remote != expected_remote:
        raise ValueError("remote READY file inventory differs from admitted archive")
    return bindings


def main() -> int:
    """Poll the admitted kernel and collect all terminal output and log evidence."""
    bindings = verify_root_bindings()
    provider_targets = CLOUD / "provider_targets.json"
    snapshot = poll_provider.collect(provider_targets)
    poll_provider.append_jsonl(HERE / "provider_quota_status.jsonl", snapshot)
    status_row = terminal._target_status(snapshot, KERNEL)
    if status_row is None:
        raise ValueError("exact profile kernel is absent from provider snapshot")
    if not terminal.terminal_state(str(status_row.get("status", ""))):
        print(
            json.dumps(
                {
                    "state": "provider_not_terminal",
                    "status": status_row,
                    "quota": snapshot["quota"],
                },
                sort_keys=True,
            )
        )
        return 0

    target = {
        "kernel": KERNEL,
        "resource": "gpu",
        "root_admission_sha256": bindings["ROOT_PROFILE_DISPATCH_ADMISSION_R1.json"]["sha256"],
        "root_admission": "root_bindings/ROOT_PROFILE_DISPATCH_ADMISSION_R1.json",
        "push_receipt": "root_bindings/ROOT_PROFILE_PUSH_R1.json",
        "asset_ready_receipt": "root_bindings/ROOT_PRIVATE_DATASET_READY_READBACK_R1.json",
        "artifact_dir": "attempts",
        "terminal_receipt": "TERMINAL_COLLECTION_RECEIPT_R1.json",
    }
    config = {
        "repo_root": str(ROOT),
        "config_dir": str(HERE),
        "minimum_free_bytes": 4 * 1024**3,
        "maximum_download_bytes": 8 * 1024**3,
        "max_runtime_seconds": 420,
        "max_attempts": 3,
        "kaggle_cli": str(MAIN / ".venv/bin/kaggle"),
        "expected_cli_version": "Kaggle CLI 2.2.4",
    }
    result = terminal.collect_terminal_target(
        target=target, status_row=status_row, config=config, repo_root=ROOT, config_dir=HERE
    )
    attempt = HERE / "attempts" / str(result.get("attempt", "attempt_001"))
    if result.get("state") in {"terminal_outputs_and_logs_collected", "already_collected"}:
        deadline = time.monotonic() + 45
        inventory_result = terminal.run_bounded_command(
            [
                config["kaggle_cli"],
                "kernels",
                "files",
                "--format",
                "json",
                "--page-size",
                "200",
                KERNEL + "/1",
            ],
            stdout_path=attempt / "provider_files_inventory.stdout.json",
            stderr_path=attempt / "provider_files_inventory.stderr.txt",
            monitor_dir=attempt,
            min_free_bytes=config["minimum_free_bytes"],
            max_bytes=config["maximum_download_bytes"],
            deadline=deadline,
        )
    else:
        inventory_result = {"exit_code": None, "stop_reason": "output_or_log_collection_incomplete"}
    inventory_path = attempt / "provider_files_inventory.stdout.json"
    provider_inventory: Any = None
    parse_error = None
    if inventory_result.get("exit_code") == 0:
        try:
            provider_inventory = json.loads(inventory_path.read_text())
        except json.JSONDecodeError as error:
            parse_error = str(error)
    terminal_receipt = HERE / "TERMINAL_COLLECTION_RECEIPT_R1.json"
    handoff = {
        "schema": "wnn.clean_t3_f0_profile_terminal_handoff.r1",
        "kernel": KERNEL,
        "provider_status": status_row,
        "quota_snapshot": snapshot["quota"],
        "root_bindings": bindings,
        "archive_sha256": ARCHIVE_SHA256,
        "collector_source": {
            "collect_terminal_outputs.py": sha256(CLOUD / "collect_terminal_outputs.py"),
            "poll_provider.py": sha256(CLOUD / "poll_provider.py"),
        },
        "terminal_collector_receipt": {
            "path": terminal_receipt.relative_to(ROOT).as_posix(),
            "sha256": sha256(terminal_receipt) if terminal_receipt.is_file() else None,
            "collection_complete": terminal.collection_is_complete(
                terminal_receipt, HERE / "attempts"
            ),
        },
        "provider_files_inventory": {
            "argv": [
                config["kaggle_cli"],
                "kernels",
                "files",
                "--format",
                "json",
                "--page-size",
                "200",
                KERNEL + "/1",
            ],
            "result": inventory_result,
            "stdout_sha256": sha256(inventory_path) if inventory_path.is_file() else None,
            "stderr_sha256": sha256(attempt / "provider_files_inventory.stderr.txt")
            if (attempt / "provider_files_inventory.stderr.txt").is_file()
            else None,
            "parsed": provider_inventory,
            "parse_error": parse_error,
        },
        "attempt_files": terminal.file_inventory(attempt) if attempt.is_dir() else {},
    }
    terminal.atomic_write_json(HERE / "PROFILE_TERMINAL_HANDOFF_R1.json", handoff)
    print(
        json.dumps(
            {
                "state": result.get("state"),
                "inventory_exit_code": inventory_result.get("exit_code"),
                "receipt": str(HERE / "PROFILE_TERMINAL_HANDOFF_R1.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if handoff["terminal_collector_receipt"]["collection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
