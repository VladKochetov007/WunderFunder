from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn-agent-worktrees/f1-core-hft-review-20261012")
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/r6_bootstrap_guard_successor_20261012_r1"
KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"
INVENTORY_PATH = PACKET / "terminal_collection_v2/PROVIDER_INVENTORY_R1.json"
ROOT_RELEASE = PACKET.parent / "ROOT_R6_VERSION2_DISPATCH_RELEASE_R1.json"
PUSH_RECEIPT = PACKET / "R6_KERNEL_VERSION2_PUSH_RECEIPT_R1.json"
POLL_PATH = Path("/tmp/wnn_r6_kernel_v2_status_poll_20261012.jsonl")
ARTIFACT_DIR = "terminal_collection_v2/provider_outputs"
TERMINAL_RECEIPT = "terminal_collection_v2/COMPLETE_COLLECTOR_RECEIPT_R1.json"


def file_sha256(path: Path) -> str:
    """Hash a file without loading a large artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    """Load the approved generic collector without modifying its source."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("approved collector module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Run the authorized one-shot terminal download through reviewed primitives."""
    release = json.loads(ROOT_RELEASE.read_text())
    push = json.loads(PUSH_RECEIPT.read_text())
    poll = json.loads(POLL_PATH.read_text().splitlines()[-1])
    inventory = json.loads(INVENTORY_PATH.read_text())
    bounds = release["terminal_collection"]
    if (
        release.get("authorized") is not True
        or release.get("status") != "admitted_one_version2_push"
        or release.get("kernel_id") != KERNEL
        or release.get("intended_kernel_version") != 2
        or push.get("kernel_id") != KERNEL
        or push.get("provider_version") != 2
        or push.get("status") != "accepted_provider_queued"
        or poll["status"][0].get("status") != "KernelWorkerStatus.ERROR"
        or inventory.get("kernel_id") != KERNEL
        or inventory.get("accepted_kernel_version") != 2
        or inventory.get("inventory_complete") is not True
        or inventory.get("signed_urls_persisted") is not False
    ):
        print(json.dumps({"state": "refused_identity_or_inventory_gate_mismatch"}))
        return 2
    if (
        bounds.get("max_attempts") != 1
        or bounds.get("maximum_download_bytes") != 8 * 1024**3
        or bounds.get("minimum_free_bytes") != 4 * 1024**3
        or bounds.get("max_runtime_seconds") != 1800
    ):
        print(json.dumps({"state": "refused_root_bounds_mismatch"}))
        return 3
    artifact_path = PACKET / ARTIFACT_DIR
    receipt_path = PACKET / TERMINAL_RECEIPT
    if artifact_path.exists() or receipt_path.exists():
        print(json.dumps({"state": "refused_existing_collector_attempt"}))
        return 4
    disk = shutil.disk_usage(PACKET)
    if disk.free < int(bounds["pre_download_free_disk_required_bytes"]):
        print(json.dumps({"state": "refused_pre_download_free_space"}))
        return 5

    collector = load_module(
        "reviewed_complete_terminal_collector",
        MAIN / "research/campaign_2026-10-06/cloud_preflight/collect_terminal_outputs.py",
    )
    target = {
        "kernel": KERNEL,
        "resource": "gpu",
        "root_admission_sha256": file_sha256(ROOT_RELEASE),
        "push_receipt": "r6_bootstrap_guard_successor_20261012_r1/R6_KERNEL_VERSION2_PUSH_RECEIPT_R1.json",
        "artifact_dir": ARTIFACT_DIR,
        "terminal_receipt": TERMINAL_RECEIPT,
    }
    status_row = {
        "kernel": KERNEL,
        "status": poll["status"][0]["status"],
        "failure_message": poll["status"][0].get("failure_message", ""),
    }
    config = {
        "minimum_free_bytes": int(bounds["minimum_free_bytes"]),
        "maximum_download_bytes": int(bounds["maximum_download_bytes"]),
        "max_runtime_seconds": int(bounds["max_runtime_seconds"]),
        "max_attempts": int(bounds["max_attempts"]),
        "kaggle_cli": str(MAIN / ".venv/bin/kaggle"),
        "expected_cli_version": "Kaggle CLI 2.2.4",
    }
    result = collector.collect_terminal_target(
        target=target,
        status_row=status_row,
        config=config,
        repo_root=ROOT,
        config_dir=PACKET,
    )
    receipt = json.loads(receipt_path.read_text())
    attempt_path = artifact_path / receipt["terminal_collection"]["attempt_directory"]
    attempt_bytes = collector.directory_bytes(attempt_path)
    summary = {
        "state": result["state"],
        "attempt": result.get("attempt"),
        "provider_output_file_count": result.get("provider_output_file_count"),
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "receipt_file_count": len(receipt["terminal_collection"]["files"]),
        "collector_exit_codes": {
            "logs": receipt["terminal_collection"]["log_exit_code"],
            "outputs": receipt["terminal_collection"]["output_exit_code"],
        },
        "attempt_bytes": attempt_bytes,
        "free_disk_after_bytes": shutil.disk_usage(PACKET).free,
        "elapsed_seconds": receipt["attempt_result"].get("logs", {}).get("elapsed_seconds", 0)
        + receipt["attempt_result"].get("output", {}).get("elapsed_seconds", 0),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

