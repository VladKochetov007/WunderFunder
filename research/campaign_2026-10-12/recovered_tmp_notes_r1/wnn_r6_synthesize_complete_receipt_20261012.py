from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn-agent-worktrees/f1-core-hft-review-20261012")
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/r6_bootstrap_guard_successor_20261012_r1"
COLLECTION = PACKET / "terminal_collection_v2"
ARTIFACT_DIR = COLLECTION / "provider_outputs"
ATTEMPT_DIR = ARTIFACT_DIR / "attempt_001"
INVENTORY_PATH = COLLECTION / "PROVIDER_INVENTORY_R1.json"
COMPLETE_RECEIPT = COLLECTION / "COMPLETE_COLLECTOR_RECEIPT_R1.json"
ROOT_RELEASE = PACKET.parent / "ROOT_R6_VERSION2_DISPATCH_RELEASE_R1.json"
PUSH_RECEIPT = PACKET / "R6_KERNEL_VERSION2_PUSH_RECEIPT_R1.json"
POLL_PATH = Path("/tmp/wnn_r6_kernel_v2_status_poll_20261012.jsonl")
KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"


def file_sha256(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    """Load the approved generic collector without changing its source."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("approved collector module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Create the missing collector receipt from its already-finished attempt."""
    if COMPLETE_RECEIPT.exists() or not ATTEMPT_DIR.is_dir():
        print(json.dumps({"state": "refused_existing_receipt_or_missing_attempt"}))
        return 2
    inventory = json.loads(INVENTORY_PATH.read_text())
    poll = json.loads(POLL_PATH.read_text().splitlines()[-1])
    status = poll["status"][0]
    if (
        inventory.get("inventory_complete") is not True
        or inventory.get("kernel_id") != KERNEL
        or status.get("kernel") != KERNEL
        or status.get("status") != "KernelWorkerStatus.ERROR"
    ):
        print(json.dumps({"state": "refused_inventory_or_terminal_status"}))
        return 3
    result = json.loads((ATTEMPT_DIR / "attempt_result.json").read_text())
    cli_path = str(MAIN / ".venv/bin/kaggle")
    config = {
        "repo_root": str(ROOT),
        "config_dir": str(PACKET),
        "minimum_free_bytes": 4 * 1024**3,
        "maximum_download_bytes": 8 * 1024**3,
        "max_runtime_seconds": 1800,
        "max_attempts": 1,
        "kaggle_cli": cli_path,
        "expected_cli_version": "Kaggle CLI 2.2.4",
    }
    target = {
        "kernel": KERNEL,
        "resource": "gpu",
        "root_admission_sha256": file_sha256(ROOT_RELEASE),
        "push_receipt": "R6_KERNEL_VERSION2_PUSH_RECEIPT_R1.json",
    }
    collector = load_module(
        "reviewed_complete_terminal_collector",
        MAIN / "research/campaign_2026-10-06/cloud_preflight/collect_terminal_outputs.py",
    )
    cli_info = collector.kaggle_cli_version(cli_path)
    receipt = collector._terminal_receipt(
        target=target,
        status_row=status,
        attempt_dir=ATTEMPT_DIR,
        artifact_dir=ARTIFACT_DIR,
        result=result,
        cli_info=cli_info,
        config=config,
    )
    collector.atomic_write_json(COMPLETE_RECEIPT, receipt)
    os.chmod(COMPLETE_RECEIPT, 0o600)
    terminal_collection = receipt["terminal_collection"]
    file_map = terminal_collection["files"]
    prefix = "attempt_001/outputs/"
    local_outputs = {name.removeprefix(prefix) for name in file_map if name.startswith(prefix)}
    remote_outputs = {row["name"] for row in inventory["files"]}
    slug_log = KERNEL.split("/")[-1] + ".log"
    log_relative = prefix + slug_log
    log_metadata = file_map.get(log_relative)
    log_close = bool(
        log_metadata
        and inventory.get("response_log")
        and log_metadata.get("size_bytes") == inventory["response_log"].get("size_bytes")
        and log_metadata.get("sha256") == inventory["response_log"].get("sha256")
    )
    exact_files = local_outputs == remote_outputs | {slug_log}
    complete = collector.collection_is_complete(COMPLETE_RECEIPT, ARTIFACT_DIR)
    summary = {
        "state": "receipt_reconstructed_from_existing_attempt",
        "receipt_path": str(COMPLETE_RECEIPT),
        "receipt_sha256": file_sha256(COMPLETE_RECEIPT),
        "collector_complete": complete,
        "command_exit_codes": {
            "logs": terminal_collection["log_exit_code"],
            "outputs": terminal_collection["output_exit_code"],
        },
        "provider_remote_names": len(remote_outputs),
        "downloaded_output_names": len(local_outputs),
        "remote_output_name_set_exact_plus_response_log": exact_files,
        "response_log_bytes_sha256_match": log_close,
        "local_response_log": log_metadata,
        "attempt_bytes": collector.directory_bytes(ATTEMPT_DIR),
        "receipt_file_count": len(file_map),
        "cli_version": cli_info,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

