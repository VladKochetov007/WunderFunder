from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-student-20261012-r24"
SOURCE_DATASET = "vladkochetov07/wnn-clean-t3-f0-treatment-source-20261012-r22"
ASSET_DATASET = "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22"
KERNEL_SHA256 = "31206a824b03ec9e8f182f74bef79fad378456c51ddfa7383ea8485b3b3f269a"
ROOT_RELEASE_SHA256 = "4c397f2f08cada184ce029745041d5458bf90a7b39184fb8d6a54dcb48559745"
PUSH_RECEIPT_SHA256 = "06ed8c461e64ecdfeab76f64ba8d142ac05758d631e89c3f4b92bf860d263916"
SOURCE_VERSION = 2
ASSET_VERSION = 1
WALL_SECONDS = 10_000
PROVIDER_TIMEOUT_SECONDS = 10_800
LOGICAL_HOLD_GPU_SECONDS = 21_600
COHORT_GPU_SECONDS = 20_000
MAX_DOWNLOAD_BYTES = 8_589_934_592
MIN_FREE_BYTES = 4_294_967_296
MAX_RUNTIME_SECONDS = 800
MONITOR_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/monitor_20261012_1207_r24"
)
PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261012/clean_t3_f0_student_treatment_kaggle_r22"
)
ROOT_RELEASE_REL = PACKET_REL / "ROOT_TREATMENT_RELEASE_R7_R1.json"
PUSH_RECEIPT_REL = MONITOR_REL / "KERNEL_PUSH_R7_R1.json"
POLL_TARGET_NAME = "R7_PROVIDER_STATUS_TARGET_R1.json"
POLL_LOG_NAME = "R7_PROVIDER_POLL_LOG_R1.jsonl"
ARTIFACT_DIR = "provider_output_r1"
TERMINAL_RECEIPT = "R7_TERMINAL_OPERATOR_RECEIPT_R1.json"
KAGGLE_CLI = "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/kaggle"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def validate_root_and_push_receipts(repo_root: Path) -> dict[str, Any]:
    root_path = repo_root / ROOT_RELEASE_REL
    push_path = repo_root / PUSH_RECEIPT_REL
    if sha256_file(root_path) != ROOT_RELEASE_SHA256:
        raise ValueError("R7 root release bytes do not match the admitted receipt")
    if sha256_file(push_path) != PUSH_RECEIPT_SHA256:
        raise ValueError("R7 provider push receipt bytes do not match the accepted push")
    root_release = read_json(root_path)
    expected_target = {
        "kernel_id": KERNEL_ID,
        "source_dataset": SOURCE_DATASET,
        "asset_dataset": ASSET_DATASET,
    }
    if (
        root_release.get("schema") != "wnn.clean_f0_student_treatment_root_release.r2"
        or root_release.get("status") != "admitted"
        or root_release.get("authorized") is not True
        or root_release.get("target") != expected_target
        or root_release.get("kernel_sha256") != KERNEL_SHA256
        or root_release.get("fit_authorized") is not True
    ):
        raise ValueError("R7 root release does not authorize this exact kernel and treatment")
    identity = root_release.get("identity_successor", {})
    budget = root_release.get("quota_budget", {})
    if (
        identity.get("source_dataset_version") != SOURCE_VERSION
        or budget.get("controller_hold_gpu_seconds") != LOGICAL_HOLD_GPU_SECONDS
        or budget.get("provider_timeout_seconds") != PROVIDER_TIMEOUT_SECONDS
        or budget.get("wall_ceiling_seconds") != WALL_SECONDS
        or budget.get("cohort_gpu_seconds_upper_bound") != COHORT_GPU_SECONDS
        or budget.get("paid_spending_authorized") is not False
    ):
        raise ValueError("R7 root release quota/source-version binding is inconsistent")

    push = read_json(push_path)
    response = push.get("response", {})
    if (
        push.get("kernel") != KERNEL_ID
        or push.get("kernel_sha256") != KERNEL_SHA256
        or push.get("source_version") != SOURCE_VERSION
        or push.get("asset_version") != ASSET_VERSION
        or push.get("source_root_release_sha256") != ROOT_RELEASE_SHA256
        or push.get("source_all_four_file_sha_match") is not True
        or push.get("supersedes_failed_R6_same_cohort") is not True
        or response.get("error") != ""
        or response.get("versionNumber") != 1
        or response.get("invalidDatasetSources") != []
        or response.get("invalidKernelSources") != []
    ):
        raise ValueError("R7 push receipt does not prove the admitted version-1 dispatch")
    return {
        "root_release_sha256": ROOT_RELEASE_SHA256,
        "push_receipt_sha256": PUSH_RECEIPT_SHA256,
        "kernel_id": KERNEL_ID,
        "kernel_version": 1,
        "source_dataset_version": SOURCE_VERSION,
        "asset_dataset_version": ASSET_VERSION,
        "fit_authorized": True,
        "provider_write_performed_by_collector": False,
    }


def load_collector_modules(repo_root: Path) -> tuple[Any, Any]:
    cloud_preflight = repo_root / "research/campaign_2026-10-06/cloud_preflight"
    sys.path.insert(0, str(cloud_preflight))
    import collect_terminal_outputs
    import poll_provider

    return poll_provider, collect_terminal_outputs


def poll_and_collect(repo_root: Path) -> dict[str, Any]:
    monitor = repo_root / MONITOR_REL
    config_path = monitor / POLL_TARGET_NAME
    poll_provider, collector = load_collector_modules(repo_root)
    snapshot = poll_provider.collect(config_path)
    poll_provider.append_jsonl(monitor / POLL_LOG_NAME, snapshot)
    rows = snapshot.get("status", [])
    row = next((item for item in rows if item.get("kernel") == KERNEL_ID), None)
    if row is None:
        raise RuntimeError("read-only provider poll returned no row for the exact R7 kernel")
    state = str(row.get("status", ""))
    if row.get("monitoring_state") == "stopped_after_terminal_evidence_downloaded":
        return {"state": "already_collected", "provider_status": state, "snapshot": snapshot}
    if not collector.terminal_state(state):
        return {"state": "waiting_for_terminal", "provider_status": state, "snapshot": snapshot}

    target = {
        "kernel": KERNEL_ID,
        "resource": "gpu",
        "artifact_dir": ARTIFACT_DIR,
        "terminal_receipt": TERMINAL_RECEIPT,
        "root_admission": "../../ops_continuation_20261012/clean_t3_f0_student_treatment_kaggle_r22/ROOT_TREATMENT_RELEASE_R7_R1.json",
        "root_admission_sha256": ROOT_RELEASE_SHA256,
        "push_receipt": "KERNEL_PUSH_R7_R1.json",
        "push_receipt_sha256": PUSH_RECEIPT_SHA256,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": SOURCE_VERSION,
        "asset_dataset": ASSET_DATASET,
        "asset_dataset_version": ASSET_VERSION,
        "provisional_reservation_seconds": LOGICAL_HOLD_GPU_SECONDS,
        "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
        "worker_max_seconds": WALL_SECONDS,
    }
    config: dict[str, Any] = {
        "repo_root": str(repo_root.resolve()),
        "config_dir": str(monitor.resolve()),
        "kaggle_cli": KAGGLE_CLI,
        "expected_cli_version": "Kaggle CLI 2.2.4",
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_download_bytes": MAX_DOWNLOAD_BYTES,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "max_attempts": 3,
    }
    result = collector.collect_terminal_target(
        target=target,
        status_row=row,
        config=config,
        repo_root=repo_root.resolve(),
        config_dir=monitor.resolve(),
    )
    return {
        "state": result["state"],
        "provider_status": state,
        "collection": result,
        "snapshot": snapshot,
        "classification_note": (
            "Collector verifies terminal command completion and hashes; CP5/full-pipeline "
            "classification remains a separate artifact audit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and collect the admitted R7 kernel only.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    bindings = validate_root_and_push_receipts(root)
    if args.check_only:
        print(json.dumps({"state": "binding_check_pass", "bindings": bindings}, sort_keys=True))
        return 0
    result = poll_and_collect(root)
    result["observed_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    result["bindings"] = bindings
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
