"""Poll and collect only the admitted native GPU FP16 kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Provider responses and JSON receipts are untyped external boundaries.
REPO_ROOT = Path(__file__).resolve().parents[6]
NATIVE_DIR = Path(__file__).resolve().parent
CLOUD_PREFLIGHT = REPO_ROOT / "research/campaign_2026-10-06/cloud_preflight"
sys.path.insert(0, str(CLOUD_PREFLIGHT))

from collect_terminal_outputs import (  # noqa: E402
    collect_terminal_target,
    resolve_repo_path,
    sha256_file,
    terminal_state,
)
from poll_provider import append_jsonl  # noqa: E402
from poll_provider import collect as collect_provider_snapshot  # noqa: E402

CONFIG_NAME = "native_terminal_collection_targets_r1.json"
DISPATCH_NAME = "ROOT_NATIVE_GPU_FP16_DISPATCH_R1.json"
EXPECTED_DISPATCH_SHA256 = "f0750930dc3afcd5573bcc89b0aefe2b50843885b7254358ffa1144a09981231"
EXPECTED_CLI_VERSION = "Kaggle CLI 2.2.4"


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from a local receipt boundary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_binding(native_dir: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the native collector target against the admitted provider artifacts.

    The existing collector owns bounded Kaggle output downloads. This adapter
    binds its helper to the one native packet and its immutable push receipt.
    """
    native_dir = native_dir.resolve()
    repo_root = repo_root.resolve()
    if not native_dir.is_relative_to(repo_root):
        raise ValueError("native collector directory must remain inside the repository")

    config_path = native_dir / CONFIG_NAME
    config = _read_json_object(config_path)
    if config.get("schema") != "wnn.kaggle-terminal-collector-targets.v1":
        raise ValueError("unsupported native collector config schema")
    if (
        config.get("kaggle_cli")
        != "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/kaggle"
        or config.get("expected_cli_version") != EXPECTED_CLI_VERSION
        or config.get("minimum_free_bytes") != 4 * 1024**3
        or config.get("maximum_download_bytes") != 8 * 1024**3
        or config.get("max_runtime_seconds") != 800
        or config.get("max_attempts") != 3
        or config.get("poll_log") != "NATIVE_GPU_FP16_TERMINAL_COLLECTOR_POLL_LOG_R1.jsonl"
    ):
        raise ValueError("native collector runtime bounds or paths changed")

    dispatch_path = native_dir / DISPATCH_NAME
    if sha256_file(dispatch_path) != EXPECTED_DISPATCH_SHA256:
        raise ValueError("root dispatch receipt hash mismatch")
    dispatch = _read_json_object(dispatch_path)
    if dispatch.get("state") != "one_exact_private_source_create_then_kernel_push_authorized":
        raise ValueError("root dispatch state mismatch")

    packet_path = repo_root / str(dispatch.get("packet", ""))
    code_path = native_dir / "kernel/wnn-native-gpu-fp16-20261009-r1.py"
    metadata_path = native_dir / "kernel/kernel-metadata.json"
    source_manifest_path = native_dir / "source_manifest.json"
    asset_manifest_path = native_dir / "source_asset/assets-manifest.json"
    admission_relative_path = "../native_precision_gpu_r1/ROOT_NATIVE_GPU_FP16_ADMISSION_R1.json"
    admission_path = resolve_repo_path(repo_root, native_dir, admission_relative_path)
    push_path = native_dir / "NATIVE_GPU_FP16_PROVIDER_PUSH_R1.json"
    source_ready_path = native_dir / "SOURCE_READY_REMOTE_VERIFICATION_R1.json"
    prepush_path = native_dir / "FRESH_GPU_QUOTA_ADMISSION_R1.json"

    pinned_files = (
        (packet_path, dispatch.get("packet_sha256")),
        (code_path, dispatch.get("bootstrap_sha256")),
        (metadata_path, dispatch.get("metadata_sha256")),
        (source_manifest_path, dispatch.get("source_manifest_sha256")),
        (asset_manifest_path, dispatch.get("source_asset_manifest_sha256")),
        (admission_path, dispatch.get("root_admission_sha256")),
    )
    for path, expected_sha256 in pinned_files:
        if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
            raise ValueError(f"admitted source hash mismatch: {path.name}")

    source_ready = _read_json_object(source_ready_path)
    if (
        source_ready.get("state") != "verified_ready_v1_exact_inventory_and_manifest"
        or source_ready.get("dataset") != dispatch.get("source_dataset")
        or source_ready.get("provider_status") != {"status": "ready", "current_version_number": 1}
    ):
        raise ValueError("source readiness receipt does not bind READY v1")

    prepush = _read_json_object(prepush_path)
    if (
        prepush.get("result") != "pass"
        or prepush.get("dispatch_receipt_sha256") != EXPECTED_DISPATCH_SHA256
        or prepush.get("kernel") != dispatch.get("kernel")
    ):
        raise ValueError("fresh pre-push admission check did not pass")

    push = _read_json_object(push_path)
    if (
        push.get("state") != "accepted_v1_worker_running_phase_not_yet_confirmed"
        or push.get("push_exit_code") != 0
        or push.get("push_argv") != dispatch.get("push_argv")
        or push.get("dispatch_receipt_sha256") != EXPECTED_DISPATCH_SHA256
        or push.get("packet_sha256") != dispatch.get("packet_sha256")
        or push.get("root_admission_sha256") != dispatch.get("root_admission_sha256")
        or push.get("source_ready_receipt_sha256") != sha256_file(source_ready_path)
        or push.get("prepush_readiness_sha256") != sha256_file(prepush_path)
    ):
        raise ValueError("provider push receipt does not match the admitted source and packet")
    provider = push.get("provider_observation")
    if not isinstance(provider, dict):
        raise ValueError("provider observation is missing from the push receipt")
    provider_metadata = provider.get("pulled_metadata")
    if not isinstance(provider_metadata, dict) or (
        provider_metadata.get("id") != dispatch.get("kernel")
        or provider_metadata.get("is_private") is not True
        or provider_metadata.get("enable_gpu") is not True
        or provider_metadata.get("enable_tpu") is not False
        or provider_metadata.get("machine_shape") != dispatch.get("accelerator")
    ):
        raise ValueError("actual provider kernel metadata does not match the admitted T4")
    if provider.get("pulled_code_sha256") != dispatch.get("bootstrap_sha256"):
        raise ValueError("provider-pulled source differs from the admitted bootstrap")

    targets = config.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        raise ValueError("native collector must have exactly one target")
    target = targets[0]
    expected_target = {
        "kernel": dispatch["kernel"],
        "resource": "gpu",
        "metadata_id": dispatch["kernel"],
        "root_admission": admission_relative_path,
        "root_admission_sha256": dispatch["root_admission_sha256"],
        "artifact_dir": "terminal_collect_r1",
        "terminal_receipt": "NATIVE_GPU_FP16_TERMINAL_OPERATOR_RECEIPT_R1.json",
        "push_receipt": push_path.name,
        "push_receipt_sha256": sha256_file(push_path),
        "admission_bindings": {
            "packet_sha256": dispatch["packet_sha256"],
            "bootstrap_sha256": dispatch["bootstrap_sha256"],
            "metadata_sha256": dispatch["metadata_sha256"],
            "source_manifest_sha256": dispatch["source_manifest_sha256"],
            "source_asset_manifest_sha256": dispatch["source_asset_manifest_sha256"],
        },
        "source_dataset": dispatch["source_dataset"],
        "source_dataset_version": dispatch["source_dataset_version_required"],
        "source_manifest_sha256": dispatch["source_manifest_sha256"],
        "provisional_reservation_seconds": dispatch["provisional_gpu_seconds"],
        "provider_timeout_seconds": dispatch["provider_timeout_seconds"],
        "worker_max_seconds": dispatch["worker_wall_seconds"],
    }
    if target != expected_target:
        raise ValueError("native collector target differs from the exact admitted binding")
    return config, target


def run_once(native_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Poll the admitted kernel and collect its artifacts after terminal status."""
    config, target = validate_binding(native_dir, repo_root)
    native_dir = native_dir.resolve()
    repo_root = repo_root.resolve()
    config["repo_root"] = str(repo_root)
    config["config_dir"] = str(native_dir)
    snapshot = collect_provider_snapshot(native_dir / CONFIG_NAME)
    poll_log = resolve_repo_path(repo_root, native_dir, str(config["poll_log"]))
    append_jsonl(poll_log, snapshot)

    status_row = next(
        (row for row in snapshot.get("status", []) if row.get("kernel") == target["kernel"]),
        None,
    )
    if status_row is None:
        outcome = {"kernel": target["kernel"], "state": "status_missing"}
    elif status_row.get("monitoring_state") == "stopped_after_terminal_evidence_downloaded":
        outcome = {"kernel": target["kernel"], "state": "already_collected"}
    elif not terminal_state(str(status_row.get("status", ""))):
        outcome = {
            "kernel": target["kernel"],
            "state": "provider_not_terminal",
            "status": status_row.get("status"),
        }
    else:
        outcome = collect_terminal_target(
            target=target,
            status_row=status_row,
            config=config,
            repo_root=repo_root,
            config_dir=native_dir,
        )

    result = {
        "observed_utc": snapshot.get("observed_at_utc"),
        "quota": snapshot.get("quota"),
        "targets": [outcome],
    }
    append_jsonl(poll_log, {"collector_result": result})
    return result


def parse_args() -> argparse.Namespace:
    """Parse the native-only collector paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=NATIVE_DIR)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    """Perform one bounded status poll and terminal collection pass."""
    args = parse_args()
    result = run_once(args.native_dir, args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
