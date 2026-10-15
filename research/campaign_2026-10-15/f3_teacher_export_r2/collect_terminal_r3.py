"""Collect and validate the terminal F3 export, including runner heartbeat metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any  # JSON/provider records are dynamically typed at this boundary.

PACKET = Path("research/campaign_2026-10-15/f3_teacher_export_r2")
R2_COLLECTOR = PACKET / "collect_terminal_r2.py"
R2_COLLECTOR_SHA256 = "b7ad2f306ab49d8ae2b839e9cca5189c05d0583a953f46a7e902d0fb7e7acbdc"
BUNDLE_DIRECTORY = "bundle_wnn-f3-t3-cp2-exports-20261015-r2"
RUNNER_SOURCE_SHA256 = "ff95040cf4f964f3a7c86067a4603660bf80a5468ab71445ce7fcd36482fa9da"
HEARTBEAT_PATH = "results/heartbeat.json"
HEARTBEAT_MAX_BYTES = 16_384
OUTPUT_RELATIVE = PACKET / "collected/attempt_002"
TERMINAL_RECEIPT = PACKET / "TERMINAL_COLLECTION_R3.json"
HEARTBEAT_KEYS = {"event", "active", "finished", "not_started", "scope"}
HEARTBEAT_SCOPE = (
    "measured worker resource activity; not proof of epoch progress or scientific success"
)
ACTIVE_KEYS = {
    "name",
    "gpu_index",
    "gpu_uuid",
    "elapsed_seconds",
    "worker_timeout_seconds",
    "peak_ram_bytes",
    "peak_gpu_bytes",
}


def load_r2(root: Path) -> ModuleType:
    """Load the frozen R2 collector only after verifying its source hash."""
    path = root / R2_COLLECTOR
    if path.stat().st_size > 1_000_000:
        raise ValueError("frozen R2 collector exceeds the source-size bound")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != R2_COLLECTOR_SHA256:
        raise ValueError("frozen R2 collector differs from the reviewed source pin")
    spec = importlib.util.spec_from_file_location("f3_teacher_export_r2_collector", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen R2 collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_runner_source(root: Path, r2: ModuleType) -> dict[str, Any]:
    """Bind the heartbeat schema to the exact runner in the frozen deployed bundle."""
    bundle_dir = root / PACKET / BUNDLE_DIRECTORY
    binding = r2.read_json(root / PACKET / "SOURCE_BINDING_R2.json")
    manifest = r2.read_json(bundle_dir / "bundle-manifest.json")
    archive = bundle_dir / "bundle.wnn"
    expected_archive = binding["bundle"]["archive_sha256"]
    if (
        manifest.get("archive_sha256") != expected_archive
        or archive.stat().st_size != binding["bundle"].get("archive_bytes")
        or r2.sha256_file(archive) != expected_archive
    ):
        raise ValueError("deployed bundle archive differs from the frozen source binding")

    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("deployed bundle manifest has no file list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("path") == "scripts/kaggle_gpu_runner.py"
    ]
    if (
        len(matches) != 1
        or matches[0].get("sha256") != RUNNER_SOURCE_SHA256
        or type(matches[0].get("size")) is not int
    ):
        raise ValueError("bundle manifest does not bind the expected GPU runner source")

    source_path = root / "scripts/kaggle_gpu_runner.py"
    if r2.sha256_file(source_path) != RUNNER_SOURCE_SHA256:
        raise ValueError("local runner source differs from the deployed runner pin")
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = [
            member
            for member in bundle.getmembers()
            if member.name == "scripts/kaggle_gpu_runner.py"
        ]
        if len(members) != 1:
            raise ValueError("bundle archive has a missing or duplicate GPU runner path")
        info = members[0]
        if info.size > 1_000_000:
            raise ValueError("bundled GPU runner exceeds the source-size bound")
        if info.size != matches[0]["size"]:
            raise ValueError("bundled GPU runner size differs from its manifest")
        stream = bundle.extractfile(info)
        if stream is None:
            raise ValueError("bundled GPU runner cannot be read")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != RUNNER_SOURCE_SHA256:
        raise ValueError("archive GPU runner differs from its manifest pin")
    return {
        "path": "scripts/kaggle_gpu_runner.py",
        "sha256": RUNNER_SOURCE_SHA256,
        "heartbeat_output_path": "results/heartbeat.json",
        "heartbeat_writer": "run() writes resource_heartbeat() every 30 seconds",
    }


def validate_heartbeat_document(document: object) -> dict[str, int]:
    """Validate the small whitelisted operational JSON emitted by the runner."""
    if not isinstance(document, dict) or set(document) != HEARTBEAT_KEYS:
        raise ValueError("heartbeat JSON has an unexpected top-level schema")
    if (
        document.get("event") != "campaign_resource_heartbeat"
        or document.get("scope") != HEARTBEAT_SCOPE
    ):
        raise ValueError("heartbeat JSON does not match the pinned runner schema")

    active = document.get("active")
    finished = document.get("finished")
    not_started = document.get("not_started")
    if not isinstance(active, list) or len(active) > 32:
        raise ValueError("heartbeat active-worker list is outside its bound")
    if not isinstance(finished, list) or len(finished) > 32:
        raise ValueError("heartbeat finished-worker list is outside its bound")
    if not isinstance(not_started, list) or len(not_started) > 32:
        raise ValueError("heartbeat queued-worker list is outside its bound")

    for worker in active:
        if not isinstance(worker, dict) or set(worker) != ACTIVE_KEYS:
            raise ValueError("heartbeat active-worker record has unexpected fields")
        if (
            not isinstance(worker.get("name"), str)
            or not worker["name"]
            or len(worker["name"]) > 256
            or type(worker.get("gpu_index")) is not int
            or worker["gpu_index"] < 0
            or not isinstance(worker.get("gpu_uuid"), str)
            or len(worker["gpu_uuid"]) > 256
            or type(worker.get("worker_timeout_seconds")) is not int
            or worker["worker_timeout_seconds"] < 1
        ):
            raise ValueError("heartbeat active-worker identity or timeout is invalid")
        elapsed = worker.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("heartbeat elapsed time is invalid")
        for key in ("peak_ram_bytes", "peak_gpu_bytes"):
            if type(worker.get(key)) is not int or worker[key] < 0:
                raise ValueError(f"heartbeat {key} is invalid")

    for worker in finished:
        if (
            not isinstance(worker, dict)
            or set(worker) != {"name", "status"}
            or not isinstance(worker.get("name"), str)
            or not worker["name"]
            or len(worker["name"]) > 256
            or worker.get("status") not in {"complete", "failed"}
        ):
            raise ValueError("heartbeat finished-worker record is invalid")
    if any(not isinstance(name, str) or not name or len(name) > 256 for name in not_started):
        raise ValueError("heartbeat queued-worker name is invalid")
    return {
        "active_workers": len(active),
        "finished_workers": len(finished),
        "not_started_workers": len(not_started),
    }


def validate_heartbeat(path: Path, r2: ModuleType) -> dict[str, Any]:
    """Bound, parse and hash one collected heartbeat file."""
    size = path.stat().st_size
    if not 0 < size <= HEARTBEAT_MAX_BYTES:
        raise ValueError("runner heartbeat exceeds its 16 KiB size bound")
    document = json.loads(path.read_text(encoding="utf-8"))
    counts = validate_heartbeat_document(document)
    return {
        "status": "passed",
        "path": HEARTBEAT_PATH,
        "size_bytes": size,
        "sha256": r2.sha256_file(path),
        **counts,
    }


def inventory(
    files: list[object], log_bytes: int, helper: ModuleType, r2: ModuleType, *, complete: bool
) -> tuple[list[dict[str, Any]], list[str], object | None]:
    """Bound exact runner outputs, allowing only its pinned heartbeat metadata."""
    expected = r2.expected_remote_paths()
    selected: dict[str, object] = {}
    heartbeat = None
    for item in files:
        name = str(getattr(item, "file_name", ""))
        if not name.startswith("results/"):
            continue
        helper.relative_result_path(name)
        if name == HEARTBEAT_PATH:
            if heartbeat is not None:
                raise ValueError("duplicate runner heartbeat in Kaggle result inventory")
            heartbeat = item
            continue
        if name not in expected:
            raise ValueError(f"undeclared Kaggle result path: {name}")
        if name in selected:
            raise ValueError("duplicate path in Kaggle result inventory")
        selected[name] = item

    rows, missing = r2.inventory(list(selected.values()), log_bytes, helper, complete=complete)
    if heartbeat is not None:
        heartbeat_bytes = r2.declared_length(str(getattr(heartbeat, "url", "")), helper)
        if not 0 < heartbeat_bytes <= HEARTBEAT_MAX_BYTES:
            raise ValueError("declared runner heartbeat exceeds its 16 KiB size bound")
        rows.append({"path": HEARTBEAT_PATH, "size_bytes": heartbeat_bytes})
    else:
        missing.append(HEARTBEAT_PATH)

    total = log_bytes + r2.MANIFEST_RESERVE_BYTES + sum(int(row["size_bytes"]) for row in rows)
    if total > r2.MAX_BYTES:
        raise ValueError("declared result inventory exceeds the 2.5 GB collection cap")
    return sorted(rows, key=lambda row: row["path"]), sorted(missing), heartbeat


def collect(root: Path) -> dict[str, Any]:
    """Stream one already-terminal kernel version to a new immutable local attempt."""
    root = root.resolve(strict=True)
    packet = root / PACKET
    output = root / OUTPUT_RELATIVE
    terminal_path = root / TERMINAL_RECEIPT
    if output.exists() or terminal_path.exists():
        raise FileExistsError("R3 collection destination exists; automatic overwrite is disabled")

    fs_type = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(packet)], text=True
    ).strip()
    if fs_type in {"tmpfs", "ramfs"} or not fs_type:
        raise ValueError("terminal collection requires a physical filesystem")
    if shutil.disk_usage(packet).free < 4_000_000_000:
        raise OSError("terminal collection requires four GB free physical disk")

    r2 = load_r2(root)
    runner = verify_runner_source(root, r2)
    release = r2.verify_release_receipts(root)
    helper = r2.load_helper(root)
    api = helper.KaggleApi()
    api.authenticate()
    status, failure = helper.terminal_status(api)
    terminal_state = status.rsplit(".", maxsplit=1)[-1].upper()
    quota_before = helper.quota_snapshot(api)
    files, session_log = helper.output_listing(api)
    log_data = session_log.encode("utf-8")
    rows, missing, heartbeat_item = inventory(
        files, len(log_data), helper, r2, complete=terminal_state == "COMPLETE"
    )
    selected = {
        str(getattr(item, "file_name", "")): item
        for item in files
        if str(getattr(item, "file_name", "")) in {row["path"] for row in rows}
    }

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    helper.write_json(
        output / "remote_inventory.json",
        {
            "kernel_id": r2.KERNEL_ID,
            "kernel_version_from_root_push_receipt": release["kernel_version"],
            "terminal_status": status,
            "files": rows,
            "missing_expected_paths": missing,
            "session_log_bytes": len(log_data),
            "inventory_cap_bytes": r2.MAX_BYTES,
            "heartbeat_contract": {
                "path": HEARTBEAT_PATH,
                "maximum_bytes": HEARTBEAT_MAX_BYTES,
                "runner_source_sha256": RUNNER_SOURCE_SHA256,
            },
        },
    )
    log_path = output / "kernel.log"
    with log_path.open("xb") as stream:
        stream.write(log_data)
    downloaded = len(log_data)
    downloaded_files: list[dict[str, Any]] = [
        {
            "path": "kernel.log",
            "size_bytes": len(log_data),
            "sha256": hashlib.sha256(log_data).hexdigest(),
        }
    ]
    for row in rows:
        record = helper.download_result_file(
            selected[row["path"]],
            output,
            r2.MAX_BYTES - downloaded - r2.MANIFEST_RESERVE_BYTES,
        )
        if record["size_bytes"] != row["size_bytes"]:
            raise OSError("downloaded result length differs from the pre-download inventory")
        downloaded += int(record["size_bytes"])
        downloaded_files.append(record)

    heartbeat_validation: dict[str, Any]
    if heartbeat_item is None:
        heartbeat_validation = {"status": "missing", "path": HEARTBEAT_PATH}
    else:
        try:
            heartbeat_validation = validate_heartbeat(output / HEARTBEAT_PATH, r2)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            heartbeat_validation = {
                "status": "failed",
                "path": HEARTBEAT_PATH,
                "failure": f"{type(exc).__name__}: {exc}",
            }

    complete_outputs = terminal_state == "COMPLETE" and not missing
    artifact_validation: dict[str, Any] = {
        "status": "not_run",
        "reason": "provider session or expected result inventory was incomplete",
    }
    if complete_outputs:
        try:
            artifact_validation = r2.validate_collected_payload(output, downloaded_files)
        except Exception as exc:
            artifact_validation = {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
            }
    complete_export = (
        complete_outputs
        and artifact_validation.get("status") == "passed"
        and heartbeat_validation.get("status") == "passed"
    )
    collection = {
        "schema": "wnn.f3-teacher-cp2-export-terminal-collection.r3",
        "recorded_utc": helper.utc_now(),
        "kernel_id": r2.KERNEL_ID,
        "kernel_version": release["kernel_version"],
        "provider_status": status,
        "provider_failure_message": failure,
        "source_binding_sha256": r2.sha256_file(packet / "SOURCE_BINDING_R2.json"),
        "release_receipts": release,
        "r2_collector_sha256": R2_COLLECTOR_SHA256,
        "collection_source_sha256": r2.sha256_file(Path(__file__)),
        "runner_source": runner,
        "maximum_collection_bytes": r2.MAX_BYTES,
        "downloaded_bytes_before_manifest": downloaded,
        "missing_expected_paths": missing,
        "files": downloaded_files,
        "quota_before": quota_before,
        "quota_after": helper.quota_snapshot(api),
        "provider_outputs_complete": complete_outputs,
        "artifact_validation": artifact_validation,
        "heartbeat_validation": heartbeat_validation,
        "complete_export_payload": complete_export,
        "automatic_retry": False,
        "output_scope": "prediction exports and execution receipts only; no score or model selection",
    }
    actual_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_size > r2.MAX_BYTES:
        raise ValueError("collection metadata exceeded the 2.5 GB reserved byte cap")
    collection["bytes_including_inventory_and_log"] = actual_size
    helper.write_json(terminal_path, collection)
    if complete_outputs and not complete_export:
        raise RuntimeError("downloaded R3 outputs failed validation; failure receipt was preserved")
    return collection


def main() -> None:
    """Collect after the reviewed kernel version has reached terminal status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.repo_root)
    print(
        json.dumps(
            {
                "kernel_id": result["kernel_id"],
                "kernel_version": result["kernel_version"],
                "provider_status": result["provider_status"],
                "complete_export_payload": result["complete_export_payload"],
                "bytes_including_inventory_and_log": result["bytes_including_inventory_and_log"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
