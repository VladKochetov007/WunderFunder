"""Collect only the terminal output of the reviewed F3 CP2 export kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

PACKET = Path("research/campaign_2026-10-15/f3_teacher_export_r1")
HELPER = Path(
    "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/collect_terminal_r4.py"
)
HELPER_SHA256 = "a6e258be44c92d52642046f3fb1ffb45787a2e2e3b66a8f71bcb8f37ad7b1899"
KERNEL_ID = "vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r1-runner"
JOB_NAME = "export_f3_teacher_cp2_triplet_20261015_r1"
MAX_BYTES = 2_500_000_000
MIN_FREE_BYTES = 4_000_000_000
MANIFEST_RESERVE_BYTES = 4_000_000
CHUNK_BYTES = 8 * 1024 * 1024

PREDICTION_ARTIFACTS = (
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.json",
    "results/f3_teacher_export_pipeline.json",
)
SUPPORT_OUTPUTS = (
    "results/inventory.json",
    "results/summary.json",
    f"results/{JOB_NAME}.input_identity.json",
    f"results/{JOB_NAME}/worker.log",
    f"results/{JOB_NAME}/spec.json",
    f"results/{JOB_NAME}/timeout_resolution.json",
    f"results/{JOB_NAME}/result.json",
)


def sha256_file(path: Path) -> str:
    """Hash a file with fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 4_000_000) -> Any:
    """Read bounded local receipts only after checking their size."""
    if path.stat().st_size > max_bytes:
        raise ValueError(f"receipt exceeds {max_bytes} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_remote_paths() -> set[str]:
    """Return the exact worker outputs plus runner inventory and terminal evidence."""
    expected = set(SUPPORT_OUTPUTS)
    expected.update(f"results/{JOB_NAME}/artifacts/{artifact}" for artifact in PREDICTION_ARTIFACTS)
    return expected


def load_helper(root: Path) -> ModuleType:
    """Load the reviewed streaming collector without changing its source."""
    helper_path = root / HELPER
    if sha256_file(helper_path) != HELPER_SHA256:
        raise ValueError("shared terminal-collection helper differs from the reviewed pin")
    spec = importlib.util.spec_from_file_location("f3_cp2_collection_support", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen streaming-collection helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.KERNEL_ID = KERNEL_ID
    helper.OWNER, helper.KERNEL_SLUG = KERNEL_ID.split("/", maxsplit=1)
    return helper


def declared_length(url: str, helper: ModuleType) -> int:
    """Get an HTTPS result length without downloading its response body."""
    if urlparse(url).scheme != "https":
        raise ValueError("provider returned a non-HTTPS result URL")
    try:
        with helper.requests.get(
            url, stream=True, allow_redirects=True, timeout=(15, 90)
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"result inventory GET headers returned HTTP {response.status_code}"
                )
            length = response.headers.get("Content-Length")
    except helper.requests.RequestException:
        raise RuntimeError("result inventory GET headers failed; no payload downloaded") from None
    if length is None or not length.isdecimal():
        raise ValueError("result inventory requires an explicit Content-Length")
    return int(length)


def inventory(
    files: list[object], log_bytes: int, helper: ModuleType, *, complete: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bound all result lengths before downloading, rejecting undeclared paths."""
    expected = expected_remote_paths()
    selected: dict[str, object] = {}
    for item in files:
        name = str(getattr(item, "file_name", ""))
        if not name.startswith("results/"):
            continue
        helper.relative_result_path(name)
        if name in selected:
            raise ValueError("duplicate path in Kaggle result inventory")
        if name not in expected:
            raise ValueError(f"undeclared Kaggle result path: {name}")
        selected[name] = item
    missing = sorted(expected - set(selected))
    rows: list[dict[str, Any]] = []
    total = log_bytes + MANIFEST_RESERVE_BYTES
    for name, item in sorted(selected.items()):
        length = declared_length(str(getattr(item, "url", "")), helper)
        total += length
        if total > MAX_BYTES:
            raise ValueError("declared result inventory exceeds the 2.5 GB collection cap")
        rows.append({"path": name, "size_bytes": length})
    if complete and missing:
        # Preserve partial results and session log, but mark the collection as incomplete.
        return rows, missing
    if not rows and not log_bytes:
        raise ValueError("terminal Kaggle session returned neither results nor a session log")
    return rows, missing


def verify_release_receipts(root: Path) -> dict[str, Any]:
    """Require root's exact kernel admission and pushed version before collection."""
    binding = read_json(root / PACKET / "SOURCE_BINDING_R1.json")
    admission = read_json(root / PACKET / "ROOT_ADMISSION_R1.json")
    push = read_json(root / PACKET / "KERNEL_PUSH_R1.json")
    expected_archive = binding["bundle"]["archive_sha256"]
    expected_metadata = binding["bundle"]["kernel_metadata_sha256"]
    if (
        admission.get("status") != "admitted"
        or admission.get("kernel_id") != KERNEL_ID
        or admission.get("bundle_archive_sha256") != expected_archive
        or admission.get("worker_timeout_seconds") != 18_000
        or push.get("kernel_id") != KERNEL_ID
        or type(push.get("kernel_version")) is not int
        or push["kernel_version"] < 1
        or push.get("bundle_archive_sha256") != expected_archive
        or push.get("kernel_metadata_sha256") != expected_metadata
        or push.get("private") is not True
    ):
        raise ValueError("root admission/push receipts do not bind this private export packet")
    return {
        "admission_sha256": sha256_file(root / PACKET / "ROOT_ADMISSION_R1.json"),
        "push_receipt_sha256": sha256_file(root / PACKET / "KERNEL_PUSH_R1.json"),
        "kernel_version": push["kernel_version"],
        "bundle_archive_sha256": expected_archive,
        "kernel_metadata_sha256": expected_metadata,
    }


def collect(root: Path) -> dict[str, Any]:
    """Inventory and stream one terminal version without overwrite or retry."""
    root = root.resolve(strict=True)
    packet = root / PACKET
    output = packet / "collected/attempt_001"
    terminal_path = packet / "TERMINAL_COLLECTION_R1.json"
    if output.exists() or terminal_path.exists():
        raise FileExistsError("collection path already exists; no automatic overwrite or retry")

    fs_type = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(packet)], text=True
    ).strip()
    if fs_type in {"tmpfs", "ramfs"} or not fs_type:
        raise ValueError("terminal collection requires a physical filesystem")
    if shutil.disk_usage(packet).free < MIN_FREE_BYTES:
        raise OSError("terminal collection requires four GB free physical disk")

    release = verify_release_receipts(root)
    helper = load_helper(root)
    api = helper.KaggleApi()
    api.authenticate()
    status, failure = helper.terminal_status(api)
    terminal_state = status.rsplit(".", maxsplit=1)[-1].upper()
    quota_before = helper.quota_snapshot(api)
    files, session_log = helper.output_listing(api)
    log_data = session_log.encode("utf-8")
    rows, missing = inventory(files, len(log_data), helper, complete=terminal_state == "COMPLETE")
    selected = {
        str(getattr(item, "file_name", "")): item
        for item in files
        if str(getattr(item, "file_name", "")) in {row["path"] for row in rows}
    }

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    helper.write_json(
        output / "remote_inventory.json",
        {
            "kernel_id": KERNEL_ID,
            "kernel_version_from_root_push_receipt": release["kernel_version"],
            "terminal_status": status,
            "files": rows,
            "missing_expected_paths": missing,
            "session_log_bytes": len(log_data),
            "inventory_cap_bytes": MAX_BYTES,
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
            selected[row["path"]], output, MAX_BYTES - downloaded - MANIFEST_RESERVE_BYTES
        )
        if record["size_bytes"] != row["size_bytes"]:
            raise OSError("downloaded result length differs from the pre-download inventory")
        downloaded += int(record["size_bytes"])
        downloaded_files.append(record)

    complete_outputs = terminal_state == "COMPLETE" and not missing
    collection = {
        "schema": "wnn.f3-teacher-cp2-export-terminal-collection.r1",
        "recorded_utc": helper.utc_now(),
        "kernel_id": KERNEL_ID,
        "kernel_version": release["kernel_version"],
        "provider_status": status,
        "provider_failure_message": failure,
        "source_binding_sha256": sha256_file(packet / "SOURCE_BINDING_R1.json"),
        "release_receipts": release,
        "collection_helper_sha256": HELPER_SHA256,
        "collection_source_sha256": sha256_file(Path(__file__)),
        "maximum_collection_bytes": MAX_BYTES,
        "downloaded_bytes_before_manifest": downloaded,
        "missing_expected_paths": missing,
        "files": downloaded_files,
        "quota_before": quota_before,
        "quota_after": helper.quota_snapshot(api),
        "complete_export_payload": complete_outputs,
        "automatic_retry": False,
        "output_scope": "prediction exports and execution receipts only; no score or model selection",
    }
    actual_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_size > MAX_BYTES:
        raise ValueError("collection metadata exceeded the 2.5 GB reserved byte cap")
    collection["bytes_including_inventory_and_log"] = actual_size
    helper.write_json(terminal_path, collection)
    return collection


def main() -> None:
    """Collect after root has admitted, pushed, and recorded the exact kernel version."""
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
