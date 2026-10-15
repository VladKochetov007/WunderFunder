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

PACKET = Path("research/campaign_2026-10-15/f3_teacher_export_r2")
HELPER = Path(
    "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/collect_terminal_r4.py"
)
HELPER_SHA256 = "a6e258be44c92d52642046f3fb1ffb45787a2e2e3b66a8f71bcb8f37ad7b1899"
KERNEL_ID = "vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r2-runner"
JOB_NAME = "export_f3_teacher_cp2_triplet_20261015_r2"
KERNEL_PROCESS_WALL_SECONDS_MAX = 21_600
OWNER_RUN = "clean_t3_f3_muonall_h384_e3_s3_r5"
CHECKPOINT_SHA256 = "d67bed97eaa16f73338ea275c316bf76c785b37f8451aa5cee8fe5daadf2f1b5"
PROVENANCE_SHA256 = "c60719d1db7d44d17e04e0379f576616a43c6ad2564fd58e192ced5453bc8da3"
SPLIT_SHA256 = "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343"
WRAPPER_SHA256 = "246586bfd4be3a4e3d95ac026860bb835611f09958193b75798124c9500455ab"
PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
MAX_BYTES = 2_500_000_000
MIN_FREE_BYTES = 4_000_000_000
MANIFEST_RESERVE_BYTES = 4_000_000
CHUNK_BYTES = 8 * 1024 * 1024
SEQUENCE_LENGTH = 20_000

PREDICTION_ARTIFACTS = (
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.json",
    "results/f3_teacher_export_pipeline_r2.json",
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
PREDICTION_STAGES = (
    {
        "name": "fit_train",
        "split": "train",
        "rows": 10539,
        "scope": "train",
        "array": "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.npy",
    },
    {
        "name": "fit_valid",
        "split": "valid",
        "rows": 1368,
        "scope": "valid",
        "array": "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.npy",
    },
    {
        "name": "full_valid",
        "split": "valid",
        "rows": 1873,
        "scope": "full-valid-feature-only",
        "array": "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.npy",
    },
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


def validate_collected_payload(
    output: Path, downloaded_files: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check downloaded runner receipts and stream-scan all three FP32 arrays."""
    import numpy as np

    result_path = output / "results" / JOB_NAME / "result.json"
    timeout_path = output / "results" / JOB_NAME / "timeout_resolution.json"
    pipeline_path = (
        output / "results" / JOB_NAME / "artifacts/results/f3_teacher_export_pipeline_r2.json"
    )
    result = read_json(result_path)
    timeout = read_json(timeout_path)
    pipeline = read_json(pipeline_path)
    if (
        result.get("name") != JOB_NAME
        or result.get("status") != "complete"
        or result.get("returncode") != 0
        or not isinstance(result.get("artifacts"), dict)
        or result["artifacts"].get("missing") != []
    ):
        raise ValueError("runner worker result is not a complete successful R2 export")
    if (
        timeout.get("status") != "resolved_before_worker_launch"
        or type(timeout.get("resolved_worker_timeout_seconds")) is not int
        or not 1 <= timeout["resolved_worker_timeout_seconds"] <= 18_000
        or timeout.get("kernel_deadline_monotonic") is None
        or timeout.get("finalization_reserve_seconds") != 300
    ):
        raise ValueError("worker timeout receipt does not fit the R2 kernel deadline")

    if (
        pipeline.get("schema") != "wnn.f3-teacher-cp2-export-pipeline.r2"
        or pipeline.get("status") != "complete"
        or pipeline.get("run") != OWNER_RUN
        or pipeline.get("checkpoint_epoch") != 2
        or pipeline.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or pipeline.get("provenance_sha256") != PROVENANCE_SHA256
        or pipeline.get("split_manifest_sha256") != SPLIT_SHA256
        or pipeline.get("source_wrapper_sha256") != WRAPPER_SHA256
        or pipeline.get("source_predictor_sha256") != PREDICTOR_SHA256
    ):
        raise ValueError("pipeline receipt does not bind the frozen F3 CP2 and R1 predictor")

    expected_artifact_names = set(PREDICTION_ARTIFACTS)
    artifact_rows = result["artifacts"].get("files")
    if not isinstance(artifact_rows, list):
        raise ValueError("runner result has no artifact hash manifest")
    artifact_map = {row.get("path"): row for row in artifact_rows if isinstance(row, dict)}
    if len(artifact_rows) != len(artifact_map) or set(artifact_map) != expected_artifact_names:
        raise ValueError("runner artifact manifest differs from the declared R2 outputs")
    downloaded_map = {row["path"]: row for row in downloaded_files}
    if len(downloaded_map) != len(downloaded_files):
        raise ValueError("duplicate path in the local downloaded-file manifest")
    downloaded_hashes = {path: row["sha256"] for path, row in downloaded_map.items()}
    pipeline_remote = f"results/{JOB_NAME}/artifacts/results/f3_teacher_export_pipeline_r2.json"
    pipeline_artifact = artifact_map["results/f3_teacher_export_pipeline_r2.json"]
    pipeline_path_stat = (output / pipeline_remote).stat()
    if (
        downloaded_hashes.get(pipeline_remote) != pipeline_artifact.get("sha256")
        or downloaded_map[pipeline_remote].get("size_bytes") != pipeline_artifact.get("size")
        or pipeline_path_stat.st_size != pipeline_artifact.get("size")
    ):
        raise ValueError("downloaded pipeline receipt differs from the runner hash manifest")

    stages = pipeline.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != [
        stage["name"] for stage in PREDICTION_STAGES
    ]:
        raise ValueError("pipeline receipt does not contain the three declared export stages")
    validated_stages: list[dict[str, Any]] = []
    payload_bytes = 0
    for stage_spec, stage in zip(PREDICTION_STAGES, stages, strict=True):
        if stage.get("status") != "complete":
            raise ValueError(f"pipeline stage is not complete: {stage_spec['name']}")
        array_relative = stage_spec["array"]
        sidecar_relative = str(Path(array_relative).with_suffix(".json"))
        remote_array = f"results/{JOB_NAME}/artifacts/{array_relative}"
        remote_sidecar = f"results/{JOB_NAME}/artifacts/{sidecar_relative}"
        array_path = output / remote_array
        sidecar_path = output / remote_sidecar
        metadata = read_json(sidecar_path)
        artifact_array = artifact_map[array_relative]
        artifact_sidecar = artifact_map[sidecar_relative]
        array_hash = downloaded_hashes.get(remote_array)
        if (
            metadata.get("run") != OWNER_RUN
            or metadata.get("split") != stage_spec["split"]
            or metadata.get("epoch") != 2
            or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or metadata.get("split_manifest_sha256") != SPLIT_SHA256
            or metadata.get("inference_precision") != "fp32"
            or metadata.get("campaign_fit_scope") != stage_spec["scope"]
            or not isinstance(metadata.get("row_groups"), list)
            or len(metadata["row_groups"]) != stage_spec["rows"]
            or len(set(metadata["row_groups"])) != stage_spec["rows"]
            or array_hash != metadata.get("prediction_sha256")
            or array_hash != stage.get("array_sha256")
            or array_hash != artifact_array.get("sha256")
            or sidecar_path.stat().st_size != artifact_sidecar.get("size")
            or downloaded_map[remote_sidecar].get("size_bytes") != artifact_sidecar.get("size")
            or downloaded_hashes.get(remote_sidecar) != artifact_sidecar.get("sha256")
            or stage.get("sidecar_bytes") != sidecar_path.stat().st_size
            or stage.get("sidecar_path") != sidecar_relative
            or stage.get("array_path") != array_relative
        ):
            raise ValueError(f"sidecar or transferred hash identity differs: {stage_spec['name']}")

        array_bytes = array_path.stat().st_size
        expected_payload_bytes = stage_spec["rows"] * SEQUENCE_LENGTH * 2 * 4
        if not expected_payload_bytes <= array_bytes <= expected_payload_bytes + 4096:
            raise ValueError(f"prediction array byte count differs: {stage_spec['name']}")
        expected_download_size = artifact_array.get("size")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if (
            array_bytes != expected_download_size
            or downloaded_map[remote_array].get("size_bytes") != expected_download_size
            or array.shape != (stage_spec["rows"], SEQUENCE_LENGTH, 2)
            or array.dtype != np.dtype("float32")
        ):
            raise ValueError(f"prediction shape or dtype differs: {stage_spec['name']}")
        finite_values = 0
        for first in range(0, stage_spec["rows"], 32):
            block = array[first : first + 32]
            finite = np.isfinite(block)
            if not bool(finite.all()):
                raise ValueError(f"non-finite prediction values: {stage_spec['name']} row {first}")
            finite_values += int(finite.size)
        del array
        expected_finite = stage_spec["rows"] * SEQUENCE_LENGTH * 2
        if (
            finite_values != expected_finite
            or stage.get("finite_values") != expected_finite
            or stage.get("shape") != [stage_spec["rows"], SEQUENCE_LENGTH, 2]
            or stage.get("dtype") != "float32"
            or stage.get("array_bytes") != array_bytes
            or stage.get("row_group_count") != stage_spec["rows"]
        ):
            raise ValueError(f"pipeline geometry or finite count differs: {stage_spec['name']}")
        payload_bytes += array_bytes
        validated_stages.append(
            {
                "name": stage_spec["name"],
                "shape": [stage_spec["rows"], SEQUENCE_LENGTH, 2],
                "dtype": "float32",
                "array_sha256": array_hash,
                "finite_values": finite_values,
                "array_bytes": array_bytes,
            }
        )
    if pipeline.get("prediction_payload_bytes") != payload_bytes:
        raise ValueError("pipeline total prediction bytes differ from collected arrays")
    return {
        "status": "passed",
        "worker_returncode": result["returncode"],
        "resolved_worker_timeout_seconds": timeout["resolved_worker_timeout_seconds"],
        "outer_kernel_process_wall_seconds_max": KERNEL_PROCESS_WALL_SECONDS_MAX,
        "prediction_payload_bytes": payload_bytes,
        "stages": validated_stages,
    }


def verify_release_receipts(root: Path) -> dict[str, Any]:
    """Require root's exact kernel admission and pushed version before collection."""
    binding = read_json(root / PACKET / "SOURCE_BINDING_R2.json")
    admission = read_json(root / PACKET / "ROOT_ADMISSION_R2.json")
    push = read_json(root / PACKET / "KERNEL_PUSH_R2.json")
    expected_archive = binding["bundle"]["archive_sha256"]
    expected_metadata = binding["bundle"]["kernel_metadata_sha256"]
    if (
        binding["bundle"].get("kernel_process_wall_seconds_max") != KERNEL_PROCESS_WALL_SECONDS_MAX
        or admission.get("status") != "admitted"
        or admission.get("kernel_id") != KERNEL_ID
        or admission.get("bundle_archive_sha256") != expected_archive
        or admission.get("kernel_process_wall_seconds_max") != KERNEL_PROCESS_WALL_SECONDS_MAX
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
        "admission_sha256": sha256_file(root / PACKET / "ROOT_ADMISSION_R2.json"),
        "push_receipt_sha256": sha256_file(root / PACKET / "KERNEL_PUSH_R2.json"),
        "kernel_version": push["kernel_version"],
        "kernel_process_wall_seconds_max": KERNEL_PROCESS_WALL_SECONDS_MAX,
        "bundle_archive_sha256": expected_archive,
        "kernel_metadata_sha256": expected_metadata,
    }


def collect(root: Path) -> dict[str, Any]:
    """Inventory and stream one terminal version without overwrite or retry."""
    root = root.resolve(strict=True)
    packet = root / PACKET
    output = packet / "collected/attempt_001"
    terminal_path = packet / "TERMINAL_COLLECTION_R2.json"
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
    artifact_validation: dict[str, Any] = {
        "status": "not_run",
        "reason": "provider session or expected result inventory was incomplete",
    }
    if complete_outputs:
        try:
            artifact_validation = validate_collected_payload(output, downloaded_files)
        except Exception as exc:
            artifact_validation = {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
            }
    complete_export = complete_outputs and artifact_validation.get("status") == "passed"
    collection = {
        "schema": "wnn.f3-teacher-cp2-export-terminal-collection.r2",
        "recorded_utc": helper.utc_now(),
        "kernel_id": KERNEL_ID,
        "kernel_version": release["kernel_version"],
        "provider_status": status,
        "provider_failure_message": failure,
        "source_binding_sha256": sha256_file(packet / "SOURCE_BINDING_R2.json"),
        "release_receipts": release,
        "collection_helper_sha256": HELPER_SHA256,
        "collection_source_sha256": sha256_file(Path(__file__)),
        "maximum_collection_bytes": MAX_BYTES,
        "downloaded_bytes_before_manifest": downloaded,
        "missing_expected_paths": missing,
        "files": downloaded_files,
        "quota_before": quota_before,
        "quota_after": helper.quota_snapshot(api),
        "provider_outputs_complete": complete_outputs,
        "artifact_validation": artifact_validation,
        "complete_export_payload": complete_export,
        "automatic_retry": False,
        "output_scope": "prediction exports and execution receipts only; no score or model selection",
    }
    actual_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_size > MAX_BYTES:
        raise ValueError("collection metadata exceeded the 2.5 GB reserved byte cap")
    collection["bytes_including_inventory_and_log"] = actual_size
    helper.write_json(terminal_path, collection)
    if complete_outputs and not complete_export:
        raise RuntimeError("downloaded R2 outputs failed validation; failure receipt was preserved")
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
