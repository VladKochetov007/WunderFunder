"""Export the reviewed H232 fixed epoch-3 checkpoint without fitting or scoring."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from membership_audit import audit_fit_membership

PACKET_ID = "E529_H232_V7B_EP3_FP32_EXPORT_ONLY_20261008_R1"
RUN_NAME = "bd26_h232_full_s3_f0_amp4096_v7_proactive_20261008"
CHECKPOINT_SHA256 = "aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829"
SOURCE_ARCHIVE_SHA256 = "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4"
SOURCE_MANIFEST_SHA256 = "6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
WORKER_WALL_CAP_S = 4_500
PREDICTION_EXPORT_CAP_S = 1_800
EXPECTED_VALID_COUNT = 1_873
EXPECTED_EVALUATION_COUNT = 468
EXPECTED_PREDICTION_SHAPE = (EXPECTED_VALID_COUNT, 20_000, 2)
EXPECTED_PREDICTION_BYTES = 2_996_800_000
WORK_ROOT_NAME = "h232_v7b_ep3_export_only_20261008"
SOURCE_ROOT_NAME = "h232_v7b_ep3_export_source"
OUTPUT_NAME = "h232_v7b_ep3_export_only_20261008"
_ACTIVE_CHILD: subprocess.Popen[bytes] | None = None


class ExportAbort(RuntimeError):
    """Raised when a delivery-only identity, resource, or export gate fails."""


def _sha256_file(path: Path) -> str:
    """Hash a file using bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a JSON receipt in the worker output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _phase(path: Path, phase: str, **details: Any) -> None:
    """Write a durable phase heartbeat."""
    _atomic_json(
        path,
        {
            "packet_id": PACKET_ID,
            "status": "active",
            "phase": phase,
            "updated_unix": time.time(),
            **details,
        },
    )


def _verify_payload(packet_root: Path, packet_meta: dict[str, Any]) -> None:
    """Check all embedded source, checkpoint, and recipe hashes before execution."""
    files = packet_meta.get("payload_file_sha256")
    if not isinstance(files, dict) or not files:
        raise ExportAbort("packet payload hash map is missing")
    for relative, expected in files.items():
        path = packet_root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ExportAbort(f"embedded payload hash mismatch: {relative}")
    if _sha256_file(packet_root / "source_archive.tar") != SOURCE_ARCHIVE_SHA256:
        raise ExportAbort("frozen H232 source archive digest differs")
    if _sha256_file(packet_root / "source_archive_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ExportAbort("frozen H232 source archive manifest digest differs")
    if _sha256_file(packet_root / "model_ep3.pt") != CHECKPOINT_SHA256:
        raise ExportAbort("fixed epoch-3 checkpoint digest differs")
    frozen = packet_meta.get("frozen_identity")
    if not isinstance(frozen, dict):
        raise ExportAbort("frozen source/checkpoint identity is missing")
    if _sha256_file(packet_root / "provenance.json") != frozen.get("provenance_sha256"):
        raise ExportAbort("V7B trainer provenance digest differs")
    if _sha256_file(packet_root / "checkpoint_source_review.json") != frozen.get(
        "checkpoint_source_review_sha256"
    ):
        raise ExportAbort("checkpoint tensor/source review receipt digest differs")
    if _sha256_file(packet_root / "terminal_receipt.json") != frozen.get("terminal_receipt_sha256"):
        raise ExportAbort("V7B terminal receipt digest differs")


def _extract_source(packet_root: Path, source_root: Path) -> dict[str, Any]:
    """Extract only regular source members matching the frozen SHA manifest."""
    if source_root.exists():
        raise ExportAbort(f"refusing to reuse source path: {source_root}")
    manifest = json.loads((packet_root / "source_archive_manifest.json").read_text())
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ExportAbort("source archive manifest members are missing")
    expected = {
        item["path"]: (int(item["bytes"]), item["sha256"])
        for item in members
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if len(expected) != len(members) or manifest.get("member_count") != len(expected):
        raise ExportAbort("source archive manifest has duplicate or malformed paths")
    source_root.mkdir(parents=True)
    observed: set[str] = set()
    with tarfile.open(packet_root / "source_archive.tar", mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ExportAbort(f"unsafe source archive path: {member.name}")
            if member.isdir():
                source_root.joinpath(*name.parts).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.name not in expected:
                raise ExportAbort(f"unexpected nonregular source archive member: {member.name}")
            if member.name in observed:
                raise ExportAbort(f"duplicate source archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ExportAbort(f"source archive member cannot be read: {member.name}")
            data = stream.read()
            size, digest = expected[member.name]
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ExportAbort(f"source member hash mismatch: {member.name}")
            destination = source_root.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            observed.add(member.name)
    if observed != set(expected):
        raise ExportAbort("source archive is missing one or more manifest members")
    return {"member_count": len(observed), "manifest_sha256": SOURCE_MANIFEST_SHA256}


def _handle_termination(signum: int, _frame: object) -> None:
    """Stop an active prediction subprocess before recording worker termination."""
    global _ACTIVE_CHILD
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
    raise SystemExit(128 + signum)


def _run_prediction(
    *,
    source_root: Path,
    receipts: Path,
    guard: Any,
    phase_path: Path,
    worker_start: float,
) -> dict[str, Any]:
    """Invoke the frozen FP32 predictor with fixed epoch 3 and strict time caps."""
    global _ACTIVE_CHILD
    predictor = source_root / "scripts/predict_run.py"
    argv = [
        sys.executable,
        str(predictor),
        RUN_NAME,
        "valid",
        "--epoch",
        "3",
        "--output-name",
        OUTPUT_NAME,
        "--inference-precision",
        "fp32",
    ]
    stdout_path = receipts / "prediction.stdout.log"
    stderr_path = receipts / "prediction.stderr.log"
    prediction = source_root / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    partial = prediction.with_name(prediction.name + ".partial")
    if any(path.exists() for path in (prediction, sidecar, partial)):
        raise ExportAbort("refusing to overwrite an existing fixed-epoch export")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "WNN_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "WNN_LOADER_WORKERS": "2",
            "WNN_LOADER_RAM_GB": "0.30",
            "WNN_GPU_FRACTION": "0.75",
        }
    )
    export_start = time.monotonic()
    next_heartbeat = export_start
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        child = subprocess.Popen(
            argv,
            cwd=source_root,
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        _ACTIVE_CHILD = child
        while child.poll() is None:
            now = time.monotonic()
            if guard.failure:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise ExportAbort(f"hard resource guard failed: {guard.failure}")
            if now - export_start > PREDICTION_EXPORT_CAP_S:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise ExportAbort("fixed epoch-3 FP32 prediction exceeded 1,800 seconds")
            if now - worker_start > WORKER_WALL_CAP_S:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise ExportAbort("export-only worker exceeded 4,500 seconds")
            if now >= next_heartbeat:
                _phase(
                    phase_path,
                    "fp32_valid_prediction_export",
                    export_elapsed_seconds=round(now - export_start, 3),
                    worker_elapsed_seconds=round(now - worker_start, 3),
                    child_pid=child.pid,
                )
                next_heartbeat = now + 10
            time.sleep(1)
        return_code = child.wait()
        _ACTIVE_CHILD = None
    guard.raise_if_failed()
    export_seconds = round(time.monotonic() - export_start, 6)
    if return_code != 0:
        raise ExportAbort(f"fixed epoch-3 predictor exited with status {return_code}")
    if not prediction.is_file() or not sidecar.is_file() or partial.exists():
        raise ExportAbort("predictor did not finish both output files cleanly")

    import numpy as np

    array = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if array.shape != EXPECTED_PREDICTION_SHAPE or array.dtype != np.float32:
        raise ExportAbort("fixed-epoch prediction has unexpected shape or dtype")
    if prediction.stat().st_size != EXPECTED_PREDICTION_BYTES:
        raise ExportAbort("fixed-epoch prediction has an unexpected file size")
    finite_count = 0
    for start in range(0, EXPECTED_VALID_COUNT, 32):
        block = array[start : start + 32]
        finite_count += int(np.isfinite(block).sum())
    expected_values = EXPECTED_VALID_COUNT * 20_000 * 2
    if finite_count != expected_values:
        raise ExportAbort("fixed-epoch prediction contains a nonfinite value")
    del array
    metadata = json.loads(sidecar.read_text())
    with np.load(source_root / "cache/valid_targets.npz", allow_pickle=False) as targets:
        expected_groups = targets["row_group"].astype(np.int64, copy=False).tolist()
    if metadata.get("run") != RUN_NAME or metadata.get("split") != "valid":
        raise ExportAbort("prediction sidecar identifies a different run or split")
    if metadata.get("epoch") != 3 or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ExportAbort("prediction sidecar is not bound to the fixed epoch-3 checkpoint")
    if metadata.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ExportAbort("prediction sidecar split fingerprint differs from fold 0")
    if metadata.get("inference_precision") != "fp32":
        raise ExportAbort("prediction sidecar does not certify FP32 inference")
    if (
        metadata.get("row_groups") != expected_groups
        or len(expected_groups) != EXPECTED_VALID_COUNT
    ):
        raise ExportAbort("prediction row order differs from the verified valid target cache")
    if _sha256_file(prediction) != metadata.get("prediction_sha256"):
        raise ExportAbort("prediction sidecar digest differs from output bytes")
    return {
        "status": "complete",
        "argv": argv,
        "prediction_relative_path": str(prediction.relative_to(source_root)),
        "prediction_sha256": metadata["prediction_sha256"],
        "prediction_bytes": prediction.stat().st_size,
        "prediction_shape": list(EXPECTED_PREDICTION_SHAPE),
        "prediction_dtype": "float32",
        "finite_value_count": finite_count,
        "valid_sequence_count": len(expected_groups),
        "evaluation_sequence_count": EXPECTED_EVALUATION_COUNT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sidecar_sha256": _sha256_file(sidecar),
        "export_seconds": export_seconds,
        "scoring_enabled": False,
    }


def main(packet_root: Path) -> None:
    """Validate the frozen attempt, run only the epoch-3 export, and persist receipts."""
    packet_root = packet_root.resolve(strict=True)
    work = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    receipts = work / WORK_ROOT_NAME
    receipts.mkdir(parents=True, exist_ok=False)
    source_root = work / SOURCE_ROOT_NAME
    phase_path = receipts / "phase.json"
    started = time.monotonic()
    started_unix = time.time()
    phase = {"packet_id": PACKET_ID, "status": "active", "phase": "payload_review"}
    _atomic_json(phase_path, phase)
    packet_meta: dict[str, Any] = {}
    guard = None
    try:
        signal.signal(signal.SIGTERM, _handle_termination)
        packet_meta = json.loads((packet_root / "packet_meta.json").read_text())
        if packet_meta.get("packet_id") != PACKET_ID:
            raise ExportAbort("packet identity differs from this export-only worker")
        if any(
            packet_meta.get(key) is not False
            for key in (
                "training_enabled",
                "fit_enabled",
                "profile_enabled",
                "scoring_enabled",
                "provider_push_authorized",
            )
        ):
            raise ExportAbort(
                "packet permits training, profiling, scoring, or an unauthorized push"
            )
        _verify_payload(packet_root, packet_meta)
        _phase(phase_path, "frozen_source_extract")
        source_audit = _extract_source(packet_root, source_root)

        sys.path.insert(0, str(packet_root))
        import torch
        from profile_worker import (
            LiveGuard,
            require_physical_gpu_inventory,
            resource_snapshot,
            runtime_identity,
            verify_and_stage_inputs,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ExportAbort("export requires the admitted single CUDA device")
        torch.cuda.init()
        identity = runtime_identity(torch)
        require_physical_gpu_inventory(identity)
        if "T4" not in identity.get("selected_gpu_name", ""):
            raise ExportAbort("runtime device differs from the admitted Tesla T4 export lane")
        selected_uuid = identity.get("selected_physical_gpu_uuid")
        if not isinstance(selected_uuid, str) or not selected_uuid:
            raise ExportAbort("physical GPU UUID binding is unavailable")
        _atomic_json(receipts / "runtime_identity.json", identity)
        guard = LiveGuard(torch, work, receipts, started, selected_uuid)
        initial_snapshot = resource_snapshot(torch, work, expected_physical_gpu_uuid=selected_uuid)
        guard.check(initial_snapshot)
        guard.start_polling()

        _phase(phase_path, "input_identity_and_fold_verification")
        mounted = verify_and_stage_inputs(
            Path("/kaggle/input"),
            source_root,
            receipts / "input_identity.json",
            packet_root / "prior_input_identity_v2.json",
        )
        provenance_path = packet_root / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        split_path = source_root / "research/2026-09-29/artifacts/splits/fold0.json"
        # The source archive uses the repository-relative research path fixed in V7.
        if not split_path.is_file():
            split_path = source_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        split_manifest = json.loads(split_path.read_text())
        membership = audit_fit_membership(
            provenance, split_manifest, expected_split_sha256=SPLIT_SHA256
        )
        if membership.get("evaluation_count") != EXPECTED_EVALUATION_COUNT:
            raise ExportAbort(
                "source-qualified fold audit did not recover the 468-group evaluation set"
            )
        if membership.get("cross_source_integer_alias_count") != 463:
            raise ExportAbort("actual V7B integer-alias regression signature changed")
        if membership.get("fit_evaluation_component_overlap_count") != 0:
            raise ExportAbort("source-qualified fit/evaluation component overlap exists")
        _atomic_json(receipts / "source_qualified_membership_audit.json", membership)
        _atomic_json(
            receipts / "input_and_membership_validation.json",
            {
                "status": "passed",
                "input_identity_status": mounted["identity"].get("status"),
                "split_sha256": mounted["split"].sha256,
                "source_qualified_membership": membership,
                "component_checks_enabled": True,
                "scoring_enabled": False,
            },
        )

        checkpoint = packet_root / "model_ep3.pt"
        run_dir = source_root / "artifacts/runs" / RUN_NAME
        if run_dir.exists():
            raise ExportAbort("refusing to overwrite an existing run directory")
        run_dir.mkdir(parents=True)
        checkpoint_path = run_dir / "model_ep3.pt"
        shutil.copyfile(checkpoint, checkpoint_path)
        if _sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
            raise ExportAbort("staged checkpoint digest changed during copy")
        shutil.copyfile(provenance_path, run_dir / "provenance.json")
        (run_dir / "export_ancestry.json").write_text(
            json.dumps(
                {
                    "attempt": PACKET_ID,
                    "parent_fit_attempt": "H232_V7B_SHORTSLUG",
                    "selected_epoch": 3,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "source_qualified_membership_audit": membership,
                    "training_enabled": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        _phase(
            phase_path, "fp32_valid_prediction_export", export_cap_seconds=PREDICTION_EXPORT_CAP_S
        )
        export = _run_prediction(
            source_root=source_root,
            receipts=receipts,
            guard=guard,
            phase_path=phase_path,
            worker_start=started,
        )
        guard.raise_if_failed()
        _atomic_json(receipts / "valid_prediction_export.json", export)
        _atomic_json(
            receipts / "terminal.json",
            {
                "packet_id": PACKET_ID,
                "status": "complete",
                "started_unix": started_unix,
                "completed_unix": time.time(),
                "worker_wall_seconds": round(time.monotonic() - started, 6),
                "worker_wall_cap_seconds": WORKER_WALL_CAP_S,
                "source_archive": source_audit,
                "source_qualified_membership": membership,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "prediction": export,
                "training_enabled": False,
                "profile_enabled": False,
                "scoring_enabled": False,
            },
        )
        _atomic_json(
            phase_path,
            {
                "packet_id": PACKET_ID,
                "status": "complete",
                "phase": "terminal_export_verified",
                "updated_unix": time.time(),
            },
        )
    except BaseException as error:
        if guard is not None:
            guard.stop_polling()
        _atomic_json(
            receipts / "terminal.json",
            {
                "packet_id": PACKET_ID,
                "status": "failed",
                "started_unix": started_unix,
                "failed_unix": time.time(),
                "worker_wall_seconds": round(time.monotonic() - started, 6),
                "error_type": type(error).__name__,
                "error": str(error)[:4_000],
                "training_enabled": False,
                "scoring_enabled": False,
            },
        )
        _atomic_json(
            phase_path,
            {
                "packet_id": PACKET_ID,
                "status": "failed",
                "phase": "terminal_failure",
                "updated_unix": time.time(),
                "error_type": type(error).__name__,
            },
        )
        raise
    finally:
        if guard is not None:
            guard.stop_polling()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_worker.py PACKET_ROOT")
    main(Path(sys.argv[1]))
