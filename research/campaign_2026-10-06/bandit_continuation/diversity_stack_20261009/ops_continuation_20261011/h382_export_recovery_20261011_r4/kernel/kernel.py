"""Recover validated H382 predictions from rescued fold-0 checkpoints."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path

BUNDLE_SHA256 = "29da33ce7ad3273d830313c15614f2327ddf72597f6d6a9d55fd19e33880d31e"
BUNDLE_BYTES = int("359355")
SOURCE_PACKET_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3"
PACKET_ID = "vladkochetov07/wnn-h382-export-recovery-20261011-r4"
BUNDLE_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r3"
CHECKPOINT_DATASET_ID = "vladkochetov07/wnn-h382-rescued-checkpoints-20261011-r4"
CHECKPOINT_DATASET_SLUG = CHECKPOINT_DATASET_ID.split("/", 1)[1]
CHECKPOINT_ARCHIVE_BYTES = 12441600
CHECKPOINT_ARCHIVE_SHA256 = "a57df09373de67618c7a5826a11a3756fbdb1640b38781af1467a6cbbc468285"
CHECKPOINT_MANIFEST_SHA256 = "bca79e00f62e89bae7b44a4261682d9cfc4ac6d280c6610bbe14f48619685164"
WUNDER_ROOT = Path("/kaggle/input/datasets/vladkochetov07/wunder-funder")
ASSET_ROOT = Path("/kaggle/input/datasets/vladkochetov07/wnn-c20261004-fold0-assets")
BUNDLE_PATH = Path(f"/kaggle/input/datasets/{BUNDLE_ID}/bundle.wnn")
ROOT = Path("/root/wnn")
OUT = Path("/kaggle/working/out")
STATUS = OUT / "status.json"
INPUT_IDENTITY = OUT / "input_identity.json"
RUNTIME_SECONDS = 3_600
STEP_KILL_GRACE_SECONDS = 10
ACCELERATOR_REQUEST = "NvidiaTeslaT4"
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_MODEL_TOKEN = "T4"

OUT.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
deadline = started + RUNTIME_SECONDS - STEP_KILL_GRACE_SECONDS
state_lock = threading.RLock()
state: dict[str, object] = {
    "schema": "h382-export-recovery-status-v1",
    "phase": "start",
    "packet_id": PACKET_ID,
    "bundle_id": BUNDLE_ID,
    "bundle_sha256": BUNDLE_SHA256,
    "started_unix": time.time(),
    "jobs": {},
    "resource_samples": [],
    "accelerator_request": ACCELERATOR_REQUEST,
    "expected_gpu_count": EXPECTED_GPU_COUNT,
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(**updates: object) -> None:
    """Persist a complete status snapshot atomically for external monitoring."""
    with state_lock:
        state.update(updates)
        state["updated_unix"] = time.time()
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
        temporary.replace(STATUS)


def save_input_observation(observation: dict[str, object]) -> None:
    """Persist incremental expected and observed input identities before fitting."""
    temporary = INPUT_IDENTITY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(observation, indent=2, sort_keys=True, default=str))
    temporary.replace(INPUT_IDENTITY)


def sample_gpus() -> dict[str, object]:
    """Collect a read-only snapshot of device occupancy and memory."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return {
        "rc": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr[-500:],
    }


def require_expected_gpu_allocation() -> dict[str, object]:
    """Persist and enforce the admitted two-T4 allocation before inputs or fits."""
    inventory = sample_gpus()
    lines = [line.strip() for line in str(inventory["stdout"]).splitlines() if line.strip()]
    save(
        phase="device_gate",
        accelerator_request=ACCELERATOR_REQUEST,
        expected_gpu_count=EXPECTED_GPU_COUNT,
        observed_gpu_count=len(lines),
        gpus=inventory,
    )
    if inventory["rc"] != 0:
        raise RuntimeError(f"H382 R3 cannot inspect requested devices: {inventory['stderr']}")
    if len(lines) != EXPECTED_GPU_COUNT or any(
        EXPECTED_GPU_MODEL_TOKEN not in line for line in lines
    ):
        raise RuntimeError(
            "H382 R3 requires exactly two visible T4 devices before any input setup or fit; "
            f"observed {len(lines)} devices: {lines}"
        )
    return inventory


def extract_bundle(path: Path) -> None:
    """Extract a hash-bound flat file bundle and reject unsafe or duplicate names."""
    if path.stat().st_size != BUNDLE_BYTES or sha256_file(path) != BUNDLE_SHA256:
        raise RuntimeError("mounted H382 bundle differs from the frozen packet")
    ROOT.mkdir(parents=True, exist_ok=True)
    if any(ROOT.iterdir()):
        raise RuntimeError("H382 project root is not empty; refusing to overwrite an existing tree")
    with tarfile.open(path, mode="r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("H382 bundle contains duplicate archive members")
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError(f"unsafe H382 bundle member: {member.name}")
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read H382 bundle member: {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
    manifest = json.loads((ROOT / "h382_source_manifest.json").read_text())
    for item in manifest["files"]:
        source = ROOT / item["path"]
        if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
            raise RuntimeError(f"extracted source mismatch: {item['path']}")


def npy_header(path: Path) -> tuple[tuple[int, ...], str, int]:
    """Read a NumPy array's dtype and shape without loading its payload."""
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a NumPy array: {path}")
        version = stream.read(2)
        if version == b"\x01\x00":
            length = int.from_bytes(stream.read(2), "little")
        elif version in (b"\x02\x00", b"\x03\x00"):
            length = int.from_bytes(stream.read(4), "little")
        else:
            raise ValueError(f"unsupported NumPy format version: {version!r}")
        header_bytes = stream.read(length)
        if len(header_bytes) != length:
            raise ValueError(f"truncated NumPy header: {path}")
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
        data_offset = stream.tell()
    return tuple(header["shape"]), str(header["descr"]), data_offset




def scan_prediction_finiteness(path: Path, chunk_rows: int = 16) -> dict[str, object]:
    """Scan a prediction NPY in bounded mmap row chunks and reject NaN/Inf."""
    import numpy as np

    if chunk_rows < 1:
        raise ValueError("finite-scan chunk size must be positive")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    checked_rows = 0
    try:
        if values.ndim != 3 or values.dtype.kind != "f":
            raise ValueError(f"expected a floating-point 3D prediction array: {path}")
        for first in range(0, values.shape[0], chunk_rows):
            block = values[first : first + chunk_rows]
            checked_rows += int(block.shape[0])
            if not bool(np.isfinite(block).all()):
                return {
                    "finite": False,
                    "checked_rows": checked_rows,
                    "chunk_rows": chunk_rows,
                }
    finally:
        del values
    return {"finite": True, "checked_rows": checked_rows, "chunk_rows": chunk_rows}


def validate_export(job: dict[str, object]) -> dict[str, object]:
    """Bind a full-valid prediction to its exact H382 checkpoint and split."""
    name = str(job["name"])
    epoch = int(job["checkpoint_epoch"])
    prediction = ROOT / "cache" / f"pred_{name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    checkpoint = ROOT / "artifacts" / "runs" / name / f"model_ep{epoch}.pt"
    provenance_path = ROOT / "artifacts" / "runs" / name / "provenance.json"
    if not prediction.is_file() or not sidecar.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"missing H382 export/checkpoint: {name}")
    shape, dtype, payload_offset = npy_header(prediction)
    if shape != (1873, 20_000, 2) or dtype not in ("<f4", "=f4"):
        raise RuntimeError(f"unexpected H382 prediction array: {name} {shape} {dtype}")
    expected_bytes = payload_offset + 1873 * 20_000 * 2 * 4
    if prediction.stat().st_size != expected_bytes:
        raise RuntimeError(f"H382 prediction byte count mismatch: {name}")
    finite_scan = scan_prediction_finiteness(prediction)
    if not finite_scan["finite"]:
        raise RuntimeError(f"H382 prediction contains nonfinite values: {name}")
    meta = json.loads(sidecar.read_text())
    jobs = json.loads((ROOT / "kaggle_jobs.json").read_text())
    expected_split = jobs["split_manifest_sha256"]
    if (
        meta.get("run") != name
        or meta.get("split") != "valid"
        or meta.get("epoch") != epoch
        or meta.get("split_manifest_sha256") != expected_split
        or meta.get("checkpoint_sha256") != sha256_file(checkpoint)
        or meta.get("prediction_sha256") != sha256_file(prediction)
        or meta.get("row_groups") != list(range(1873))
    ):
        raise RuntimeError(f"H382 prediction sidecar mismatch: {name}")
    provenance = json.loads(provenance_path.read_text())
    args = provenance.get("args", {})
    expected_args = {
        "name": name,
        "seed": 3,
        "valid_fold": 0,
        "epochs": int(job["epochs"]),
        "checkpoint_epoch": epoch,
        "batch": 32,
        "chunk": 1000,
        "muon_scope": "all",
        "propensity": "c20260929_prop_f0",
        "prop_power": float(job["prop_power"]),
        "split_manifest_sha256": expected_split,
        "train_source": "train+valid",
        "eval_source": "vfold",
        "loss": "wmse",
        "loss_norm": "chunk",
        "sched": "onecycle",
        "precision": "fp16",
    }
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise RuntimeError(f"H382 checkpoint recipe provenance mismatch: {name}")
    if provenance.get("fit_membership_identity") != jobs["fit_membership"]:
        raise RuntimeError(f"H382 fit membership mismatch: {name}")
    return {
        "run": name,
        "prediction_sha256": meta["prediction_sha256"],
        "prediction_bytes": prediction.stat().st_size,
        "prediction_shape": list(shape),
        "prediction_dtype": dtype,
        "prediction_finite": bool(finite_scan["finite"]),
        "finite_scan_checked_rows": finite_scan["checked_rows"],
        "finite_scan_chunk_rows": finite_scan["chunk_rows"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint.stat().st_size,
        "sidecar_sha256": sha256_file(sidecar),
        "sidecar_bytes": sidecar.stat().st_size,
        "source_packet_id": SOURCE_PACKET_ID,
        "source_bundle_sha256": BUNDLE_SHA256,
        "split_manifest_sha256": expected_split,
        "epoch": epoch,
    }



def stage_rescued_checkpoints(jobs: dict[str, object]) -> dict[str, object]:
    """Verify selected R3 checkpoints and stage them without running fit code."""
    owner = CHECKPOINT_DATASET_ID.split("/", 1)[0]
    candidates = [
        Path("/kaggle/input/datasets") / owner / CHECKPOINT_DATASET_SLUG,
        Path("/kaggle/input") / owner / CHECKPOINT_DATASET_SLUG,
        Path("/kaggle/input") / CHECKPOINT_DATASET_SLUG,
    ]
    mounted = [
        path.resolve()
        for path in candidates
        if (path / "rescued_checkpoints.wnn").is_file()
        and (path / "checkpoint_manifest.json").is_file()
    ]
    if len(set(mounted)) != 1:
        raise RuntimeError(
            "expected exactly one checkpoint dataset mount with both sealed files; "
            f"found {len(set(mounted))} candidates"
        )
    checkpoint_root = mounted[0]
    checkpoint_archive = checkpoint_root / "rescued_checkpoints.wnn"
    checkpoint_manifest = checkpoint_root / "checkpoint_manifest.json"
    save(checkpoint_asset_mount=str(checkpoint_root))
    if checkpoint_archive.stat().st_size != CHECKPOINT_ARCHIVE_BYTES or sha256_file(checkpoint_archive) != CHECKPOINT_ARCHIVE_SHA256:
        raise RuntimeError("rescued checkpoint archive differs from its sealed digest")
    manifest_bytes = checkpoint_manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError("rescued checkpoint manifest differs from its sealed digest")
    manifest = json.loads(manifest_bytes)
    if manifest.get("source_packet_id") != SOURCE_PACKET_ID or manifest.get("source_bundle_sha256") != BUNDLE_SHA256:
        raise RuntimeError("rescued checkpoints do not bind the completed H382 source packet")
    rows = manifest.get("runs")
    if not isinstance(rows, list) or {row.get("name") for row in rows} != {job["name"] for job in jobs["jobs"]}:
        raise RuntimeError("rescued checkpoint run set differs from H382 R3")
    file_rows = manifest.get("files", [])
    expected_files = {item["path"]: item for item in file_rows}
    if len(expected_files) != len(file_rows) or len(expected_files) != 3 * len(rows):
        raise RuntimeError("rescued checkpoint manifest has duplicate or incomplete file rows")
    with tarfile.open(checkpoint_archive, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected_files):
            raise RuntimeError("rescued archive file set differs from its manifest")
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError(f"unsafe rescued checkpoint member: {member.name}")
            entry = expected_files[member.name]
            if member.size != entry["bytes"]:
                raise RuntimeError(f"rescued checkpoint size mismatch: {member.name}")
            destination = ROOT / relative
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite source or output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read rescued checkpoint: {member.name}")
            with stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1 << 20)
            if destination.stat().st_size != entry["bytes"] or sha256_file(destination) != entry["sha256"]:
                raise RuntimeError(f"rescued checkpoint hash mismatch: {member.name}")
    for job in jobs["jobs"]:
        name = str(job["name"])
        row = next(item for item in rows if item["name"] == name)
        epoch = int(job["checkpoint_epoch"])
        if row.get("checkpoint_epoch") != epoch:
            raise RuntimeError(f"rescued checkpoint epoch mismatch: {name}")
        provenance = json.loads((ROOT / row["provenance_path"]).read_text())
        checkpoint = ROOT / row["checkpoint_path"]
        args = provenance.get("args", {})
        expected_args = {
            "name": name,
            "seed": 3,
            "valid_fold": 0,
            "train_source": "train+valid",
            "eval_source": "vfold",
            "epochs": int(job["epochs"]),
            "checkpoint_epoch": epoch,
            "optim": "muon",
            "muon_scope": "all",
            "batch": 32,
            "chunk": 1000,
            "lr": 0.001,
            "wd": 0.01,
            "propensity": "c20260929_prop_f0",
            "prop_power": float(job["prop_power"]),
            "include_heldout_valid": False,
        }
        if any(args.get(key) != value for key, value in expected_args.items()):
            raise RuntimeError(f"rescued checkpoint recipe provenance mismatch: {name}")
        if (
            sha256_file(checkpoint) != row["checkpoint_sha256"]
            or provenance.get("split_manifest_sha256") != jobs["split_manifest_sha256"]
        ):
            raise RuntimeError(f"rescued checkpoint lineage mismatch: {name}")
        if provenance.get("fit_membership_identity") != jobs["fit_membership"]:
            raise RuntimeError(f"rescued fitting membership mismatch: {name}")
    manifest["runs_by_name"] = {row["name"]: row for row in rows}
    return manifest


def validate_collected_prediction_pair(
    run: str,
    prediction: Path,
    sidecar: Path,
    receipt: dict[str, object],
    expected: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    for key, value in expected.items():
        if receipt.get(key) != value:
            errors.append(f"{run}:fresh_receipt_{key}_mismatch")
    if receipt.get("run") != run:
        errors.append(f"{run}:fresh_receipt_run_mismatch")
    files: dict[str, object] = {}
    for label, path, size_key, hash_key in (
        ("npy", prediction, "prediction_bytes", "prediction_sha256"),
        ("sidecar", sidecar, "sidecar_bytes", "sidecar_sha256"),
    ):
        if not path.is_file():
            errors.append(f"{run}:collected_{label}_missing")
            files[label] = {"path": str(path), "present": False}
            continue
        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(path)
        expected_bytes = receipt.get(size_key)
        expected_sha256 = receipt.get(hash_key)
        files[label] = {
            "path": str(path),
            "present": True,
            "expected_bytes": expected_bytes,
            "observed_bytes": observed_bytes,
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
        }
        if not isinstance(expected_bytes, int) or observed_bytes != expected_bytes:
            errors.append(f"{run}:collected_{label}_bytes_mismatch")
        if not isinstance(expected_sha256, str) or observed_sha256 != expected_sha256:
            errors.append(f"{run}:collected_{label}_sha256_mismatch")
    receipt_digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "run": run,
        "valid": not errors,
        "errors": errors,
        "fresh_export_receipt_sha256": receipt_digest,
        "files": files,
    }

def parquet_footer_identity(path: Path, footer_bytes: int) -> dict[str, object]:
    """Hash the Parquet footer with its length prefix and trailing magic bytes."""
    file_size = path.stat().st_size
    if footer_bytes < 0 or footer_bytes > file_size - 8:
        raise RuntimeError(f"invalid registered Parquet footer length: {path}")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        if len(trailer) != 8:
            raise RuntimeError(f"truncated Parquet trailer: {path}")
        observed_footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or observed_footer_bytes != footer_bytes:
            raise RuntimeError(f"Parquet trailer differs from registry: {path}")
        range_bytes = footer_bytes + 8
        start_offset = file_size - range_bytes
        stream.seek(-range_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = range_bytes
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise RuntimeError(f"truncated Parquet footer range: {path}")
            digest.update(block)
            remaining -= len(block)
    return {
        "observed_footer_bytes": observed_footer_bytes,
        "observed_footer_sha256": digest.hexdigest(),
        "footer_range_bytes": range_bytes,
        "footer_start_offset": start_offset,
        "footer_end_offset": file_size,
    }


def link_inputs(jobs: dict[str, object]) -> dict[str, object]:
    """Mount raw and content-addressed fold-0 inputs at the trainer's paths."""
    bundle_manifest = ROOT / "h382_fold0_asset_manifest.json"
    asset_manifest_bytes = bundle_manifest.read_bytes()
    if hashlib.sha256(asset_manifest_bytes).hexdigest() != jobs["asset_manifest_sha256"]:
        raise RuntimeError("bundled fold-0 asset manifest changed")
    mounted_asset_manifest = ASSET_ROOT / "assets-manifest.json"
    if sha256_file(mounted_asset_manifest) != jobs["asset_manifest_sha256"]:
        raise RuntimeError("mounted fold-0 asset manifest differs from sealed receipt")
    registered = {
        item["path"]: item for item in json.loads(mounted_asset_manifest.read_text())["files"]
    }
    observation: dict[str, object] = {
        "schema": "h382-input-identity-v2",
        "packet_id": PACKET_ID,
        "bundle_sha256": BUNDLE_SHA256,
        "asset_manifest_sha256": jobs["asset_manifest_sha256"],
        "split_manifest_sha256": jobs["split_manifest_sha256"],
        "portable_registry_sha256": jobs["portable_registry_sha256"],
        "fit_membership": jobs["fit_membership"],
        "assets": [],
        "raw_files": [],
    }
    save_input_observation(observation)
    observed_assets: list[dict[str, object]] = observation["assets"]  # type: ignore[assignment]
    bundled_paths = set(jobs["bundled_materialized_paths"])
    for expected in jobs["asset_files"]:
        item = registered.get(expected["path"])
        if item != expected:
            raise RuntimeError(f"fold-0 asset receipt mismatch: {expected['path']}")
        source = ASSET_ROOT / expected["filename"]
        record: dict[str, object] = {
            "path": expected["path"],
            "expected_filename": expected["filename"],
            "expected_bytes": expected["size"],
            "expected_sha256": expected["sha256"],
            "observed_bytes": None,
            "observed_sha256": None,
        }
        observed_assets.append(record)
        save_input_observation(observation)
        if not source.is_file():
            raise FileNotFoundError(f"mounted fold-0 asset is missing: {source}")
        record["observed_bytes"] = source.stat().st_size
        if record["observed_bytes"] != expected["size"]:
            save_input_observation(observation)
            raise RuntimeError(f"mounted fold-0 asset has wrong size: {source}")
        if not source.name.startswith(expected["sha256"] + "."):
            save_input_observation(observation)
            raise RuntimeError(f"fold-0 asset name does not bind its SHA: {source}")
        record["observed_sha256"] = sha256_file(source)
        save_input_observation(observation)
        if record["observed_sha256"] != expected["sha256"]:
            raise RuntimeError(f"fold-0 asset content hash mismatch: {source}")
        destination = ROOT / expected["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if expected["path"] in bundled_paths:
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError(f"H382 bundled input is absent or redirected: {destination}")
            if (
                destination.stat().st_size != expected["size"]
                or sha256_file(destination) != expected["sha256"]
            ):
                raise RuntimeError(
                    f"H382 bundled input differs from its mounted asset: {destination}"
                )
            record["materialization"] = "bundle_verified_against_mounted_asset"
        else:
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(f"H382 input path already exists: {destination}")
            destination.symlink_to(source)
            record["materialization"] = "mounted_asset_symlink"
        record["linked_path"] = expected["path"]
    ROOT.joinpath("datasets").mkdir(parents=True, exist_ok=True)
    observed_raw: list[dict[str, object]] = observation["raw_files"]  # type: ignore[assignment]
    registry = json.loads((ROOT / jobs["portable_registry_path"]).read_text())
    for filename in ("train.parquet", "valid.parquet", "valid_mask.parquet"):
        source = WUNDER_ROOT / filename
        raw_record: dict[str, object] = {
            "name": filename,
            "expected_bytes": None,
            "observed_bytes": None,
        }
        observed_raw.append(raw_record)
        save_input_observation(observation)
        if not source.is_file():
            raise FileNotFoundError(f"Kaggle source dataset is missing {source}")
        destination = ROOT / "datasets" / filename
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"H382 raw input path already exists: {destination}")
        registry_entry = registry["sources"].get(f"datasets/{filename}")
        actual_bytes = source.stat().st_size
        raw_record["observed_bytes"] = actual_bytes
        if registry_entry is not None:
            original = registry_entry["original_identity"]
            raw_record["expected_bytes"] = original["bytes"]
            raw_record["expected_sha256"] = registry_entry["sha256"]
            raw_record["expected_footer_sha256"] = original.get("footer_sha256")
            raw_record["footer_bytes"] = original.get("footer_bytes")
            if actual_bytes != original["bytes"]:
                save_input_observation(observation)
                raise RuntimeError(f"Kaggle raw input size differs from registry: {filename}")
            footer_bytes = original.get("footer_bytes")
            footer_digest = original.get("footer_sha256")
            if footer_bytes is not None and footer_digest is not None:
                footer_observation = parquet_footer_identity(source, int(footer_bytes))
                raw_record.update(footer_observation)
                save_input_observation(observation)
                if footer_observation["observed_footer_sha256"] != footer_digest:
                    raise RuntimeError(f"Kaggle raw input footer differs from registry: {filename}")
        save_input_observation(observation)
        destination.symlink_to(source)
    prop_model = ROOT / "artifacts/runs/c20260929_prop_f0/model.pt"
    if sha256_file(prop_model) != jobs["prop_model_sha256"]:
        raise RuntimeError("mounted fold-0 propensity model digest mismatch")
    prop_sidecars = []
    for split, rows, filled in (("train", 10607, 10606), ("valid", 1873, 1873)):
        meta_path = ROOT / "cache" / f"propcache_c20260929_prop_f0_{split}.json"
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("model") != "c20260929_prop_f0"
            or meta.get("model_sha256") != jobs["prop_model_sha256"]
            or meta.get("split") != split
            or meta.get("rows") != rows
            or meta.get("n_filled") != filled
        ):
            raise RuntimeError(f"fold-0 propensity cache provenance mismatch: {split}")
        prop_cache = ROOT / "cache" / f"propcache_c20260929_prop_f0_{split}.npy"
        shape, dtype, offset = npy_header(prop_cache)
        if shape != (rows, 20_000) or dtype not in ("<f4", "=f4"):
            raise RuntimeError(f"fold-0 propensity cache layout mismatch: {split}")
        if prop_cache.stat().st_size != offset + rows * 20_000 * 4:
            raise RuntimeError(f"fold-0 propensity cache byte count mismatch: {split}")
        prop_sidecars.append(
            {"split": split, "rows": rows, "n_filled": filled, "shape": list(shape), "dtype": dtype}
        )
    observation.update(
        {
            "propensity_model_sha256": sha256_file(prop_model),
            "propensity_caches": prop_sidecars,
            "inputs_validated": True,
        }
    )
    save_input_observation(observation)
    return observation


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    """Stop a timed-out trainer and bound its termination grace."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=STEP_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def run_step(job_name: str, step: dict[str, object], env: dict[str, str]) -> tuple[int, float]:
    """Run a step with both per-step and cohort deadlines plus periodic heartbeats."""
    label = str(step["label"])
    log_path = OUT / "logs" / f"{job_name}.{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step_started = time.monotonic()
    step_limit = float(step["timeout"])
    rec = state["jobs"][job_name]  # type: ignore[index]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            step["argv"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )
        while True:
            elapsed = time.monotonic() - step_started
            remain = min(step_limit - elapsed, deadline - time.monotonic())
            if remain <= 0:
                terminate_group(process)
                return 124, time.monotonic() - step_started
            try:
                returncode = process.wait(timeout=min(60.0, remain))
                return returncode, time.monotonic() - step_started
            except subprocess.TimeoutExpired:
                sample = {
                    "unix": time.time(),
                    "job": job_name,
                    "step": label,
                    "gpus": sample_gpus(),
                }
                with state_lock:
                    state["resource_samples"].append(sample)  # type: ignore[union-attr]
                    rec["elapsed_s"] = round(time.monotonic() - step_started, 1)
                    rec["heartbeat_unix"] = sample["unix"]
                    rec["phase"] = label
                    temporary = STATUS.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
                    temporary.replace(STATUS)


def main() -> None:
    """Validate exact inputs, export rescued checkpoints, and collect predictions."""
    save(phase="device_gate", listing=sorted(os.listdir("/kaggle/input")))
    require_expected_gpu_allocation()
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Kaggle bundle dataset is absent: {BUNDLE_PATH}")
    if sha256_file(BUNDLE_PATH) != BUNDLE_SHA256:
        raise RuntimeError("Kaggle bundle digest differs from sealed H382 manifest")
    extract_bundle(BUNDLE_PATH)
    jobs = json.loads((ROOT / "kaggle_jobs.json").read_text())
    if jobs["packet_id"] != SOURCE_PACKET_ID or jobs["bundle_id"] != BUNDLE_ID:
        raise RuntimeError("mounted source bundle identity mismatch")
    source_manifest = ROOT / "h382_source_manifest.json"
    if sha256_file(source_manifest) != jobs["source_manifest_sha256"]:
        raise RuntimeError("H382 source manifest digest differs from the frozen job receipt")
    for entry in jobs["source_bindings"]:
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"H382 bound source differs from packet receipt: {entry['path']}")
    input_identity = link_inputs(jobs)
    if sha256_file(ROOT / jobs["portable_registry_path"]) != jobs["portable_registry_sha256"]:
        raise RuntimeError("fold-0 portable input registry changed")
    split_path = ROOT / jobs["split_manifest_path"]
    if sha256_file(split_path) != jobs["split_manifest_sha256"]:
        raise RuntimeError("fold-0 split manifest changed")
    if not input_identity.get("inputs_validated"):
        raise RuntimeError("fold-0 mounted input identity was not validated")
    checkpoint_manifest = stage_rescued_checkpoints(jobs)
    for job in jobs["jobs"]:
        export_steps = [step for step in job["steps"] if step["label"] == "export"]
        if (
            len(export_steps) != 1
            or len(job["steps"]) != 2
            or {step["label"] for step in job["steps"]} != {"fit", "export"}
        ):
            raise RuntimeError(f"unexpected source job steps: {job['name']}")
        expected_argv = ["python", str(ROOT / "scripts/predict_run.py"), job["name"], "valid", "--epoch", str(job["checkpoint_epoch"])]
        if export_steps[0]["argv"] != expected_argv:
            raise RuntimeError(f"source export argv changed: {job['name']}")
        if any("train_rnn.py" in str(arg) for step in export_steps for arg in step["argv"]):
            raise RuntimeError("export recovery may not execute a fit command")
        job["steps"] = export_steps
        job["reference_export_receipt"] = checkpoint_manifest["runs_by_name"][job["name"]]["reference_export_receipt"]
        if len(job["steps"]) != 1 or job["steps"][0]["label"] != "export":
            raise RuntimeError("fit steps remain in export-only plan")
    input_identity["packet_id"] = PACKET_ID
    input_identity["source_packet_id"] = SOURCE_PACKET_ID
    input_identity["bundle_sha256"] = BUNDLE_SHA256
    input_identity["checkpoint_manifest_sha256"] = CHECKPOINT_MANIFEST_SHA256
    input_identity["checkpoint_archive_sha256"] = CHECKPOINT_ARCHIVE_SHA256
    save_input_observation(input_identity)
    save(
        phase="ready_to_export",
        inputs_verified=True,
        input_identity_sha256=sha256_file(INPUT_IDENTITY),
        gpus=sample_gpus(),
    )
    base_env = os.environ | {
        "WNN_PROJECT_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
        **jobs["env"],
    }
    queue = list(jobs["jobs"])
    queue_lock = threading.Lock()
    worker_errors: list[str] = []

    def worker(slot: dict[str, object]) -> None:
        while time.monotonic() < deadline:
            with queue_lock:
                if not queue:
                    return
                job = queue.pop(0)
            name = str(job["name"])
            with state_lock:
                state["jobs"][name] = {"state": "starting", "gpu": slot["gpu"]}  # type: ignore[index]
            env = base_env | {
                "CUDA_VISIBLE_DEVICES": str(slot["gpu"]),
                "OMP_NUM_THREADS": str(slot["threads"]),
                "MKL_NUM_THREADS": str(slot["threads"]),
                "WNN_THREADS": str(slot["threads"]),
            }
            for step in job["steps"]:
                with state_lock:
                    state["jobs"][name]["phase"] = step["label"]  # type: ignore[index]
                    state["jobs"][name]["step_started_unix"] = time.time()  # type: ignore[index]
                save(phase="exports", active_job=name)
                rc, seconds = run_step(name, step, env)
                with state_lock:
                    rec = state["jobs"][name]  # type: ignore[index]
                    rec[f"{step['label']}_rc"] = rc
                    rec[f"{step['label']}_seconds"] = round(seconds, 1)
                    rec["last_gpus"] = sample_gpus()
                save()
                if rc != 0:
                    with state_lock:
                        state["jobs"][name]["state"] = "failed"  # type: ignore[index]
                    worker_errors.append(f"{name}:{step['label']} rc={rc}")
                    break
                if step["label"] == "export":
                    try:
                        receipt = validate_export(job)
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        with state_lock:
                            state["jobs"][name]["validate_error"] = str(error)  # type: ignore[index]
                            state["jobs"][name]["state"] = "failed"  # type: ignore[index]
                        worker_errors.append(f"{name}:export validation: {error}")
                        break
                    with state_lock:
                        state["jobs"][name]["export_receipt"] = receipt  # type: ignore[index]
                        state["jobs"][name]["state"] = "validated"  # type: ignore[index]

    save(phase="jobs", expected_jobs=[job["name"] for job in jobs["jobs"]], gpus=sample_gpus())
    threads: list[threading.Thread] = []
    for slot in jobs["slots"]:
        thread = threading.Thread(target=worker, args=(slot,), daemon=True)
        thread.start()
        if len(jobs["slots"]) > 1:
            time.sleep(jobs["stagger_seconds"])
        threads.append(thread)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        worker_errors.append("cohort hard wall reached before all worker threads exited")
    save(phase="collect", worker_errors=worker_errors, gpus=sample_gpus())
    runs = OUT / "runs"
    runs.mkdir(exist_ok=True)
    collected: list[str] = []
    for job in jobs["jobs"]:
        name = job["name"]
        source = ROOT / "artifacts/runs" / name
        if source.is_dir() and state["jobs"].get(name, {}).get("state") == "validated":  # type: ignore[union-attr]
            shutil.copytree(
                source,
                runs / name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("training_state.pt"),
            )
            collected.append(name)
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    collected_predictions: list[str] = []
    collection_checks: list[dict[str, object]] = []
    for job in jobs["jobs"]:
        name = str(job["name"])
        copied: dict[str, Path] = {}
        for suffix, label in (("npy", "prediction"), ("json", "sidecar")):
            source_path = ROOT / "cache" / f"pred_{name}_valid.{suffix}"
            destination = prediction_dir / source_path.name
            if source_path.is_file():
                shutil.copy2(source_path, destination)
            copied[label] = destination
        fresh_receipt = state.get("jobs", {}).get(name, {}).get("export_receipt", {})
        expected_binding = {
            "run": name,
            "source_packet_id": SOURCE_PACKET_ID,
            "source_bundle_sha256": BUNDLE_SHA256,
            "split_manifest_sha256": jobs["split_manifest_sha256"],
            "checkpoint_sha256": job["reference_export_receipt"]["checkpoint_sha256"],
            "checkpoint_bytes": job["reference_export_receipt"]["checkpoint_bytes"],
            "epoch": int(job["checkpoint_epoch"]),
            "prediction_shape": [1873, 20_000, 2],
            "prediction_dtype": "<f4",
            "prediction_finite": True,
        }
        check = validate_collected_prediction_pair(
            name,
            copied["prediction"],
            copied["sidecar"],
            fresh_receipt,
            expected_binding,
        )
        collection_checks.append(check)
        if check["valid"]:
            collected_predictions.extend(
                (copied["prediction"].name, copied["sidecar"].name)
            )
        else:
            worker_errors.extend(check["errors"])
    terminal = {
        "schema": "h382-terminal-inventory-v1",
        "expected_jobs": [job["name"] for job in jobs["jobs"]],
        "validated_jobs": collected,
        "complete": len(collected) == len(jobs["jobs"]) and len(collected_predictions) == 2 * len(jobs["jobs"]) and len(collection_checks) == len(jobs["jobs"]) and all(check["valid"] for check in collection_checks) and not worker_errors,
        "runtime_seconds": round(time.monotonic() - started, 1),
        "files": [],
        "collection_checks": collection_checks,
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"terminal_inventory.json", "status.json"}:
            terminal["files"].append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    terminal["reference_prediction_hashes"] = {
        str(job["name"]): {
            "expected": job["reference_export_receipt"]["prediction_sha256"],
            "collected": next(
                (
                    item["sha256"]
                    for item in terminal["files"]
                    if item["path"] == f"predictions/pred_{job['name']}_valid.npy"
                ),
                None,
            ),
            "matches_reference_export": False,
        }
        for job in jobs["jobs"]
    }
    for receipt in terminal["reference_prediction_hashes"].values():
        receipt["matches_reference_export"] = receipt["collected"] == receipt["expected"]
    terminal["complete"] = bool(terminal["complete"]) and all(
        receipt["collected"] is not None
        for receipt in terminal["reference_prediction_hashes"].values()
    ) and len(terminal["collection_checks"]) == len(jobs["jobs"]) and all(
        check["valid"] for check in terminal["collection_checks"]
    )
    (OUT / "terminal_inventory.json").write_text(json.dumps(terminal, indent=2, sort_keys=True))
    save(phase="complete" if terminal["complete"] else "partial_or_failed", terminal=terminal)
    if not terminal["complete"]:
        raise RuntimeError("H382 export recovery ended without all six prediction files")


try:
    main()
except BaseException as error:
    save(phase="failed", error=f"{type(error).__name__}: {error}")
    raise
