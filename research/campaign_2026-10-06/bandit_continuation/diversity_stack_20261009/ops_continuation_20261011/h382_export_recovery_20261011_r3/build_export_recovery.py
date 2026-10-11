"""Build a sealed H382 export-only recovery from the rescued R3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # Nested producer receipts have schema-specific JSON records.

HERE = Path(__file__).resolve().parent
MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
COLLECTED_ROOT = (
    MAIN_ROOT / "research/campaign_2026-10-06/bandit_continuation/ops_continuation_20261011/"
    "kaggle_h382_delivery_20261011_r3/provider_output/out"
)
R3_KERNEL = (
    MAIN_ROOT / "research/campaign_2026-10-06/bandit_continuation/ops_continuation_20261011/"
    "kaggle_h382_delivery_20261011_r3/remote_readback/"
    "wnn-kaggle-crossfold-h382-20261011-r3.py"
)
SOURCE_JOBS = HERE / "source/kaggle_jobs_r3.json"
SOURCE_JOBS_SHA256 = "0fda9c0fe9d8370a6c7d48afb6fa1009834689c4aed2400ed2301495bbfa45cd"
SOURCE_MANIFEST_SHA256 = "a98f035691dbd4f1d9e69ccbffc37d12fcbd0ec979a30d649c7919d1e0e9b979"
R3_PACKET_MANIFEST = (
    MAIN_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/kaggle_crossfold_H382_r3_20261011/"
    "H382_PACKET_R3.json"
)
R3_PACKET_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3"
R3_BUNDLE_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r3"
CHECKPOINT_DATASET_ID = "vladkochetov07/wnn-h382-rescued-checkpoints-20261011-r3"
KERNEL_ID = "vladkochetov07/wnn-h382-export-recovery-20261011-r3"
KERNEL_TITLE = "WNN H382 Export Recovery 20261011 R3"
R3_KERNEL_SHA256 = "b7f959750dfef32ec8ee43675dc05a859e0d99ed1d1c5468c6542f5bc13c12ed"
R3_PACKET_SHA256 = "2f32067be4b80c9a235756d2ced1d6084b1029fb2c64ca847266d68d512b1e65"
EXPECTED_RUNS = (
    ("kg_h382r3_f0_muonall_prop2_e3_s3", 2, 3, 2.0),
    ("kg_h382r3_f0_muonall_prop0_e3_s3", 2, 3, 0.0),
    ("kg_h382r3_f0_muonall_prop0_e4_s3", 3, 4, 0.0),
)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    """Encode deterministic review and provider metadata."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify R3 output and return its job, status, and identity receipts."""
    jobs = json.loads(SOURCE_JOBS.read_text())
    packet = json.loads(R3_PACKET_MANIFEST.read_text())
    status = json.loads((COLLECTED_ROOT / "status.json").read_text())
    identity = json.loads((COLLECTED_ROOT / "input_identity.json").read_text())
    inventory = json.loads((COLLECTED_ROOT / "terminal_inventory.json").read_text())
    if sha256_file(SOURCE_JOBS) != SOURCE_JOBS_SHA256:
        raise ValueError("extracted R3 job receipt changed from the source bundle")
    if sha256_file(HERE / "source/h382_source_manifest_r3.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("extracted R3 source manifest changed from the source bundle")
    if jobs["packet_id"] != R3_PACKET_ID or jobs["bundle_id"] != R3_BUNDLE_ID:
        raise ValueError("R3 jobs file differs from the reviewed source packet")
    if (
        packet.get("packet_id") != R3_PACKET_ID
        or packet.get("bundle_id") != R3_BUNDLE_ID
        or packet.get("bundle_sha256") != identity.get("bundle_sha256")
        or packet.get("kernel_sha256") != R3_KERNEL_SHA256
        or sha256_file(R3_PACKET_MANIFEST) != R3_PACKET_SHA256
    ):
        raise ValueError("R3 manifest, source bundle, and terminal input identity disagree")
    jobs["bundle_sha256"] = packet["bundle_sha256"]
    if status.get("packet_id") != R3_PACKET_ID or status.get("phase") != "complete":
        raise ValueError("rescued R3 worker status is not terminal-complete")
    if status.get("worker_errors"):
        raise ValueError("rescued R3 status contains worker errors")
    if not inventory.get("complete"):
        raise ValueError("rescued R3 terminal inventory is incomplete")
    expected_names = {row[0] for row in EXPECTED_RUNS}
    if set(status.get("jobs", {})) != expected_names:
        raise ValueError("R3 status run names differ from the frozen three-job cohort")
    if set(inventory.get("validated_jobs", [])) != expected_names:
        raise ValueError("R3 inventory does not validate each expected checkpoint run")
    if identity.get("packet_id") != R3_PACKET_ID or not identity.get("inputs_validated"):
        raise ValueError("rescued R3 input identity is not validated")
    if identity.get("bundle_sha256") != jobs["bundle_sha256"]:
        raise ValueError("rescued R3 input identity binds a different source bundle")
    if identity.get("split_manifest_sha256") != jobs["split_manifest_sha256"]:
        raise ValueError("rescued R3 split identity differs from the source packet")
    raw_files = {item.get("name"): item for item in identity.get("raw_files", [])}
    if set(raw_files) != {"train.parquet", "valid.parquet", "valid_mask.parquet"}:
        raise ValueError("rescued R3 raw-data file inventory differs from the frozen inputs")
    for filename in ("train.parquet", "valid.parquet"):
        item = raw_files[filename]
        if (
            not item.get("expected_footer_sha256")
            or item.get("expected_footer_sha256") != item.get("observed_footer_sha256")
            or item.get("expected_bytes") != item.get("observed_bytes")
        ):
            raise ValueError(f"rescued R3 raw-data footer identity is incomplete: {filename}")
    if not raw_files["valid_mask.parquet"].get("observed_bytes"):
        raise ValueError("rescued R3 valid mask is absent or empty")
    if any(
        item.get("expected_sha256") != item.get("observed_sha256")
        or item.get("expected_bytes") != item.get("observed_bytes")
        for item in identity.get("assets", [])
    ):
        raise ValueError("rescued R3 asset identity is incomplete")
    if len(identity.get("assets", [])) != 11:
        raise ValueError("rescued R3 fold-0 asset count differs from the reviewed packet")
    expected_inventory_files = {
        item["path"]: (item["bytes"], item["sha256"]) for item in inventory["files"]
    }
    for relative, (expected_bytes, expected_sha256) in expected_inventory_files.items():
        path = COLLECTED_ROOT / relative
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
            raise ValueError(f"rescued R3 output differs from terminal inventory: {relative}")
    if sha256_file(COLLECTED_ROOT / "input_identity.json") != status["input_identity_sha256"]:
        raise ValueError("R3 status does not bind the collected input identity")
    if len(jobs["jobs"]) != len(EXPECTED_RUNS):
        raise ValueError("R3 source job count changed")
    for job, (name, epoch, epochs, power) in zip(jobs["jobs"], EXPECTED_RUNS, strict=True):
        if (
            job["name"] != name
            or job["checkpoint_epoch"] != epoch
            or job["epochs"] != epochs
            or job["prop_power"] != power
        ):
            raise ValueError(f"R3 source recipe changed for {name}")
        run = COLLECTED_ROOT / "runs" / name
        provenance = json.loads((run / "provenance.json").read_text())
        args = provenance["args"]
        expected_args = {
            "name": name,
            "seed": 3,
            "valid_fold": 0,
            "train_source": "train+valid",
            "eval_source": "vfold",
            "epochs": epochs,
            "checkpoint_epoch": epoch,
            "optim": "muon",
            "muon_scope": "all",
            "batch": 32,
            "chunk": 1000,
            "lr": 0.001,
            "wd": 0.01,
            "propensity": "c20260929_prop_f0",
            "prop_power": power,
            "include_heldout_valid": False,
        }
        if any(args.get(key) != value for key, value in expected_args.items()):
            raise ValueError(f"R3 provenance recipe mismatch for {name}")
        if provenance.get("split_manifest_sha256") != identity["split_manifest_sha256"]:
            raise ValueError(f"R3 split provenance mismatch for {name}")
        if provenance.get("fit_membership_identity") != jobs["fit_membership"]:
            raise ValueError(f"R3 fitting membership mismatch for {name}")
        if status["jobs"][name].get("state") != "validated":
            raise ValueError(f"R3 export was not validated for {name}")
        if status["jobs"][name].get("fit_rc") != 0 or status["jobs"][name].get("export_rc") != 0:
            raise ValueError(f"R3 fit or export process failed for {name}")
        receipt = status["jobs"][name].get("export_receipt", {})
        if (
            receipt.get("run") != name
            or receipt.get("epoch") != epoch
            or receipt.get("prediction_shape") != [1873, 20_000, 2]
            or receipt.get("prediction_dtype") not in ("<f4", "=f4")
            or receipt.get("prediction_bytes") != 299_680_128
            or receipt.get("checkpoint_sha256") != sha256_file(run / f"model_ep{epoch}.pt")
        ):
            raise ValueError(f"R3 export receipt binding failed for {name}")
    return jobs, status, identity


def create_checkpoint_assets(
    jobs: dict[str, Any], status: dict[str, Any], identity: dict[str, Any], stage: Path
) -> tuple[dict[str, Any], Path]:
    """Create a minimal private archive containing only selected CPs and lineage."""
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(stage, 0o700)
    archive_path = stage / "rescued_checkpoints.wnn"
    if archive_path.exists():
        raise FileExistsError(f"refusing to replace checkpoint asset: {archive_path}")
    files: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    contents: list[tuple[str, Path]] = []
    for _job, (name, epoch, epochs, power) in zip(jobs["jobs"], EXPECTED_RUNS, strict=True):
        run = COLLECTED_ROOT / "runs" / name
        selected = [
            f"artifacts/runs/{name}/model_ep{epoch}.pt",
            f"artifacts/runs/{name}/provenance.json",
            f"artifacts/runs/{name}/history.json",
        ]
        for relative in selected:
            source = run / Path(relative).name
            item = {"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            files.append(item)
            contents.append((relative, source))
        export = status["jobs"][name]["export_receipt"]
        runs.append(
            {
                "name": name,
                "checkpoint_epoch": epoch,
                "epochs": epochs,
                "prop_power": power,
                "checkpoint_path": selected[0],
                "checkpoint_sha256": files[-3]["sha256"],
                "provenance_path": selected[1],
                "provenance_sha256": files[-2]["sha256"],
                "history_path": selected[2],
                "history_sha256": files[-1]["sha256"],
                "reference_export_receipt": export,
            }
        )
    with tarfile.open(archive_path, "w") as archive:
        for relative, source in contents:
            payload = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o600
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            archive.addfile(info, __import__("io").BytesIO(payload))
    os.chmod(archive_path, 0o600)
    manifest = {
        "schema": "wnn.h382-rescued-checkpoint-assets.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_packet_id": R3_PACKET_ID,
        "source_bundle_id": R3_BUNDLE_ID,
        "source_bundle_sha256": jobs["bundle_sha256"],
        "source_manifest_sha256": jobs["source_manifest_sha256"],
        "split_manifest_sha256": identity["split_manifest_sha256"],
        "input_identity_sha256": sha256_file(COLLECTED_ROOT / "input_identity.json"),
        "terminal_inventory_sha256": sha256_file(COLLECTED_ROOT / "terminal_inventory.json"),
        "status_sha256": sha256_file(COLLECTED_ROOT / "status.json"),
        "checkpoint_archive": {
            "filename": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "runs": runs,
        "files": files,
        "all_data_prediction_scope": {
            "shape": [1873, 20_000, 2],
            "dtype": "float32",
            "note": "Full-valid arrays include fold-0 training and evaluation groups; score only through the frozen fold-0 membership contract.",
        },
        "admission": "packet preparation only; no dataset upload or kernel push",
    }
    manifest_path = HERE / "checkpoint_asset_manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    os.chmod(manifest_path, 0o600)
    (stage / "checkpoint_manifest.json").write_bytes(manifest_path.read_bytes())
    os.chmod(stage / "checkpoint_manifest.json", 0o600)
    return manifest, archive_path


def patch_kernel(manifest: dict[str, Any]) -> bytes:
    """Derive an export-only worker from the byte-bound R3 consumer."""
    source = R3_KERNEL.read_text()
    if sha256_file(R3_KERNEL) != R3_KERNEL_SHA256:
        raise ValueError("R3 kernel consumer source changed after its remote readback")
    replacements = [
        (
            '"""Kaggle worker template for the bounded H382 fold-0 cohort."""',
            '"""Recover validated H382 predictions from rescued fold-0 checkpoints."""',
        ),
        (
            f'PACKET_ID = "{R3_PACKET_ID}"',
            f'SOURCE_PACKET_ID = "{R3_PACKET_ID}"\nPACKET_ID = "{KERNEL_ID}"',
        ),
        (
            'BUNDLE_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r3"',
            f'BUNDLE_ID = "{R3_BUNDLE_ID}"\n'
            f'CHECKPOINT_DATASET_ID = "{CHECKPOINT_DATASET_ID}"\n'
            'CHECKPOINT_DATASET_SLUG = CHECKPOINT_DATASET_ID.split("/", 1)[1]\n'
            f"CHECKPOINT_ARCHIVE_BYTES = {manifest['checkpoint_archive']['bytes']}\n"
            f'CHECKPOINT_ARCHIVE_SHA256 = "{manifest["checkpoint_archive"]["sha256"]}"\n'
            f'CHECKPOINT_MANIFEST_SHA256 = "{sha256_file(HERE / "checkpoint_asset_manifest.json")}"',
        ),
        ("RUNTIME_SECONDS = 14_400", "RUNTIME_SECONDS = 3_600"),
        ('"schema": "h382-kaggle-status-v2"', '"schema": "h382-export-recovery-status-v1"'),
        (
            '"""Validate exact inputs, execute three bounded fits, and collect full outputs."""',
            '"""Validate exact inputs, export rescued checkpoints, and collect predictions."""',
        ),
        (
            '    if jobs["packet_id"] != PACKET_ID or jobs["bundle_id"] != BUNDLE_ID:\n'
            '        raise RuntimeError("mounted H382 bundle identity mismatch")',
            '    if jobs["packet_id"] != SOURCE_PACKET_ID or jobs["bundle_id"] != BUNDLE_ID:\n'
            '        raise RuntimeError("mounted source bundle identity mismatch")',
        ),
        (
            '    if sha256_file(split_path) != jobs["split_manifest_sha256"]:\n'
            '        raise RuntimeError("fold-0 split manifest changed")\n',
            '    if sha256_file(split_path) != jobs["split_manifest_sha256"]:\n'
            '        raise RuntimeError("fold-0 split manifest changed")\n'
            '    if not input_identity.get("inputs_validated"):\n'
            '        raise RuntimeError("fold-0 mounted input identity was not validated")\n'
            "    checkpoint_manifest = stage_rescued_checkpoints(jobs)\n"
            '    for job in jobs["jobs"]:\n'
            '        export_steps = [step for step in job["steps"] if step["label"] == "export"]\n'
            "        if (\n"
            "            len(export_steps) != 1\n"
            '            or len(job["steps"]) != 2\n'
            '            or {step["label"] for step in job["steps"]} != {"fit", "export"}\n'
            "        ):\n"
            "            raise RuntimeError(f\"unexpected source job steps: {job['name']}\")\n"
            '        expected_argv = ["python", str(ROOT / "scripts/predict_run.py"), job["name"], "valid", "--epoch", str(job["checkpoint_epoch"])]\n'
            '        if export_steps[0]["argv"] != expected_argv:\n'
            "            raise RuntimeError(f\"source export argv changed: {job['name']}\")\n"
            '        if any("train_rnn.py" in str(arg) for step in export_steps for arg in step["argv"]):\n'
            '            raise RuntimeError("export recovery may not execute a fit command")\n'
            '        job["steps"] = export_steps\n'
            '        job["reference_export_receipt"] = checkpoint_manifest["runs_by_name"][job["name"]]["reference_export_receipt"]\n'
            '        if len(job["steps"]) != 1 or job["steps"][0]["label"] != "export":\n'
            '            raise RuntimeError("fit steps remain in export-only plan")\n',
        ),
        (
            '    input_identity["packet_id"] = PACKET_ID\n'
            '    input_identity["bundle_sha256"] = BUNDLE_SHA256\n',
            '    input_identity["packet_id"] = PACKET_ID\n'
            '    input_identity["source_packet_id"] = SOURCE_PACKET_ID\n'
            '    input_identity["bundle_sha256"] = BUNDLE_SHA256\n'
            '    input_identity["checkpoint_manifest_sha256"] = CHECKPOINT_MANIFEST_SHA256\n'
            '    input_identity["checkpoint_archive_sha256"] = CHECKPOINT_ARCHIVE_SHA256\n',
        ),
        ('phase="ready_to_fit"', 'phase="ready_to_export"'),
        ('save(phase="fits", active_job=name)', 'save(phase="exports", active_job=name)'),
        (
            '    for path in sorted((ROOT / "cache").glob("pred_kg_h382_*_valid.*")):\n'
            "        shutil.copy2(path, OUT / path.name)\n",
            '    prediction_dir = OUT / "predictions"\n'
            "    prediction_dir.mkdir(exist_ok=True)\n"
            "    collected_predictions: list[str] = []\n"
            '    for job in jobs["jobs"]:\n'
            '        name = str(job["name"])\n'
            '        for suffix in ("npy", "json"):\n'
            '            path = ROOT / "cache" / f"pred_{name}_valid.{suffix}"\n'
            "            if not path.is_file():\n"
            '                worker_errors.append(f"{name}:missing validated prediction {suffix}")\n'
            "                continue\n"
            "            shutil.copy2(path, prediction_dir / path.name)\n"
            "            collected_predictions.append(path.name)\n",
        ),
        (
            '"complete": len(collected) == len(jobs["jobs"]) and not worker_errors,',
            '"complete": len(collected) == len(jobs["jobs"]) and len(collected_predictions) == 2 * len(jobs["jobs"]) and not worker_errors,',
        ),
        (
            '    save(phase="complete" if terminal["complete"] else "partial_or_failed", terminal=terminal)\n'
            '    if not terminal["complete"]:\n'
            '        raise RuntimeError("H382 cohort ended without three validated predictions")',
            '    terminal["reference_prediction_hashes"] = {\n'
            '        str(job["name"]): {\n'
            '            "expected": job["reference_export_receipt"]["prediction_sha256"],\n'
            '            "collected": next((item["sha256"] for item in terminal["files"] if item["path"].endswith(f"pred_{job[\'name\']}_valid.npy")), None),\n'
            "        }\n"
            '        for job in jobs["jobs"]\n'
            "    }\n"
            '    save(phase="complete" if terminal["complete"] else "partial_or_failed", terminal=terminal)\n'
            '    if not terminal["complete"]:\n'
            '        raise RuntimeError("H382 export recovery ended without all six prediction files")',
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"R3 worker patch anchor is not unique: {old[:70]!r}")
        source = source.replace(old, new, 1)
    finite_helper = '''
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


'''
    validation_anchor = "\ndef validate_export(job: dict[str, object]) -> dict[str, object]:"
    if source.count(validation_anchor) != 1:
        raise ValueError("R3 prediction validator insertion anchor is not unique")
    source = source.replace(validation_anchor, "\n\n" + finite_helper + "def validate_export(job: dict[str, object]) -> dict[str, object]:", 1)
    header_check = (
        '    if prediction.stat().st_size != expected_bytes:\n'
        '        raise RuntimeError(f"H382 prediction byte count mismatch: {name}")\n'
    )
    finite_check = (
        header_check
        + '    finite_scan = scan_prediction_finiteness(prediction)\n'
        + '    if not finite_scan["finite"]:\n'
        + '        raise RuntimeError(f"H382 prediction contains nonfinite values: {name}")\n'
    )
    if source.count(header_check) != 1:
        raise ValueError("R3 prediction size validation anchor is not unique")
    source = source.replace(header_check, finite_check, 1)
    receipt_anchor = (
        '        "prediction_dtype": dtype,\n'
        '        "checkpoint_sha256": meta["checkpoint_sha256"],\n'
    )
    receipt = (
        '        "prediction_dtype": dtype,\n'
        '        "prediction_finite": bool(finite_scan["finite"]),\n'
        '        "finite_scan_checked_rows": finite_scan["checked_rows"],\n'
        '        "finite_scan_chunk_rows": finite_scan["chunk_rows"],\n'
        '        "checkpoint_sha256": meta["checkpoint_sha256"],\n'
    )
    if source.count(receipt_anchor) != 1:
        raise ValueError("R3 export receipt insertion anchor is not unique")
    source = source.replace(receipt_anchor, receipt, 1)
    inventory_anchor = '    (OUT / "terminal_inventory.json").write_text(json.dumps(terminal, indent=2, sort_keys=True))\n'
    inventory_end = '    save(phase="complete" if terminal["complete"] else "partial_or_failed", terminal=terminal)\n'
    start = source.find(inventory_anchor)
    end = source.find(inventory_end, start)
    if start < 0 or end < 0 or source.find(inventory_anchor, start + 1) >= 0:
        raise ValueError("R1 terminal inventory serialization patch anchors are not unique")
    terminal_receipt_block = '''    terminal["reference_prediction_hashes"] = {
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
    )
    (OUT / "terminal_inventory.json").write_text(json.dumps(terminal, indent=2, sort_keys=True))
'''
    source = source[:start] + terminal_receipt_block + source[end:]
    helper = '''\n\ndef stage_rescued_checkpoints(jobs: dict[str, object]) -> dict[str, object]:
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
'''
    anchor = "\ndef parquet_footer_identity(path: Path, footer_bytes: int) -> dict[str, object]:"
    if source.count(anchor) != 1:
        raise ValueError("R3 input-validation helper insertion anchor is not unique")
    source = source.replace(anchor, helper + anchor, 1)
    return source.encode()


def build_packet(stage: Path) -> dict[str, Any]:
    """Build the identified local packet and private upload asset directory."""
    jobs, status, identity = load_sources()
    asset_manifest, archive = create_checkpoint_assets(jobs, status, identity, stage)
    asset_manifest_sha = sha256_file(HERE / "checkpoint_asset_manifest.json")
    kernel = patch_kernel(asset_manifest)
    kernel_path = HERE / "kernel/kernel.py"
    kernel_path.write_bytes(kernel)
    kernel_meta = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            R3_BUNDLE_ID,
            CHECKPOINT_DATASET_ID,
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": KERNEL_ID,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": KERNEL_TITLE,
    }
    (HERE / "kernel/kernel-metadata.json").write_bytes(json_bytes(kernel_meta))
    dataset_meta = {
        "id": CHECKPOINT_DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
        "title": "WNN H382 Rescued Checkpoints 20261011 R3",
    }
    (stage / "dataset-metadata.json").write_bytes(json_bytes(dataset_meta))
    os.chmod(stage / "dataset-metadata.json", 0o600)
    packet = {
        "schema": "wnn.h382-export-recovery-packet.v3",
        "packet_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "source_packet_id": R3_PACKET_ID,
        "source_bundle_id": R3_BUNDLE_ID,
        "source_bundle_sha256": jobs["bundle_sha256"],
        "source_kernel_sha256": R3_KERNEL_SHA256,
        "checkpoint_manifest_sha256": asset_manifest_sha,
        "checkpoint_archive_sha256": asset_manifest["checkpoint_archive"]["sha256"],
        "checkpoint_archive_bytes": asset_manifest["checkpoint_archive"]["bytes"],
        "prediction_artifact_count": 6,
        "prediction_shape": [1873, 20_000, 2],
        "expected_gpu_model": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "runtime_limit_seconds": 3600,
        "fit_commands": 0,
        "export_commands": 3,
        "source_export_invocation": "python /root/wnn/scripts/predict_run.py RUN valid --epoch E",
        "source_prediction_hashes": {
            row["name"]: row["reference_export_receipt"]["prediction_sha256"]
            for row in asset_manifest["runs"]
        },
        "recovery_reason": "R3 validated each export, but its final cache collection glob pred_kg_h382_* did not match actual run names pred_kg_h382r3_*; R1 also omitted finite-value scans and serialized terminal hashes before attaching the reference hashes.",
        "r2_corrections": [
            "scan each exported float32 prediction in mmap row chunks and require all values finite",
            "persist prediction_finite=true and scanned-row count in the per-run export receipt",
            "attach expected/collected/match reference hashes before serializing terminal_inventory.json",
            "require all three fresh collected NPY hashes to be present before terminal complete; reference-hash equality is diagnostic only",
        ],
        "r3_correction": "Reference R3 prediction digests are recorded and compared diagnostically; cross-device output hash equality is not a terminal success gate. Fresh checkpoint, sidecar, hash, shape, finiteness, split, source and complete-collection validations remain required.",
        "reference_export_seconds": {
            "kg_h382r3_f0_muonall_prop2_e3_s3": 160.5,
            "kg_h382r3_f0_muonall_prop0_e3_s3": 82.2,
            "kg_h382r3_f0_muonall_prop0_e4_s3": 160.6,
            "measured_worker_setup_seconds": None,
            "timing_note": "Observed export commands only; no separate worker startup/setup duration was recorded. Runtime bound is 3600 seconds, with 1800-second per-export limits inherited from the frozen job recipe.",
        },
        "quota_admission": "fresh Kaggle quota/session and two-device gate required before any upload or push",
        "admission": "packet prepared only; no provider dataset upload or kernel push",
    }
    file_entries = {}
    for path in sorted(HERE.rglob("*")):
        if (
            path.is_file()
            and path.name != "PACKET_MANIFEST_R3.json"
            and "__pycache__" not in path.parts
        ):
            relative = path.relative_to(HERE).as_posix()
            file_entries[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    packet["packet_files"] = file_entries
    packet["asset_stage"] = {
        "dataset_metadata_sha256": sha256_file(stage / "dataset-metadata.json"),
        "checkpoint_manifest_sha256": sha256_file(stage / "checkpoint_manifest.json"),
        "checkpoint_archive_sha256": sha256_file(archive),
        "directory_mode": oct(stage.stat().st_mode & 0o777),
    }
    (HERE / "PACKET_MANIFEST_R3.json").write_bytes(json_bytes(packet))
    return packet


def main() -> None:
    """Build the sealed export-only packet without uploading or dispatching."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-stage", type=Path, default=HERE / "asset_stage")
    args = parser.parse_args()
    packet = build_packet(args.asset_stage.resolve())
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "packet_manifest_sha256": sha256_file(HERE / "PACKET_MANIFEST_R3.json"),
                "kernel_sha256": packet["packet_files"]["kernel/kernel.py"]["sha256"],
                "checkpoint_archive_sha256": packet["checkpoint_archive_sha256"],
                "checkpoint_archive_bytes": packet["checkpoint_archive_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
