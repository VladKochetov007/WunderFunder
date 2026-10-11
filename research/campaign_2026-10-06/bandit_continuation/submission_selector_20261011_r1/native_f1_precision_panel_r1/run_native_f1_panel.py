"""Run one exact Q8 package on the frozen feature-only F1 component panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
EXPECTED_NATIVE_SHA256 = "ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e"


class CallbackModel(Protocol):
    """Small structural contract for a packaged competition callback."""

    def predict(self, data_point: object) -> np.ndarray | None:
        """Advance causal state and return two values when prediction is required."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash the contiguous C-order bytes of one array."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def cgroup_snapshot() -> dict[str, str | None]:
    """Capture CPU and memory counters for this command's cgroup."""
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line.split("::", 1)[1] for line in membership if "::" in line), None)
    result: dict[str, str | None] = {"path": relative}
    if relative is None:
        return result
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for name in ("cpu.max", "cpu.stat", "memory.current", "memory.peak", "memory.events"):
        path = directory / name
        result[name] = path.read_text().strip() if path.is_file() else None
    return result


def validate_zip_names(names: list[str]) -> None:
    """Reject duplicate and unsafe archive paths before extraction."""
    if len(names) != len(set(names)):
        raise ValueError("candidate ZIP contains duplicate names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe candidate ZIP path: {name!r}")


def extract_package(source: Path, destination: Path) -> dict[str, str]:
    """Extract regular files and return the exact ZIP member digests."""
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        validate_zip_names(names)
        if archive.testzip() is not None:
            raise ValueError("candidate ZIP has a CRC error")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if mode and (mode & 0o170000) not in (0, 0o100000):
                raise ValueError(f"non-regular candidate ZIP member: {info.filename!r}")
            payload = archive.read(info.filename)
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            hashes[info.filename] = hashlib.sha256(payload).hexdigest()
    return hashes


def read_feature_panel(
    panel_path: Path, manifest_path: Path, project_root: Path
) -> tuple[object, list[dict[str, object]], dict[str, object]]:
    """Validate the frozen feature fixture without loading its row groups."""
    sys.path.insert(0, str(project_root.resolve(strict=True)))
    from utils import SEQUENCE_LENGTH as PROJECT_SEQUENCE_LENGTH  # noqa: PLC0415

    if PROJECT_SEQUENCE_LENGTH != SEQUENCE_LENGTH:
        raise ValueError("project sequence length differs from the fixed panel contract")
    manifest = json.loads(manifest_path.read_text())
    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict):
        raise ValueError("feature-panel receipt has no fixture record")
    expected_fixture = Path(str(fixture.get("path", ""))).resolve(strict=True)
    if expected_fixture != panel_path.resolve(strict=True):
        raise ValueError("feature-panel path differs from the frozen receipt")
    if sha256_file(panel_path) != fixture.get("sha256"):
        raise ValueError("feature fixture digest differs from the frozen receipt")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("seed") != 572911:
        raise ValueError("feature receipt is not bound to the root-owned selection seed")
    if len(fixture.get("group_records", [])) != 41:
        raise ValueError("feature receipt does not contain the exact 41-group panel")

    parquet = pq.ParquetFile(panel_path)
    if parquet.metadata.num_row_groups != 41:
        raise ValueError("feature fixture has the wrong row-group count")
    expected_records = fixture["group_records"]
    if (
        not isinstance(expected_records, list)
        or len(expected_records) != parquet.metadata.num_row_groups
    ):
        raise ValueError("feature manifest and parquet row-group counts differ")
    if set(parquet.schema_arrow.names) & {"t0", "t1", "is_scored"}:
        raise ValueError("feature fixture unexpectedly contains labels or scoring mask")
    return parquet, expected_records, manifest


def load_runtime(
    package_path: Path, package_receipt_path: Path, level: str, package_dir: Path
) -> tuple[CallbackModel, dict[str, object], dict[str, str]]:
    """Load the exact package and assert its Q8VNNI runtime and source payloads."""
    receipt = json.loads(package_receipt_path.read_text())
    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("candidate receipt has no artifact record")
    if sha256_file(package_path) != artifact.get("sha256"):
        raise ValueError("candidate ZIP digest differs from its build receipt")
    if receipt.get("precision") != "q8vnni":
        raise ValueError("candidate receipt does not require Q8VNNI")

    payload_hashes = extract_package(package_path, package_dir)
    if payload_hashes != artifact.get("members"):
        raise ValueError("candidate ZIP member hashes differ from its receipt")
    native_relative = f"{level}/wnn_native.abi3.so"
    native_path = package_dir / native_relative
    if not native_path.is_file() or sha256_file(native_path) != EXPECTED_NATIVE_SHA256:
        raise ValueError("candidate package has the wrong pinned native library")
    source_receipt = receipt.get("source_code")
    if isinstance(source_receipt, dict) and payload_hashes.get("solution.py") != source_receipt.get(
        "sha256"
    ):
        raise ValueError("packaged solution source differs from its source binding")
    os.environ["WNN_NATIVE_LEVEL"] = level
    spec = importlib.util.spec_from_file_location(
        f"wnn_native_f1_{receipt.get('candidate')}", package_dir / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("candidate package has no importable solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    native_model = getattr(model, "model", None)
    if native_model is None or getattr(native_model, "precision", None) != "q8vnni":
        raise ValueError("actual loaded candidate precision is not q8vnni")
    if getattr(module.wnn_native, "__file__", "") != str(native_path):
        raise ValueError("runtime imported a native library outside the exact package")
    members = json.loads((package_dir / "members.json").read_text())
    if getattr(native_model, "active_members", None) != len(members):
        raise ValueError("actual active-member count differs from the fixed package")
    return (
        model,
        receipt,
        {
            "native_library_path_in_package": native_relative,
            "native_library_sha256": payload_hashes[native_relative],
            "member_count": str(len(members)),
        },
    )


def run_panel(
    package_path: Path,
    package_receipt_path: Path,
    panel_path: Path,
    panel_manifest_path: Path,
    project_root: Path,
    predictions_path: Path,
    receipt_path: Path,
    level: str,
) -> dict[str, object]:
    """Run one Q8 ensemble on the fixed F1 feature-only panel."""
    for path in (package_path, package_receipt_path, panel_path, panel_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if predictions_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite prior inference outputs")
    sys.path.insert(0, str(project_root.resolve(strict=True)))
    from utils import FEATURE_COLUMNS, DataPoint, validate_sequence  # noqa: PLC0415

    parquet, expected_records, feature_manifest = read_feature_panel(
        panel_path, panel_manifest_path, project_root
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = cgroup_snapshot()
    output = np.lib.format.open_memmap(
        predictions_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(expected_records), SEQUENCE_LENGTH, 2),
    )
    group_reports: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="wnn-native-f1-package-") as temporary:
        model, package_receipt, runtime = load_runtime(
            package_path, package_receipt_path, level, Path(temporary)
        )
        try:
            columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
            for panel_index, expected in enumerate(expected_records):
                table = parquet.read_row_group(panel_index, columns=columns, use_threads=False)
                sequence_id, need = validate_sequence(table)
                if sequence_id != expected["sequence_id"] or table.num_rows != SEQUENCE_LENGTH:
                    raise ValueError(
                        f"feature fixture sequence mismatch at panel position {panel_index}"
                    )
                features = np.column_stack(
                    [table[name].to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
                ).astype(np.float32, copy=False)
                if not np.isfinite(features).all():
                    raise ValueError(f"non-finite features at panel position {panel_index}")
                feature_hash = sha256_array(features)
                need_hash = sha256_array(need)
                if feature_hash != expected["features_sha256"]:
                    raise ValueError(f"feature digest mismatch at panel position {panel_index}")
                if need_hash != expected["need_prediction_sha256"]:
                    raise ValueError(f"need-mask digest mismatch at panel position {panel_index}")
                output[panel_index].fill(0.0)
                elapsed_ns = 0
                for step_index in range(SEQUENCE_LENGTH):
                    point = DataPoint(
                        sequence_id,
                        step_index,
                        bool(need[step_index]),
                        features[step_index],
                    )
                    callback_start = time.perf_counter_ns()
                    value = model.predict(point)
                    elapsed_ns += time.perf_counter_ns() - callback_start
                    if not need[step_index]:
                        if value is not None:
                            raise ValueError(
                                f"candidate returned a warmup prediction at sequence {sequence_id}"
                            )
                        continue
                    prediction = np.asarray(value, dtype=np.float32)
                    if prediction.shape != (2,) or not np.isfinite(prediction).all():
                        raise ValueError(
                            f"candidate returned an invalid prediction at {sequence_id}:{step_index}"
                        )
                    output[panel_index, step_index] = prediction
                group_reports.append(
                    {
                        "panel_index": panel_index,
                        "row_group": expected["row_group"],
                        "sequence_id": sequence_id,
                        "callbacks": SEQUENCE_LENGTH,
                        "callback_ns": elapsed_ns,
                        "callback_us_per_row": elapsed_ns / SEQUENCE_LENGTH / 1000,
                        "features_sha256": feature_hash,
                        "need_prediction_sha256": need_hash,
                    }
                )
                del table, features, need
            output.flush()
        finally:
            del output
            del model
            gc.collect()

    after = cgroup_snapshot()
    predictions_sha = sha256_file(predictions_path)
    prediction_array = np.load(predictions_path, mmap_mode="r", allow_pickle=False)
    if prediction_array.shape != (len(expected_records), SEQUENCE_LENGTH, 2):
        raise ValueError("prediction panel has an unexpected shape")
    if prediction_array.dtype != np.float32:
        raise ValueError("prediction panel must use float32 outputs")
    if not np.isfinite(prediction_array).all():
        raise ValueError("prediction panel contains non-finite outputs")
    receipt: dict[str, object] = {
        "schema": "wnn.native_f1_precision_predictions.r1",
        "claim_scope": (
            "Fixed native Q8 inference on an exposed fold1 component panel. This receipt does "
            "not claim a held-out score or fresh confirmation."
        ),
        "candidate": package_receipt.get("candidate"),
        "package": {
            "path": str(package_path.resolve()),
            "sha256": sha256_file(package_path),
            "build_receipt_path": str(package_receipt_path.resolve()),
            "build_receipt_sha256": sha256_file(package_receipt_path),
            "precision_asserted": "q8vnni",
            "native_runtime": runtime,
        },
        "panel": {
            "path": str(panel_path.resolve()),
            "sha256": sha256_file(panel_path),
            "manifest_path": str(panel_manifest_path.resolve()),
            "manifest_sha256": sha256_file(panel_manifest_path),
            "selection_seed": feature_manifest["selection"]["seed"],
            "row_groups": [row["row_group"] for row in group_reports],
            "sequence_ids": [row["sequence_id"] for row in group_reports],
            "feature_only": True,
            "labels_read": False,
            "group_records": group_reports,
        },
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": predictions_sha,
            "shape": list(prediction_array.shape),
            "dtype": prediction_array.dtype.str,
            "bytes": predictions_path.stat().st_size,
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "cgroup_before": before,
            "cgroup_after": after,
        },
    }
    with receipt_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Run one exact Q8 package over all 41 frozen complete sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), default="v3")
    args = parser.parse_args()
    result = run_panel(
        args.package,
        args.package_receipt,
        args.panel,
        args.panel_manifest,
        args.project_root,
        args.predictions,
        args.receipt,
        args.level,
    )
    print(
        json.dumps(
            {"candidate": result["candidate"], "predictions": result["predictions"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
