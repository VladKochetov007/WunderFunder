"""Validate exact submission ZIP callbacks on fixed complete sequence groups.

This is a serving and determinism check only. It reads no labels and does not
produce a held-out score; callers must label any data exposure separately.
"""

from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint, validate_sequence  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array using its contiguous byte representation."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def cgroup_snapshot() -> dict[str, str | None]:
    """Capture bounded process resource counters when cgroup v2 is available."""
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line.split("::", 1)[1] for line in membership if "::" in line), None)
    result: dict[str, str | None] = {"path": relative}
    if relative is None:
        return result
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for name in ("cpu.max", "cpu.stat", "memory.current", "memory.peak", "memory.events"):
        file = directory / name
        result[name] = file.read_text().strip() if file.is_file() else None
    return result


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract regular ZIP members after rejecting unsafe paths."""
    for item in archive.infolist():
        name = PurePosixPath(item.filename)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError(f"unsafe ZIP entry path: {item.filename!r}")
        if item.is_dir():
            continue
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))


def load_panel(data: Path, first_group: int, sequence_count: int) -> list[dict[str, Any]]:
    """Load fixed complete causal groups and bind their feature contents."""
    parquet = pq.ParquetFile(data)
    if (
        first_group < 0
        or sequence_count < 1
        or first_group + sequence_count > parquet.num_row_groups
    ):
        raise ValueError("requested row-group range is outside the parquet file")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
    panel: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in range(first_group, first_group + sequence_count):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id, need = validate_sequence(table)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {group} is not a complete sequence")
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence ID {sequence_id}")
        seen.add(sequence_id)
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
            np.float32, copy=False
        )
        if features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
            raise ValueError(f"unexpected feature shape {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite input in sequence {sequence_id}")
        panel.append(
            {
                "row_group": group,
                "sequence_id": sequence_id,
                "features": features,
                "need": need,
                "features_sha256": sha256_array(features),
                "need_sha256": sha256_array(need),
            }
        )
    return panel


def predict_sequence(model: Any, sequence: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Replay every row callback and return predictions plus callback time."""
    features = sequence["features"]
    need = sequence["need"]
    predictions = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    elapsed_ns = 0
    sequence_id = sequence["sequence_id"]
    for step in range(SEQUENCE_LENGTH):
        point = DataPoint(sequence_id, step, bool(need[step]), features[step])
        start = time.perf_counter_ns()
        result = model.predict(point)
        elapsed_ns += time.perf_counter_ns() - start
        if not need[step]:
            if result is not None:
                raise ValueError(f"unexpected prediction during warmup at row {step}")
            continue
        value = np.asarray(result, dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"invalid prediction at row {step} in sequence {sequence_id}")
        predictions[step] = value
    return predictions, elapsed_ns


def main() -> None:
    """Check exact ZIP callbacks, deterministic resets, and measured callback cost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--first-row-group", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), default="v3")
    args = parser.parse_args()

    submission = args.zip.resolve(strict=True)
    receipt_path = args.receipt.resolve(strict=True)
    data = args.data.resolve(strict=True)
    output = args.output.resolve()
    prediction_path = args.predictions.resolve()
    if output.exists() or prediction_path.exists():
        raise FileExistsError("refusing to replace callback evidence")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required to verify reset determinism")

    package_hash = sha256_file(submission)
    build_hash = sha256_file(receipt_path)
    build = json.loads(receipt_path.read_text())
    if build["artifact"]["sha256"] != package_hash:
        raise ValueError("build receipt ZIP hash does not match the measured package")
    if build["precision"] != "q8vnni":
        raise ValueError("callback panel is limited to Q8VNNI packages")
    if build["gate_status"] != "package_crc_passed; offline_import_and_streaming_gate_pending":
        raise ValueError("unexpected immutable build receipt status")

    data_stat = data.stat()
    data_hash = sha256_file(data)
    panel = load_panel(data, args.first_row_group, args.sequences)
    cgroup_before = cgroup_snapshot()
    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    callback_ns_by_repeat: list[int] = []

    with tempfile.TemporaryDirectory(prefix="wnn-submission-panel-") as temporary:
        package = Path(temporary)
        with zipfile.ZipFile(submission) as archive:
            if archive.testzip() is not None:
                raise ValueError("submission ZIP CRC check failed")
            member_hashes = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist())
            }
            if member_hashes != build["artifact"]["members"]:
                raise ValueError("ZIP entry hashes differ from the build receipt")
            safe_extract(archive, package)

        solution = package / "solution.py"
        if sha256_file(solution) != build["source_code"]["sha256"]:
            raise ValueError("shipped solution source differs from the build receipt")
        os.environ["WNN_NATIVE_LEVEL"] = args.level
        spec = importlib.util.spec_from_file_location("checked_submission_solution", solution)
        if spec is None or spec.loader is None:
            raise ImportError("submission ZIP has no loadable solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PredictionModel()
        if model.model.precision != build["precision"]:
            raise RuntimeError(
                "runtime precision differs from the precision bound in the build receipt: "
                f"{model.model.precision!r} != {build['precision']!r}"
            )
        native_path = Path(module.wnn_native.__file__).resolve()
        if native_path.parent != package / args.level:
            raise RuntimeError(f"unexpected native import path: {native_path}")
        native_member = f"{args.level}/wnn_native.abi3.so"
        native_hash = sha256_file(native_path)

        reference: list[np.ndarray] | None = None
        for repeat in range(args.repeats):
            repeat_predictions: list[np.ndarray] = []
            repeat_ns = 0
            for sequence in panel:
                predictions, elapsed_ns = predict_sequence(model, sequence)
                repeat_predictions.append(predictions)
                repeat_ns += elapsed_ns
            if reference is None:
                reference = repeat_predictions
                outputs = repeat_predictions
            else:
                for sequence, actual, expected in zip(
                    panel, repeat_predictions, reference, strict=True
                ):
                    if not np.array_equal(actual, expected):
                        raise ValueError(
                            f"sequence {sequence['sequence_id']} changed across resets"
                        )
            callback_ns_by_repeat.append(repeat_ns)
            reports.append(
                {
                    "repeat": repeat,
                    "callbacks": args.sequences * SEQUENCE_LENGTH,
                    "callback_ns": repeat_ns,
                    "callback_us_per_row": repeat_ns / (args.sequences * SEQUENCE_LENGTH) / 1000,
                }
            )

    assert reference is not None
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, np.stack(outputs, axis=0), allow_pickle=False)
    if sha256_file(data) != data_hash:
        raise ValueError("input data changed during callback panel")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "wnn.submission-callback-panel.v1",
        "passed": True,
        "zip_path": str(submission),
        "zip_sha256": package_hash,
        "build_receipt_path": str(receipt_path),
        "build_receipt_sha256": build_hash,
        "data_path": str(data),
        "data_bytes": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "data_sha256": data_hash,
        "panel": [
            {
                "row_group": item["row_group"],
                "sequence_id": item["sequence_id"],
                "rows": SEQUENCE_LENGTH,
                "features_sha256": item["features_sha256"],
                "need_sha256": item["need_sha256"],
            }
            for item in panel
        ],
        "native_level": args.level,
        "native_member": native_member,
        "native_sha256": native_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_before": cgroup_before,
        "callback_repeats": reports,
        "reset_replay_bitwise_equal": True,
        "predictions_path": str(prediction_path),
        "predictions_shape": list(np.stack(outputs, axis=0).shape),
        "predictions_sha256": sha256_file(prediction_path),
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "cgroup_after": cgroup_snapshot(),
        "interpretation": "serving callback/reset/finite-output panel only; reads no labels and does not establish held-out predictive performance",
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
