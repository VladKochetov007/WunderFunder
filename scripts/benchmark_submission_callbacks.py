#!/usr/bin/env python3
"""Measure one exact submission ZIP's callback cost on fixed complete sequences.

This is a serving-capacity profile, not a scoring or model-selection tool. The
measurement includes the shipped Python callback and native predictor for every
row, including warm-up rows, while excluding parquet decode and state reset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import statistics
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash an in-memory entry or tensor buffer."""
    return hashlib.sha256(value).hexdigest()


def read_cgroup_snapshot() -> dict[str, str | None]:
    """Read the current v2 cgroup counters when available."""
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line.split("::", 1)[1] for line in lines if "::" in line), None)
    result: dict[str, str | None] = {"path": relative}
    if relative is None:
        return result
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for name in (
        "cpu.max",
        "cpu.stat",
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
    ):
        path = directory / name
        result[name] = path.read_text().strip() if path.is_file() else None
    return result


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract only regular safe relative paths from the fixed ZIP."""
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError(f"unsafe ZIP entry path: {info.filename!r}")
        if info.is_dir():
            continue
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))


def cpu_model() -> str | None:
    """Return the first CPU model label exposed by Linux procfs."""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def load_sequences(data: Path, first_group: int, count: int) -> list[dict[str, Any]]:
    """Load a small, fixed list of complete causal row groups."""
    parquet = pq.ParquetFile(data)
    if first_group < 0 or count < 1 or first_group + count > parquet.num_row_groups:
        raise ValueError("requested sequence row-group range is outside the parquet file")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
    output = []
    seen: set[int] = set()
    for group in range(first_group, first_group + count):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id, need = validate_sequence(table)
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence ID {sequence_id}")
        seen.add(sequence_id)
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
            np.float32, copy=False
        )
        if features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
            raise ValueError(f"unexpected feature shape {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite raw input in sequence {sequence_id}")
        output.append(
            {
                "row_group": group,
                "sequence_id": sequence_id,
                "need_prediction_rows": int(need.sum()),
                "features_sha256": sha256_bytes(features.tobytes(order="C")),
                "features": features,
                "need": need,
            }
        )
    return output


def run_callbacks(
    model: Any,
    sequence: dict[str, Any],
) -> tuple[int, np.ndarray]:
    """Reset state and time exactly one complete callback sequence."""
    model.model.reset()
    sequence_id = sequence["sequence_id"]
    features = sequence["features"]
    need = sequence["need"]
    predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    elapsed_ns = 0
    for step in range(SEQUENCE_LENGTH):
        point = DataPoint(sequence_id, step, bool(need[step]), features[step])
        start = time.perf_counter_ns()
        value = model.predict(point)
        elapsed_ns += time.perf_counter_ns() - start
        if not need[step]:
            if value is not None:
                raise ValueError("callback must update state and return None during warm-up")
            continue
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError("callback must return two finite scores on every required row")
        predictions[step] = result
    return elapsed_ns, predictions


def main() -> None:
    """Run repeated reset-aware timing and publish immutable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-row-group", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()

    submission = args.submission.resolve(strict=True)
    receipt_path = args.receipt.resolve(strict=True)
    data = args.data.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace profile receipt: {output}")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required to verify reset determinism")

    package_hash = sha256_file(submission)
    receipt_hash = sha256_file(receipt_path)
    build = json.loads(receipt_path.read_text())
    if build["artifact"]["sha256"] != package_hash:
        raise ValueError("build receipt ZIP hash does not match the measured ZIP")
    if build["precision"] != "q8vnni":
        raise ValueError("this benchmark is scoped to shipped Q8VNNI packages")
    if build["gate_status"] != "package_crc_passed; offline_import_and_streaming_gate_pending":
        raise ValueError("unexpected package receipt gate status")

    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not allowed_cpus:
        raise RuntimeError("process has no allowed CPU affinity")
    selected_cpu = args.cpu if args.cpu is not None else allowed_cpus[0]
    if selected_cpu not in allowed_cpus:
        raise ValueError(f"CPU {selected_cpu} is outside allowed affinity {allowed_cpus}")
    os.sched_setaffinity(0, {selected_cpu})

    data_hash_before = sha256_file(data)
    sequences = load_sequences(data, args.first_row_group, args.sequences)
    start_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="wnn-fixed-native-benchmark-") as temporary:
        package_root = Path(temporary)
        with zipfile.ZipFile(submission) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            zip_members = {name: sha256_bytes(archive.read(name)) for name in archive.namelist()}
            expected_members = build["artifact"]["members"]
            if zip_members != expected_members:
                raise ValueError("ZIP entries differ from the hash-bound build receipt")
            safe_extract(archive, package_root)

        solution_path = package_root / "solution.py"
        if sha256_file(solution_path) != build["source_code"]["sha256"]:
            raise ValueError("root solution.py does not match build receipt")
        spec = importlib.util.spec_from_file_location(
            f"wnn_profile_solution_{package_hash[:12]}", solution_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot import solution.py from the measured ZIP")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PredictionModel()
        predictor = model.model
        if predictor.precision != "q8vnni":
            raise ValueError(f"native runtime selected {predictor.precision}, expected q8vnni")
        cgroup_before_measure = read_cgroup_snapshot()
        loadavg_before_measure = Path("/proc/loadavg").read_text().strip()

        members = json.loads((package_root / "members.json").read_text())
        if len(members) != len(build["fixed_members"]):
            raise ValueError("members.json count differs from the build receipt")

        expected_predictions: dict[int, np.ndarray] = {}
        trials = []
        per_sequence_ns: dict[int, list[int]] = {item["sequence_id"]: [] for item in sequences}
        for repeat in range(args.repeats):
            repeat_ns = 0
            for sequence in sequences:
                elapsed_ns, predictions = run_callbacks(model, sequence)
                sequence_id = sequence["sequence_id"]
                if repeat == 0:
                    expected_predictions[sequence_id] = predictions
                elif not np.array_equal(
                    predictions, expected_predictions[sequence_id], equal_nan=True
                ):
                    raise ValueError(
                        f"reset replay changed predictions for sequence {sequence_id}, repeat {repeat}"
                    )
                per_sequence_ns[sequence_id].append(elapsed_ns)
                repeat_ns += elapsed_ns
            trials.append(
                {
                    "repeat": repeat,
                    "rows": args.sequences * SEQUENCE_LENGTH,
                    "callback_ns": repeat_ns,
                    "callback_us_per_row": repeat_ns / (args.sequences * SEQUENCE_LENGTH) / 1000,
                }
            )

        data_hash_after = sha256_file(data)
        if data_hash_after != data_hash_before:
            raise ValueError("benchmark data changed during measurement")
        cgroup_after_measure = read_cgroup_snapshot()
        elapsed_total = time.perf_counter() - start_total
        rows_per_repeat = args.sequences * SEQUENCE_LENGTH
        all_callback_ns = sum(trial["callback_ns"] for trial in trials)
        result = {
            "schema": "wnn.native-callback-profile.v1",
            "scope": (
                "Runtime-capacity profile only. The fixed package is replayed over complete causal "
                "sequences; no metric is computed and no local/public score is claimed. Each sequence "
                "is reset before every pass, and predictions must be bitwise identical across resets."
            ),
            "candidate": build["candidate"],
            "submission": str(submission),
            "submission_sha256": package_hash,
            "submission_bytes": submission.stat().st_size,
            "build_receipt": str(receipt_path),
            "build_receipt_sha256": receipt_hash,
            "build_members": build["fixed_members"],
            "native_runtime": build["native_runtime"],
            "benchmark_runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "utils_source": {
                "path": str(ROOT / "utils.py"),
                "sha256": sha256_file(ROOT / "utils.py"),
            },
            "checkpoint_sources": [
                {
                    "run_reference": member["run_reference"],
                    "checkpoint_path": source["checkpoint_path"],
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "provenance_path": source["provenance"]["path"],
                    "provenance_sha256": source["provenance"]["sha256"],
                }
                for member in build["members"]
                for source in member["sources"]
                if "checkpoint_sha256" in source
            ],
            "runtime": {
                "precision": predictor.precision,
                "native_level": module._LEVEL,
                "model_count": len(members),
                "cpu_model": cpu_model(),
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "affinity_before": allowed_cpus,
                "affinity_used": [selected_cpu],
                "cgroup_before_measure": cgroup_before_measure,
                "cgroup_after_measure": cgroup_after_measure,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "loadavg_start": loadavg_before_measure,
                "loadavg_end": Path("/proc/loadavg").read_text().strip(),
            },
            "workload": {
                "data": str(data),
                "data_sha256": data_hash_before,
                "row_groups": [item["row_group"] for item in sequences],
                "sequence_ids": [item["sequence_id"] for item in sequences],
                "sequence_lengths": [SEQUENCE_LENGTH] * args.sequences,
                "need_prediction_rows_each": [item["need_prediction_rows"] for item in sequences],
                "raw_feature_columns": len(FEATURE_COLUMNS),
                "features_sha256_by_sequence": {
                    str(item["sequence_id"]): item["features_sha256"] for item in sequences
                },
                "repeats": args.repeats,
                "reset_before_each_sequence_and_pass": True,
                "predictions_bitwise_identical_across_replays": True,
            },
            "timing": {
                "rows_per_repeat": rows_per_repeat,
                "total_callback_calls": rows_per_repeat * args.repeats,
                "elapsed_callback_ns": all_callback_ns,
                "mean_callback_us_per_row": all_callback_ns
                / (rows_per_repeat * args.repeats)
                / 1000,
                "median_repeat_us_per_row": statistics.median(
                    trial["callback_us_per_row"] for trial in trials
                ),
                "per_sequence_mean_us_per_row": {
                    str(sequence_id): statistics.mean(times) / SEQUENCE_LENGTH / 1000
                    for sequence_id, times in per_sequence_ns.items()
                },
                "trials": trials,
                "elapsed_wall_seconds_after_sequence_decode": elapsed_total,
                "measurement": "perf_counter_ns around each exact model.predict(DataPoint) callback",
                "reset_and_parquet_decode_excluded": True,
            },
        }

    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
