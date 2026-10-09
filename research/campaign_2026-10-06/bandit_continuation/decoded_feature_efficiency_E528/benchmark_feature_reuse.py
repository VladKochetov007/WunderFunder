#!/usr/bin/env python3
"""Profile exact float32 feature reuse against the project Parquet reader."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, SequenceFile

SCRIPT = Path(__file__).resolve()
LINEAGE = SCRIPT.parent
ROOT = SCRIPT.parents[4]
PRE_REG_PATH = LINEAGE / "E528_PREREGISTRATION_V1.json"
RECEIPT_PATH = LINEAGE / "E528_FROZEN_INPUT_RECEIPT_V1.json"
PROGRESS_PATH = LINEAGE / "E528_PROGRESS_V1.json"
RESULT_PATH = LINEAGE / "E528_RESULTS_V1.json"
CACHE_PATH = LINEAGE / "valid_subset20_features_f32.npy"
MANIFEST_PATH = LINEAGE / "valid_subset20_features_f32.manifest.json"

GROUPS_PATH = ROOT / "cache/valid_subset20_groups.npy"
INDEX_PATH = ROOT / "cache/valid_subset20_idx.npy"
DATA_PATH = ROOT / "cache/valid_subset20.parquet"
READER_PATH = ROOT / "src/wnn/data.py"

EXPECTED_GROUPS = [0, 4, 8, 10, 11, 2, 6, 23, 27, 28, 1, 7, 9, 13, 17, 3, 5, 14, 18, 22]
EXPECTED_SEQ_IX = [
    256678,
    1990332,
    8348957,
    9579986,
    9831922,
    1200028,
    4490992,
    28417254,
    29941155,
    30273600,
    751005,
    6746551,
    9147005,
    16979262,
    23433965,
    1892371,
    2308749,
    19569293,
    24654613,
    28399078,
]
EXPECTED_EPOCHS = [1, 2, 4, 8]
REPETITIONS = 2
SEED = 20261007
DISK_BUDGET_BYTES = 250_000_000
RSS_CAP_BYTES = 1_000_000_000
AS_CAP_BYTES = 1_500_000_000
WALL_CAP_SECONDS = 180
EXPECTED_PYTHON = (3, 11)
EXPECTED_NUMPY = "2.2.6"
EXPECTED_PYARROW = "19.0.1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def feature_columns_sha256(columns: list[str]) -> str:
    payload = json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(document, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def file_size(path: Path) -> int:
    return path.stat().st_size


def process_limits() -> dict[str, Any]:
    def get_limit(name: int) -> list[int | str]:
        soft, hard = resource.getrlimit(name)
        return ["infinity" if soft == resource.RLIM_INFINITY else int(soft),
                "infinity" if hard == resource.RLIM_INFINITY else int(hard)]

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_affinity": affinity,
        "rlimit_as_bytes": get_limit(resource.RLIMIT_AS),
        "rlimit_rss_bytes": get_limit(resource.RLIMIT_RSS),
        "rlimit_cpu_seconds": get_limit(resource.RLIMIT_CPU),
    }


def max_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB.
    return int(usage * 1024)


def usage_counters() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "peak_rss_bytes": max_rss_bytes(),
    }


def load_groups() -> tuple[list[int], list[int]]:
    groups = np.load(GROUPS_PATH, allow_pickle=False)
    indexes = np.load(INDEX_PATH, allow_pickle=False)
    if groups.ndim != 1 or indexes.ndim != 1:
        raise ValueError("group and index arrays must be one-dimensional")
    group_order = [int(value) for value in groups.tolist()]
    index_order = [int(value) for value in indexes.tolist()]
    if group_order != EXPECTED_GROUPS or index_order != EXPECTED_GROUPS:
        raise ValueError(f"saved cohort order changed: groups={group_order}, indexes={index_order}")
    if len(group_order) != 20 or len(set(group_order)) != 20:
        raise ValueError("expected 20 unique source row-group indexes")
    return group_order, index_order


def inspect_sequences(group_order: list[int]) -> tuple[list[int], list[int]]:
    parquet = pq.ParquetFile(DATA_PATH)
    if parquet.num_row_groups != len(group_order):
        raise ValueError(f"expected {len(group_order)} row groups, got {parquet.num_row_groups}")
    names = parquet.schema_arrow.names
    missing = [name for name in ["seq_ix", *FEATURE_COLUMNS] if name not in names]
    if missing:
        raise ValueError(f"required source columns missing: {missing[:5]}")
    if len(FEATURE_COLUMNS) != N_FEATURES or len(FEATURE_COLUMNS) != 112:
        raise ValueError(f"feature count changed: {len(FEATURE_COLUMNS)}")
    if len(set(FEATURE_COLUMNS)) != N_FEATURES:
        raise ValueError("feature column order contains duplicates")
    rows_per_group = [int(parquet.metadata.row_group(i).num_rows) for i in range(parquet.num_row_groups)]
    if rows_per_group != [SEQUENCE_LENGTH] * len(group_order):
        raise ValueError(f"row-group sizes changed: {rows_per_group}")

    seq_ids: list[int] = []
    subset_positions = list(range(len(group_order)))
    for local_position in subset_positions:
        table = parquet.read_row_group(local_position, columns=["seq_ix"], use_threads=False)
        values = table.column("seq_ix").to_numpy(zero_copy_only=False)
        if values.size != SEQUENCE_LENGTH or not np.all(values == values[0]):
            raise ValueError(f"row group {local_position} does not contain one complete seq_ix")
        seq_ids.append(int(values[0]))
    if seq_ids != EXPECTED_SEQ_IX or len(set(seq_ids)) != len(seq_ids):
        raise ValueError(f"subset row order or seq_ix changed: {seq_ids}")
    return rows_per_group, seq_ids


def runtime_environment() -> dict[str, Any]:
    environment = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(f"expected Python 3.11 environment, got {environment['python']}")
    if np.__version__ != EXPECTED_NUMPY or pyarrow.__version__ != EXPECTED_PYARROW:
        raise RuntimeError(f"profile environment drifted: {environment}")
    if sys.byteorder != "little":
        raise RuntimeError("the frozen cache format requires a little-endian host")
    return environment


def preflight() -> None:
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"frozen input receipt already exists: {RECEIPT_PATH}")
    if not PRE_REG_PATH.is_file():
        raise FileNotFoundError(PRE_REG_PATH)
    contract = json.loads(PRE_REG_PATH.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "E528" or contract.get("status") != "frozen_before_timing":
        raise ValueError("E528 preregistration is not in its frozen state")

    environment = runtime_environment()
    group_order, index_order = load_groups()
    rows_per_group, seq_ids = inspect_sequences(group_order)
    expected_payload_bytes = len(group_order) * SEQUENCE_LENGTH * N_FEATURES * np.dtype(np.float32).itemsize
    if expected_payload_bytes != 179_200_000:
        raise ValueError(f"unexpected cache payload bytes: {expected_payload_bytes}")
    free_bytes = shutil.disk_usage(LINEAGE).free
    cache_header_allowance = 64 * 1024
    manifest_allowance = 128 * 1024
    expected_cache_disk_bytes = expected_payload_bytes + cache_header_allowance + manifest_allowance
    if expected_cache_disk_bytes > DISK_BUDGET_BYTES:
        raise ValueError(f"expected cache output exceeds disk cap: {expected_cache_disk_bytes}")
    if free_bytes < expected_cache_disk_bytes:
        raise OSError(f"insufficient disk headroom: {free_bytes} free, need {expected_cache_disk_bytes}")
    if CACHE_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("E528 cache output already exists; refusing to overwrite")
    if RESULT_PATH.exists():
        raise FileExistsError("E528 result already exists; refusing to rerun a completed benchmark")

    columns = list(FEATURE_COLUMNS)
    input_hashes = {
        "parquet_sha256": sha256_file(DATA_PATH),
        "groups_npy_sha256": sha256_file(GROUPS_PATH),
        "index_npy_sha256": sha256_file(INDEX_PATH),
        "reader_source_sha256": sha256_file(READER_PATH),
        "preregistration_sha256": sha256_file(PRE_REG_PATH),
        "benchmark_source_sha256": sha256_file(SCRIPT),
    }
    receipt = {
        "experiment_id": "E528",
        "status": "frozen_before_timing",
        "created_at_utc": utc_now(),
        "preregistration_path": str(PRE_REG_PATH.relative_to(ROOT)),
        "preregistration_sha256": input_hashes["preregistration_sha256"],
        "benchmark_source_path": str(SCRIPT.relative_to(ROOT)),
        "benchmark_source_sha256": input_hashes["benchmark_source_sha256"],
        "reader_source_path": str(READER_PATH.relative_to(ROOT)),
        "input_paths": {
            "parquet": str(DATA_PATH.relative_to(ROOT)),
            "groups": str(GROUPS_PATH.relative_to(ROOT)),
            "indexes": str(INDEX_PATH.relative_to(ROOT)),
        },
        "input_sizes_bytes": {
            "parquet": file_size(DATA_PATH),
            "groups": file_size(GROUPS_PATH),
            "indexes": file_size(INDEX_PATH),
        },
        "input_sha256": input_hashes,
        "feature_columns": columns,
        "feature_columns_sha256": feature_columns_sha256(columns),
        "shape": [len(group_order), SEQUENCE_LENGTH, N_FEATURES],
        "dtype": "float32",
        "layout": "C-contiguous sequence/time/feature",
        "payload_bytes": expected_payload_bytes,
        "disk_free_before_cache_bytes": free_bytes,
        "source_row_group_order": group_order,
        "subset_parquet_row_group_positions": list(range(len(group_order))),
        "source_seq_ix_order": seq_ids,
        "row_group_rows": rows_per_group,
        "cross_checked_group_order": index_order,
        "reader_call": "SequenceFile(DATA_PATH).read(subset_local_row_group_position, targets=False, mask=False)",
        "reader_selection_order": "subset local positions 0..19 correspond in order to source_row_group_order above; each read selects seq_ix followed by the exact ordered FEATURE_COLUMNS list above",
        "environment": environment,
        "resource_limits_before_timing": process_limits(),
        "cache_budget_bytes": DISK_BUDGET_BYTES,
        "no_feature_timing_collected": True,
    }
    atomic_json(RECEIPT_PATH, receipt)
    print(json.dumps({"phase": "preflight_frozen", "receipt": str(RECEIPT_PATH), "input_sha256": input_hashes}, indent=2))


def verify_frozen_inputs() -> tuple[dict[str, Any], list[int], list[int], float]:
    identity_check_start = time.perf_counter()
    if not RECEIPT_PATH.is_file():
        raise FileNotFoundError(f"run --freeze first: {RECEIPT_PATH}")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt.get("status") != "frozen_before_timing":
        raise ValueError("frozen input receipt is not frozen")
    if receipt.get("preregistration_sha256") != sha256_file(PRE_REG_PATH):
        raise RuntimeError("preregistration changed after freeze")
    if receipt.get("benchmark_source_sha256") != sha256_file(SCRIPT):
        raise RuntimeError("benchmark helper changed after freeze")
    if receipt.get("reader_source_sha256") != sha256_file(READER_PATH):
        raise RuntimeError("training reader source changed after freeze")
    current_feature_columns = list(FEATURE_COLUMNS)
    if receipt.get("feature_columns") != current_feature_columns:
        raise RuntimeError("ordered FEATURE_COLUMNS changed after freeze")
    if receipt.get("feature_columns_sha256") != feature_columns_sha256(current_feature_columns):
        raise RuntimeError("feature-column digest mismatch")
    for key, path in (
        ("parquet_sha256", DATA_PATH),
        ("groups_npy_sha256", GROUPS_PATH),
        ("index_npy_sha256", INDEX_PATH),
    ):
        if receipt["input_sha256"][key] != sha256_file(path):
            raise RuntimeError(f"frozen input changed after preflight: {path}")
    group_order, index_order = load_groups()
    if group_order != receipt["source_row_group_order"] or index_order != receipt["cross_checked_group_order"]:
        raise RuntimeError("saved subset group order changed after freeze")
    runtime_environment()
    if CACHE_PATH.exists() or MANIFEST_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("refusing to overwrite an existing E528 cache or result")
    limits = process_limits()
    affinity = limits["cpu_affinity"]
    if affinity is not None and len(affinity) != 1:
        raise RuntimeError(f"expected one-CPU taskset affinity, got {affinity}")
    if limits["rlimit_as_bytes"][0] != AS_CAP_BYTES:
        raise RuntimeError(f"address-space limit is not {AS_CAP_BYTES}: {limits['rlimit_as_bytes']}")
    if limits["rlimit_rss_bytes"][0] != RSS_CAP_BYTES:
        raise RuntimeError(f"RSS limit is not {RSS_CAP_BYTES}: {limits['rlimit_rss_bytes']}")
    if limits["rlimit_cpu_seconds"][0] != WALL_CAP_SECONDS:
        raise RuntimeError(f"CPU-time safety limit is not {WALL_CAP_SECONDS}s: {limits['rlimit_cpu_seconds']}")
    identity_check_seconds = time.perf_counter() - identity_check_start
    return receipt, group_order, list(receipt["source_seq_ix_order"]), identity_check_seconds


def write_progress(phase: str, *, details: dict[str, Any] | None = None, measurements: list[dict[str, Any]] | None = None) -> None:
    progress = {
        "experiment_id": "E528",
        "updated_at_utc": utc_now(),
        "phase": phase,
        "details": details or {},
        "completed_measurements": measurements or [],
        "resource_usage": usage_counters(),
    }
    atomic_json(PROGRESS_PATH, progress)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def build_cache(group_order: list[int], seq_ids: list[int]) -> dict[str, Any]:
    expected_shape = (len(group_order), SEQUENCE_LENGTH, N_FEATURES)
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if expected_bytes > DISK_BUDGET_BYTES:
        raise ValueError(f"cache payload exceeds disk budget: {expected_bytes}")
    if CACHE_PATH.exists():
        raise FileExistsError(CACHE_PATH)

    source_payload_hash = hashlib.sha256()
    build_start = time.perf_counter()
    reader = SequenceFile(DATA_PATH)
    output = np.lib.format.open_memmap(
        CACHE_PATH,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=expected_shape,
        fortran_order=False,
    )
    for position, _source_group_identity in enumerate(group_order):
        sequence = reader.read(position, targets=False, mask=False)
        if sequence.seq_ix != seq_ids[position]:
            raise ValueError(f"seq_ix mismatch at position {position}: {sequence.seq_ix} != {seq_ids[position]}")
        features = sequence.features
        if features.shape != expected_shape[1:] or features.dtype != np.float32 or not features.flags.c_contiguous:
            raise ValueError(f"reader output contract changed at position {position}")
        output[position] = features
        source_payload_hash.update(memoryview(features))
        del features, sequence
    output.flush()
    descriptor = os.open(CACHE_PATH, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    del output, reader
    gc.collect()
    elapsed = time.perf_counter() - build_start
    actual_bytes = file_size(CACHE_PATH)
    if actual_bytes > DISK_BUDGET_BYTES:
        raise OSError(f"cache output exceeds disk budget: {actual_bytes}")
    hash_start = time.perf_counter()
    cache_file_sha256 = sha256_file(CACHE_PATH)
    cache_file_hash_seconds = time.perf_counter() - hash_start
    return {
        "build_wall_seconds": elapsed,
        "cache_file_bytes": actual_bytes,
        "expected_payload_bytes": expected_bytes,
        "build_source_payload_sha256": source_payload_hash.hexdigest(),
        "cache_file_sha256": cache_file_sha256,
        "cache_file_hash_seconds": cache_file_hash_seconds,
    }


def verify_cache(group_order: list[int], seq_ids: list[int], build_payload_sha256: str) -> dict[str, Any]:
    reader_open_start = time.perf_counter()
    parquet_reader = SequenceFile(DATA_PATH)
    parquet_reader_open_seconds = time.perf_counter() - reader_open_start
    mapped_open_start = time.perf_counter()
    mapped = np.load(CACHE_PATH, mmap_mode="r", allow_pickle=False)
    mmap_open_seconds = time.perf_counter() - mapped_open_start
    expected_shape = (len(group_order), SEQUENCE_LENGTH, N_FEATURES)
    if mapped.shape != expected_shape or mapped.dtype != np.dtype(np.float32) or not mapped.flags.c_contiguous:
        raise ValueError(f"mmap cache layout mismatch: shape={mapped.shape}, dtype={mapped.dtype}, c={mapped.flags.c_contiguous}")
    verify_start = time.perf_counter()
    decoded_hash = hashlib.sha256()
    mapped_hash = hashlib.sha256()
    compared_values = 0
    for position, _source_group_identity in enumerate(group_order):
        sequence = parquet_reader.read(position, targets=False, mask=False)
        if sequence.seq_ix != seq_ids[position]:
            raise ValueError(f"parity seq_ix mismatch at position {position}")
        source = sequence.features
        cache_row = mapped[position]
        if source.shape != cache_row.shape or source.dtype != cache_row.dtype:
            raise ValueError(f"parity shape/dtype mismatch at position {position}")
        if not source.flags.c_contiguous or not cache_row.flags.c_contiguous:
            raise ValueError(f"parity row is not C-contiguous at position {position}")
        if not np.array_equal(source.view(np.uint32), cache_row.view(np.uint32)):
            raise ValueError(f"bitwise feature parity failed at position {position}")
        decoded_hash.update(memoryview(source))
        mapped_hash.update(memoryview(cache_row))
        compared_values += int(source.size)
        del source, cache_row, sequence
    if decoded_hash.hexdigest() != build_payload_sha256:
        raise ValueError("independent decode payload hash disagrees with cache build")
    if mapped_hash.hexdigest() != decoded_hash.hexdigest():
        raise ValueError("mmap payload hash disagrees with decoded source")
    elapsed = time.perf_counter() - verify_start
    return {
        "parity_status": "PASS_BITWISE_ALL_FEATURE_VALUES",
        "compared_float32_values": compared_values,
        "decoded_payload_sha256": decoded_hash.hexdigest(),
        "mmap_payload_sha256": mapped_hash.hexdigest(),
        "verification_wall_seconds": elapsed,
        "parquet_reader_open_seconds": parquet_reader_open_seconds,
        "mmap_open_seconds": mmap_open_seconds,
        "mapped_array": mapped,
        "parquet_reader": parquet_reader,
    }


def run_epoch(method: str, group_order: list[int], seq_ids: list[int], parquet_reader: SequenceFile,
              mapped: np.ndarray) -> dict[str, Any]:
    epoch_start = time.perf_counter()
    fetch_latencies: list[float] = []
    full_read_latencies: list[float] = []
    scan_latencies: list[float] = []
    checksum = 0.0
    for position, _source_group_identity in enumerate(group_order):
        read_start = time.perf_counter()
        if method == "parquet":
            sequence = parquet_reader.read(position, targets=False, mask=False)
            if sequence.seq_ix != seq_ids[position]:
                raise ValueError(f"timed parquet seq_ix mismatch at position {position}")
            features = sequence.features
        else:
            features = mapped[position]
        fetch_end = time.perf_counter()
        partial_sum = float(np.sum(features, dtype=np.float64))
        read_end = time.perf_counter()
        fetch_latencies.append(fetch_end - read_start)
        full_read_latencies.append(read_end - read_start)
        scan_latencies.append(read_end - fetch_end)
        checksum += partial_sum
        del features
        if method == "parquet":
            del sequence
    return {
        "epoch_wall_seconds": time.perf_counter() - epoch_start,
        "fetch_latencies_seconds": fetch_latencies,
        "fetch_plus_full_scan_latencies_seconds": full_read_latencies,
        "full_scan_latencies_seconds": scan_latencies,
        "checksum_float64_sum": checksum,
    }


def run_benchmark(receipt: dict[str, Any], group_order: list[int], seq_ids: list[int], build: dict[str, Any],
                  verification: dict[str, Any], identity_check_seconds: float,
                  run_started_at_utc: str, run_start: float) -> None:
    mapped = verification.pop("mapped_array")
    verification.pop("parquet_reader")
    benchmark_reader_open_start = time.perf_counter()
    benchmark_reader = SequenceFile(DATA_PATH)
    benchmark_reader_open_seconds = time.perf_counter() - benchmark_reader_open_start

    manifest = {
        "experiment_id": "E528",
        "source_parquet_sha256": receipt["input_sha256"]["parquet_sha256"],
        "source_group_order": group_order,
        "subset_parquet_row_group_positions": list(range(len(group_order))),
        "source_seq_ix_order": seq_ids,
        "feature_columns": receipt["feature_columns"],
        "feature_columns_sha256": receipt["feature_columns_sha256"],
        "shape": list(mapped.shape),
        "dtype": "float32",
        "layout": "C-contiguous sequence/time/feature",
        "payload_sha256": verification["mmap_payload_sha256"],
        "cache_file_sha256": build["cache_file_sha256"],
        "built_from": "SequenceFile.read(index, targets=False, mask=False).features",
    }
    atomic_json(MANIFEST_PATH, manifest)
    manifest_bytes = file_size(MANIFEST_PATH)
    cache_and_manifest_bytes = file_size(CACHE_PATH) + manifest_bytes
    if cache_and_manifest_bytes > DISK_BUDGET_BYTES:
        raise OSError(f"cache plus manifest exceeds disk cap: {cache_and_manifest_bytes}")

    schedule = [(epoch_count, repetition) for epoch_count in EXPECTED_EPOCHS for repetition in range(REPETITIONS)]
    random.Random(SEED).shuffle(schedule)
    measurements: list[dict[str, Any]] = []
    write_progress("timed_replay_started", details={"schedule": schedule}, measurements=measurements)
    replay_start_utc = utc_now()
    replay_start = time.perf_counter()
    for epoch_count, repetition in schedule:
        methods = ["parquet", "mmap"]
        random.Random(SEED + 1000 * epoch_count + repetition).shuffle(methods)
        for method in methods:
            epoch_records: list[dict[str, Any]] = []
            before = usage_counters()
            run_start = time.perf_counter()
            fetch_latencies: list[float] = []
            full_read_latencies: list[float] = []
            scan_latencies: list[float] = []
            checksums: list[float] = []
            for _ in range(epoch_count):
                record = run_epoch(method, group_order, seq_ids, benchmark_reader, mapped)
                epoch_records.append({"epoch_wall_seconds": record["epoch_wall_seconds"]})
                fetch_latencies.extend(record["fetch_latencies_seconds"])
                full_read_latencies.extend(record["fetch_plus_full_scan_latencies_seconds"])
                scan_latencies.extend(record["full_scan_latencies_seconds"])
                checksums.append(record["checksum_float64_sum"])
            run_wall = time.perf_counter() - run_start
            after = usage_counters()
            result = {
                "epoch_count_K": epoch_count,
                "repetition": repetition,
                "method": method,
                "method_block_order": methods,
                "run_wall_seconds": run_wall,
                "epoch_wall_seconds": [record["epoch_wall_seconds"] for record in epoch_records],
                "mean_epoch_wall_seconds": statistics.mean(record["epoch_wall_seconds"] for record in epoch_records),
                "fetch_latency_p50_seconds": percentile(fetch_latencies, 50),
                "fetch_latency_p90_seconds": percentile(fetch_latencies, 90),
                "fetch_plus_full_scan_latency_p50_seconds": percentile(full_read_latencies, 50),
                "fetch_plus_full_scan_latency_p90_seconds": percentile(full_read_latencies, 90),
                "full_scan_latency_p50_seconds": percentile(scan_latencies, 50),
                "full_scan_latency_p90_seconds": percentile(scan_latencies, 90),
                "fetch_latency_samples_seconds": fetch_latencies,
                "fetch_plus_full_scan_latency_samples_seconds": full_read_latencies,
                "epoch_checksums_float64_sum": checksums,
                "usage_before": before,
                "usage_after": after,
                "page_fault_delta": {
                    "minor": after["minor_page_faults"] - before["minor_page_faults"],
                    "major": after["major_page_faults"] - before["major_page_faults"],
                },
            }
            measurements.append(result)
            write_progress(
                "timed_replay_in_progress",
                details={"last_completed": {"K": epoch_count, "repetition": repetition, "method": method},
                         "replay_elapsed_seconds": time.perf_counter() - replay_start},
                measurements=measurements,
            )

    replay_wall = time.perf_counter() - replay_start
    paired_checksums: list[dict[str, Any]] = []
    for epoch_count, repetition in schedule:
        pair = [item for item in measurements if item["epoch_count_K"] == epoch_count and item["repetition"] == repetition]
        if len(pair) != 2:
            raise RuntimeError(f"incomplete paired replay for K={epoch_count}, repetition={repetition}")
        parquet_result = next(item for item in pair if item["method"] == "parquet")
        mmap_result = next(item for item in pair if item["method"] == "mmap")
        if parquet_result["epoch_checksums_float64_sum"] != mmap_result["epoch_checksums_float64_sum"]:
            raise RuntimeError(f"consumer checksum mismatch for K={epoch_count}, repetition={repetition}")
        paired_checksums.append({
            "K": epoch_count,
            "repetition": repetition,
            "checksum_equal": True,
            "checksum_float64_sum": parquet_result["epoch_checksums_float64_sum"],
        })

    summary_by_K: dict[str, Any] = {}
    for epoch_count in EXPECTED_EPOCHS:
        summary_by_K[str(epoch_count)] = {}
        for method in ("parquet", "mmap"):
            runs = [item for item in measurements if item["epoch_count_K"] == epoch_count and item["method"] == method]
            fetch = [value for item in runs for value in item["fetch_latency_samples_seconds"]]
            full_read = [value for item in runs for value in item["fetch_plus_full_scan_latency_samples_seconds"]]
            per_epoch = [item["mean_epoch_wall_seconds"] for item in runs]
            total_walls = [item["run_wall_seconds"] for item in runs]
            summary_by_K[str(epoch_count)][method] = {
                "median_mean_epoch_wall_seconds": statistics.median(per_epoch),
                "p50_mean_epoch_wall_seconds": percentile(per_epoch, 50),
                "p90_mean_epoch_wall_seconds": percentile(per_epoch, 90),
                "median_total_K_replay_wall_seconds": statistics.median(total_walls),
                "per_sequence_fetch_latency_p50_seconds": percentile(fetch, 50),
                "per_sequence_fetch_latency_p90_seconds": percentile(fetch, 90),
                "per_sequence_fetch_plus_full_scan_latency_p50_seconds": percentile(full_read, 50),
                "per_sequence_fetch_plus_full_scan_latency_p90_seconds": percentile(full_read, 90),
                "per_sequence_samples": len(full_read),
            }

    all_epoch_wall = {
        method: [wall for item in measurements if item["method"] == method for wall in item["epoch_wall_seconds"]]
        for method in ("parquet", "mmap")
    }
    median_epoch_wall = {method: statistics.median(values) for method, values in all_epoch_wall.items()}
    delta_per_epoch = median_epoch_wall["parquet"] - median_epoch_wall["mmap"]
    setup_seconds = (
        build["build_wall_seconds"]
        + build["cache_file_hash_seconds"]
        + verification["verification_wall_seconds"]
        + verification["mmap_open_seconds"]
        + verification["parquet_reader_open_seconds"]
    )
    if delta_per_epoch > 0:
        break_even_K: int | None = math.floor(setup_seconds / delta_per_epoch) + 1
        break_even_reason = "first integer K for which measured cache build + exact parity validation + mmap open plus K mapped epochs is strictly faster on this 20-sequence subset"
    else:
        break_even_K = None
        break_even_reason = "null because the median mmap epoch was not faster than the median Parquet epoch in this measured subset"

    post_hashes = {
        "parquet_sha256": sha256_file(DATA_PATH),
        "groups_npy_sha256": sha256_file(GROUPS_PATH),
        "index_npy_sha256": sha256_file(INDEX_PATH),
        "reader_source_sha256": sha256_file(READER_PATH),
    }
    expected_post_hashes = {key: receipt["input_sha256"][key] for key in post_hashes}
    if post_hashes != expected_post_hashes:
        raise RuntimeError(f"an input changed during the run: expected={expected_post_hashes}, got={post_hashes}")

    cache_file_sha = build["cache_file_sha256"]
    output_bytes = file_size(CACHE_PATH) + file_size(MANIFEST_PATH)
    usage_final = usage_counters()
    if usage_final["peak_rss_bytes"] >= RSS_CAP_BYTES:
        raise MemoryError(f"observed process peak RSS exceeded the frozen cap: {usage_final['peak_rss_bytes']}")
    if output_bytes > DISK_BUDGET_BYTES:
        raise OSError(f"cache and manifest exceed the frozen disk cap: {output_bytes}")
    result = {
        "experiment_id": "E528",
        "status": "complete",
        "run_started_at_utc": run_started_at_utc,
        "run_wall_seconds_after_frozen_input_checks": time.perf_counter() - run_start,
        "input_identity_check_seconds_before_cache_build": identity_check_seconds,
        "replay_started_at_utc": replay_start_utc,
        "completed_at_utc": utc_now(),
        "scope": "bounded CPU systems profile only; no fit, GPU, target/mask decode, metric, score or public-transfer claim",
        "frozen_receipt_path": str(RECEIPT_PATH.relative_to(ROOT)),
        "frozen_receipt_sha256": sha256_file(RECEIPT_PATH),
        "benchmark_source_sha256": receipt["benchmark_source_sha256"],
        "environment": receipt["environment"],
        "resource_limits": {
            "requested": {
                "cpu_affinity_count": 1,
                "rss_cap_bytes": RSS_CAP_BYTES,
                "address_space_cap_bytes": AS_CAP_BYTES,
                "wall_timeout_seconds": WALL_CAP_SECONDS,
                "output_cache_disk_cap_bytes": DISK_BUDGET_BYTES,
            },
            "observed_process_limits": process_limits(),
            "observed_peak_process_rss_bytes": usage_final["peak_rss_bytes"],
            "observed_peak_process_rss_below_cap": usage_final["peak_rss_bytes"] < RSS_CAP_BYTES,
            "final_usage": usage_final,
            "no_cuda_or_torch_import": True,
        },
        "inputs": {
            "parquet_path": str(DATA_PATH.relative_to(ROOT)),
            "parquet_bytes": file_size(DATA_PATH),
            "parquet_sha256_before_and_after": receipt["input_sha256"]["parquet_sha256"],
            "group_order": group_order,
            "seq_ix_order": seq_ids,
            "feature_columns_count": len(FEATURE_COLUMNS),
            "feature_columns_sha256": receipt["feature_columns_sha256"],
            "feature_columns": receipt["feature_columns"],
            "shape": [len(group_order), SEQUENCE_LENGTH, N_FEATURES],
            "reader_source_sha256": receipt["input_sha256"]["reader_source_sha256"],
            "post_run_input_hashes": post_hashes,
        },
        "cache": {
            "npy_path": str(CACHE_PATH.relative_to(ROOT)),
            "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
            "payload_bytes": int(np.prod(mapped.shape, dtype=np.int64)) * np.dtype(np.float32).itemsize,
            "npy_file_bytes": file_size(CACHE_PATH),
            "manifest_bytes": file_size(MANIFEST_PATH),
            "cache_and_manifest_disk_bytes": output_bytes,
            "cache_disk_cap_bytes": DISK_BUDGET_BYTES,
            "cache_file_sha256": cache_file_sha,
            "feature_payload_sha256": verification["mmap_payload_sha256"],
            "build_wall_seconds": build["build_wall_seconds"],
            "cache_file_sha256_wall_seconds": build["cache_file_hash_seconds"],
            "build_source_payload_sha256": build["build_source_payload_sha256"],
            "mmap_open_seconds": verification["mmap_open_seconds"],
            "parity_reader_open_seconds": verification["parquet_reader_open_seconds"],
            "benchmark_reader_open_seconds": benchmark_reader_open_seconds,
            "exact_parity_status": verification["parity_status"],
            "compared_float32_values": verification["compared_float32_values"],
            "parity_validation_wall_seconds": verification["verification_wall_seconds"],
        },
        "page_cache_scope": {
            "global_or_per_file_cache_drop": False,
            "first_timed_pass_label": "first observed replay after cache build and full parity validation; kernel page-cache state is unknown, not cold",
            "repeats_label": "subsequent repeated passes after prior accesses; not an independently controlled warm-cache condition",
            "parquet_features_accessed_before_timing": "yes, during cache build and exact parity validation",
            "mmap_features_accessed_before_timing": "yes, during exact parity validation; file was just generated",
        },
        "timing_design": {
            "epoch_counts_K": EXPECTED_EPOCHS,
            "repetitions_per_K_and_method": REPETITIONS,
            "randomization_seed": SEED,
            "replay_wall_seconds": replay_wall,
            "benchmark_reader_open_seconds": benchmark_reader_open_seconds,
            "method_by_K": summary_by_K,
            "per_run_measurements": measurements,
            "paired_consumer_checksum_parity": paired_checksums,
            "median_epoch_wall_seconds_across_all_K_and_repetitions": median_epoch_wall,
            "median_parquet_minus_mmap_epoch_seconds": delta_per_epoch,
        },
        "break_even": {
            "scope": "this valid_subset20 cohort only; no full-training extrapolation",
            "cache_setup_seconds_including_required_verification": setup_seconds,
            "input_identity_hashing_seconds_reported_separately": identity_check_seconds,
            "strict_first_K_or_null": break_even_K,
            "reason": break_even_reason,
            "formula": "setup + K*mmap_median_epoch < K*parquet_median_epoch",
            "transfer_and_fit_compute_included": False,
        },
        "limitations": [
            "Only 20 complete sequences were measured; the result does not establish full-train throughput.",
            "Targets, scored masks, data staging, model compute, training-loader workers and optimizer work were excluded.",
            "The feature cache uses 179.2 MB raw float32 payload; a 10,535-sequence cache would need approximately 94.3936 GB before other arrays and is not admitted here.",
            "No controlled cold-page-cache observation was made; the first timed replay followed feature reads during build and parity validation.",
            "The common NumPy sum is an all-values-touch surrogate, not a training step; end-to-end training speedup may differ.",
        ],
    }
    atomic_json(RESULT_PATH, result)
    write_progress("complete", details={"result_path": str(RESULT_PATH.relative_to(ROOT)), "replay_wall_seconds": replay_wall}, measurements=measurements)
    print(json.dumps({
        "status": result["status"],
        "result_path": str(RESULT_PATH),
        "build_wall_seconds": build["build_wall_seconds"],
        "parity_validation_wall_seconds": verification["verification_wall_seconds"],
        "observed_peak_process_rss_bytes": usage_final["peak_rss_bytes"],
        "cache_and_manifest_disk_bytes": output_bytes,
        "median_epoch_wall_seconds": median_epoch_wall,
        "break_even_K": break_even_K,
        "replay_wall_seconds": replay_wall,
    }, indent=2))


def run() -> None:
    run_started_at_utc = utc_now()
    run_start = time.perf_counter()
    receipt, group_order, seq_ids, identity_check_seconds = verify_frozen_inputs()
    write_progress("cache_build_started", details={
        "frozen_receipt_sha256": sha256_file(RECEIPT_PATH),
        "input_identity_check_seconds": identity_check_seconds,
    })
    try:
        build = build_cache(group_order, seq_ids)
        write_progress("cache_build_complete", details=build)
        verification = verify_cache(group_order, seq_ids, build["build_source_payload_sha256"])
        persisted_verification = {key: value for key, value in verification.items() if key not in ("mapped_array", "parquet_reader")}
        write_progress("exact_parity_complete", details={**build, **persisted_verification})
        run_benchmark(receipt, group_order, seq_ids, build, verification, identity_check_seconds,
                      run_started_at_utc, run_start)
    except Exception as exc:
        write_progress("failed", details={
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "elapsed_seconds_before_failure": time.perf_counter() - run_start,
            "cache_file_exists": CACHE_PATH.exists(),
            "cache_file_bytes": file_size(CACHE_PATH) if CACHE_PATH.exists() else 0,
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true", help="record hashes and exact sequence/feature order; no feature timing")
    modes.add_argument("--run", action="store_true", help="run the frozen bounded cache build, parity and replay profile")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze:
        preflight()
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
