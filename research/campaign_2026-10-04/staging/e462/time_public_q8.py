"""Stage a fixed CPU15 full-callback timing comparison for E462 packages.

This program is gated on passed E462 parity and holdout-scoring reports. It
preloads the first three E457 streams, then times only the exact per-row
DataPoint/predict/validate/copy loop used by utils.ScorerStepByStep.score.
It is a source draft; root admits and launches the timing phase separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
E457_DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
SOURCE_ZIP_SHA256 = "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb"
PARENT = "c20261004_combo_s3_final_r1"
HEAD = "c20261004_combo_head_s3_final"
PARENT_SHA256 = "64c2b28651b414778d6af3c33bb9d54f653a46fb0be087de2e3c44ca88c31166"
HEAD_SHA256 = "f1cb88f38b6d5b5b67859346b439e69d610928bc836baac8867e84e7b1c95d85"


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_member_sha256(path: Path, member: str) -> str:
    """Hash one member of an immutable ZIP without extracting it."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ThreadCycles:
    """Read user-thread hardware cycles when perf_event_open is available."""

    def __init__(self) -> None:
        import ctypes
        import struct

        libc = ctypes.CDLL(None, use_errno=True)
        attr = ctypes.create_string_buffer(
            struct.pack("IIQQQQQIIQ", 0, 64, 0, 0, 0, 0, 0x60, 0, 0, 0)
        )
        self.fd = int(libc.syscall(298, attr, 0, -1, -1, 0))

    def read(self) -> int | None:
        """Return the current counter, or None when the kernel disallows it."""
        return int.from_bytes(os.read(self.fd, 8), "little") if self.fd >= 0 else None

    def close(self) -> None:
        """Close the optional performance counter."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _load_verifier():
    """Load only the package callback loader and frozen parity gate helpers."""
    path = Path(__file__).with_name("verify_public_q8.py")
    spec = importlib.util.spec_from_file_location("e462_timing_verifier", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load E462 verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _score_gate(
    build_dir: Path, parity_path: Path, score_path: Path, package_paths: dict[str, Path]
) -> tuple[str, str]:
    """Require the exact passing parity and completed descriptive score reports."""
    verifier = _load_verifier()
    parity = verifier._validate_parity_gate(build_dir, parity_path)
    lineage = verifier._verify_composite_lineage(build_dir / "build_report.json", build_dir)
    score = json.loads(score_path.read_text())
    if (
        score.get("schema") != "e462-public-combo-q8-holdout-score-v1"
        or score.get("parity_report_sha256") != sha256_file(parity_path)
        or score.get("parity_passed") is not True
        or score.get("n_sequences") != 368
        or int(score.get("n_components", 0)) <= 0
        or int(score.get("selected_rows", 0)) <= 0
        or score.get("composite_lineage") != lineage
    ):
        raise ValueError("E462 score report is missing or bound to different evidence")
    for label, path in package_paths.items():
        candidate = score.get("candidates", {}).get(label, {})
        if candidate.get("zip_sha256") != sha256_file(path):
            raise ValueError(f"E462 score report does not bind package {label}")
    if not parity.get("passed"):
        raise ValueError("E462 parity gate did not pass")
    return sha256_file(score_path), sha256_file(build_dir / "build_report.json")


def _load_streams(data_path: Path) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], float]:
    """Load the exact three E457 streams before timed trials begin."""
    sys.path.insert(0, str(ROOT))
    from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH

    if sha256_file(data_path) != E457_DATA_SHA256:
        raise ValueError("timing panel differs from the fixed E457 data")
    ready = json.loads((CAMPAIGN / "E457_head_transfer/readiness.json").read_text())
    expected_ids = list(map(int, ready["parent"]["dataset_identity"]["sequence_ids"][:3]))
    started = time.perf_counter()
    parquet = pq.ParquetFile(data_path)
    if parquet.num_row_groups < 3:
        raise ValueError("E457 timing panel has fewer than three streams")
    streams = []
    for group in range(3):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS], use_threads=False
        )
        sequence = int(table.column("seq_ix")[0].as_py())
        features = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
        ).astype(np.float32)
        need = table.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool)
        if (
            sequence != expected_ids[group]
            or features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
            or need.shape != (SEQUENCE_LENGTH,)
            or not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= 99)
        ):
            raise ValueError(f"E457 timing stream {group} differs from its fixed identity")
        streams.append((sequence, features, need))
    return streams, time.perf_counter() - started


def _trial(
    label: str,
    package: Path,
    streams: list[tuple[int, np.ndarray, np.ndarray]],
    cycles: ThreadCycles,
) -> dict[str, object]:
    """Time one fresh package predictor over the exact ScorerStepByStep row loop."""
    sys.path.insert(0, str(ROOT))
    from utils import DataPoint, SEQUENCE_LENGTH

    os.environ["WNN_NATIVE_LEVEL"] = "v3"
    os.environ.pop("WNN_PRECISION", None)
    verifier = _load_verifier()
    temporary, model = verifier._read_package(package)
    row_count = 0
    output_buffers = []
    cycle_start = cycles.read()
    started = time.perf_counter()
    try:
        for sequence, features, need in streams:
            predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
            for step in range(SEQUENCE_LENGTH):
                point = DataPoint(sequence, step, bool(need[step]), features[step])
                value = model.predict(point)
                if not need[step]:
                    if value is not None:
                        raise ValueError("callback returned a prediction during warmup")
                else:
                    value = np.asarray(value, dtype=np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError("callback returned malformed or nonfinite scores")
                    predictions[step] = value
                row_count += 1
            output_buffers.append(predictions)
        elapsed = time.perf_counter() - started
        cycle_end = cycles.read()
    finally:
        temporary.cleanup()
    cycle_count = (
        None
        if cycle_start is None or cycle_end is None
        else cycle_end - cycle_start
    )
    del output_buffers
    if row_count != 60_000:
        raise ValueError(f"trial processed {row_count} rows instead of 60,000")
    return {
        "candidate": label,
        "rows": row_count,
        "elapsed_seconds": elapsed,
        "us_per_row": elapsed * 1e6 / row_count,
        "cycles": cycle_count,
        "kcycles_per_row": None if cycle_count is None else cycle_count / row_count / 1000,
    }


def run(args: argparse.Namespace) -> None:
    """Run five alternating paired comparisons, preserving each trial record."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if sorted(os.sched_getaffinity(0)) != [args.cpu]:
        raise ValueError(f"timing must run with affinity fixed to CPU {args.cpu}")
    load_start = Path("/proc/loadavg").read_text().split()
    load1 = float(load_start[0])
    if load1 > args.max_load1:
        raise RuntimeError(f"host load {load1} exceeds quiet-window limit {args.max_load1}")
    packages = {"i16": args.control, "q8r": args.q8r, "q8x": args.q8x}
    for label, path in packages.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} package missing: {path}")
    score_sha, build_sha = _score_gate(
        args.build_dir, args.parity_report, args.score_report, packages
    )
    if sha256_file(args.control) != SOURCE_ZIP_SHA256:
        raise ValueError("timing control differs from the fixed public i16 ZIP")
    streams, preload_seconds = _load_streams(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    trials_path = args.output / "trials.jsonl"
    trials = []
    cycles = ThreadCycles()
    command = list(sys.argv)
    hardware = {
        "platform": platform.platform(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_start": load_start,
        "loadavg_end": None,
        "data_sha256": sha256_file(args.data),
        "preload_seconds_outside_timing": preload_seconds,
        "score_report_sha256": score_sha,
        "build_report_sha256": build_sha,
        "parity_report_sha256": sha256_file(args.parity_report),
        "command": command,
    }
    try:
        with trials_path.open("x") as journal:
            for comparison, candidate in (("i16_vs_q8r", "q8r"), ("i16_vs_q8x", "q8x")):
                for repeat in range(args.paired_repeats):
                    order = ("i16", candidate) if repeat % 2 == 0 else (candidate, "i16")
                    for label in order:
                        record = {
                            "comparison": comparison,
                            "repeat": repeat,
                            "order": list(order),
                            "candidate": label,
                            "zip_sha256": sha256_file(packages[label]),
                            "native_library_sha256": {
                                member: zip_member_sha256(packages[label], member)
                                for member in (
                                    "v2/wnn_native.abi3.so",
                                    "v3/wnn_native.abi3.so",
                                )
                            },
                            "affinity": sorted(os.sched_getaffinity(0)),
                            "loadavg_before": Path("/proc/loadavg").read_text().split(),
                            "command": command,
                        }
                        try:
                            if float(record["loadavg_before"][0]) > args.max_load1:
                                raise RuntimeError("host left the preregistered quiet-load window")
                            record.update(_trial(label, packages[label], streams, cycles))
                            record["status"] = "passed"
                        except Exception as error:
                            record.update(status="failed", error=f"{type(error).__name__}: {error}")
                            journal.write(json.dumps(record, sort_keys=True) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                            raise
                        journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                        trials.append(record)
    finally:
        cycles.close()
    hardware["loadavg_end"] = Path("/proc/loadavg").read_text().split()
    summaries = {}
    for label in ("i16", "q8r", "q8x"):
        values = [float(row["us_per_row"]) for row in trials if row["candidate"] == label]
        summaries[label] = {
            "trials": len(values),
            "median_us_per_row": statistics.median(values),
            "all_us_per_row": values,
        }
    paired = {}
    for comparison, candidate in (("i16_vs_q8r", "q8r"), ("i16_vs_q8x", "q8x")):
        rows = [row for row in trials if row["comparison"] == comparison]
        by_repeat = {
            repeat: {row["candidate"]: float(row["us_per_row"]) for row in rows if row["repeat"] == repeat}
            for repeat in range(args.paired_repeats)
        }
        if any(set(values) != {"i16", candidate} for values in by_repeat.values()):
            raise RuntimeError(f"incomplete matched timing pair: {comparison}")
        paired[comparison] = {
            "median_candidate_minus_control_us_per_row": statistics.median(
                values[candidate] - values["i16"] for values in by_repeat.values()
            ),
            "median_candidate_over_control": statistics.median(
                values[candidate] / values["i16"] for values in by_repeat.values()
            ),
            "pairs": by_repeat,
        }
    result = {
        "schema": "e462-public-combo-q8-timing-v1",
        "passed": len(trials) == args.paired_repeats * 4,
        "sequences_per_trial": 3,
        "rows_per_trial": 60_000,
        "paired_repeats_per_comparison": args.paired_repeats,
        "native_level": "v3",
        "hardware": hardware,
        "summaries": summaries,
        "paired_comparisons": paired,
        "trials_file": "trials.jsonl",
        "scope": "CPU callback diagnostic only; no public runtime conversion or speed claim",
    }
    with (args.output / "timing_report.json").open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Parse a bounded source draft; no phase runs without explicit CLI invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--q8r", type=Path, required=True)
    parser.add_argument("--q8x", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "cache/valid_subset20.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--max-load1", type=float, default=1.0)
    parser.add_argument("--paired-repeats", type=int, choices=(5,), default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
