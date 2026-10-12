#!/usr/bin/env python3
"""One-core alternating callback timing for the fixed reference and C ZIPs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_submission_callbacks import (  # noqa: E402
    load_sequences,
    read_cgroup_snapshot,
    run_callbacks,
    sha256_bytes,
    sha256_file,
)


def load_package(archive_path: Path, receipt_path: Path, destination: Path) -> tuple[Path, dict]:
    """Extract one ZIP only after its declared file and archive hashes match."""
    receipt = json.loads(receipt_path.read_text())
    if receipt["artifact"]["sha256"] != sha256_file(archive_path):
        raise ValueError(f"archive hash mismatch: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC mismatch: {archive_path}")
        hashes = {name: sha256_bytes(archive.read(name)) for name in archive.namelist()}
        if hashes != receipt["artifact"]["members"]:
            raise ValueError(f"member hashes mismatch: {archive_path}")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError(f"unsafe ZIP path: {info.filename!r}")
            if info.is_dir():
                continue
            path = destination.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
    if sha256_file(destination / "solution.py") != receipt["source_code"]["sha256"]:
        raise ValueError(f"solution.py hash mismatch: {archive_path}")
    return destination, receipt


def run_one(package_root: Path, sequences: list[dict], cpu: int) -> dict:
    """Run complete reset-separated sequences and time callback calls only."""
    os.environ["WNN_NATIVE_LEVEL"] = "v3"
    sys.modules.pop("wnn_native", None)
    solution = package_root / "solution.py"
    spec = importlib.util.spec_from_file_location(f"paired_probe_{time.monotonic_ns()}", solution)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {solution}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    if model.model.precision != "q8vnni":
        raise ValueError(f"wrong precision: {model.model.precision}")
    elapsed = 0
    hashes = {}
    for sequence in sequences:
        duration, predictions = run_callbacks(model, sequence)
        elapsed += duration
        hashes[str(sequence["sequence_id"])] = sha256_bytes(
            predictions[sequence["need"]].tobytes(order="C")
        )
    result = {
        "elapsed_callback_ns": elapsed,
        "rows": len(sequences) * 20000,
        "callbacks_per_sequence": 20000,
        "mean_callback_us_per_row": elapsed / (len(sequences) * 20000) / 1000,
        "precision": model.model.precision,
        "native_level": module._LEVEL,
        "model_count": len(json.loads((package_root / "members.json").read_text())),
        "prediction_sha256_by_sequence": hashes,
    }
    del model, module
    sys.path.remove(str(package_root / "v3"))
    sys.modules.pop("wnn_native", None)
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--candidate-zip", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    allowed = sorted(os.sched_getaffinity(0))
    if args.cpu not in allowed:
        raise ValueError(f"CPU {args.cpu} not in allowed affinity {allowed}")
    os.sched_setaffinity(0, {args.cpu})
    loadavg_start = os.getloadavg()
    cgroup_before = read_cgroup_snapshot()
    sequences = load_sequences(args.data, 0, 10)
    order = ["reference", "candidate", "reference", "candidate"]
    archives = {
        "reference": (
            args.reference_zip.resolve(strict=True),
            args.reference_receipt.resolve(strict=True),
        ),
        "candidate": (
            args.candidate_zip.resolve(strict=True),
            args.candidate_receipt.resolve(strict=True),
        ),
    }
    prior_hashes: dict[str, dict[str, str]] = {}
    trials = []
    with tempfile.TemporaryDirectory(prefix="wnn-ref-c-paired-") as temporary:
        roots = {}
        receipts = {}
        for label, (archive, receipt) in archives.items():
            roots[label], receipts[label] = load_package(archive, receipt, Path(temporary) / label)
        for index, label in enumerate(order):
            start = time.perf_counter()
            result = run_one(roots[label], sequences, args.cpu)
            result["trial"] = index
            result["label"] = label
            result["wall_seconds_including_model_load"] = time.perf_counter() - start
            if (
                label in prior_hashes
                and result["prediction_sha256_by_sequence"] != prior_hashes[label]
            ):
                raise ValueError(f"replay changed predictions for {label}")
            prior_hashes[label] = result["prediction_sha256_by_sequence"]
            trials.append(result)
    result = {
        "schema": "wnn.paired-native-callback-probe.v1",
        "scope": "Alternating callback-capacity probe only; no metric, score or EPYC runtime claim.",
        "order": order,
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_before": cgroup_before,
        "cgroup_after": read_cgroup_snapshot(),
        "loadavg_start": loadavg_start,
        "input_data": str(args.data.resolve(strict=True)),
        "input_sha256": sha256_file(args.data.resolve(strict=True)),
        "sequence_ids": [sequence["sequence_id"] for sequence in sequences],
        "feature_sha256_by_sequence": {
            str(sequence["sequence_id"]): sequence["features_sha256"] for sequence in sequences
        },
        "reference": {
            "zip": str(archives["reference"][0]),
            "sha256": sha256_file(archives["reference"][0]),
            "receipt_sha256": sha256_file(archives["reference"][1]),
        },
        "candidate": {
            "zip": str(archives["candidate"][0]),
            "sha256": sha256_file(archives["candidate"][0]),
            "receipt_sha256": sha256_file(archives["candidate"][1]),
        },
        "trials": trials,
        "cross_replay_bitwise_identical": True,
        "loadavg_end": os.getloadavg(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trials": [
                    {
                        "label": trial["label"],
                        "mean_callback_us_per_row": trial["mean_callback_us_per_row"],
                        "precision": trial["precision"],
                        "native_level": trial["native_level"],
                        "model_count": trial["model_count"],
                    }
                    for trial in trials
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
