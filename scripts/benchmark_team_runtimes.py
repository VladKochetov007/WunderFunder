"""Paired, callback-level benchmark for pair_ftt2 C++ and Rust runtimes.

The frozen campaign loader supplies both runtime implementations and the scorer's
DataPoint/replay adapter. This worker compares the same archived weights, members,
coefficients, and two complete validation sequences; it is not a quality grader.
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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


RUNTIMES = ("cpp", "rust")
ABBA_ORDER = ("cpp", "rust", "rust", "cpp")
RESET_SEQUENCE = -987654321


def digest(path: Path) -> str:
    """Hash a file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash a contiguous array without materializing a second bytes copy."""
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def abba_order(cycles: int) -> tuple[tuple[int, int, str], ...]:
    """Build repeated A-B-B-A blocks, labeled by cycle and within-block phase."""
    if cycles < 1 or cycles > 10:
        raise ValueError("cycles must be in [1, 10]")
    return tuple(
        (cycle, phase, runtime)
        for cycle in range(cycles)
        for phase, runtime in enumerate(ABBA_ORDER)
    )


def parity_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool]:
    """Summarize paired output differences without assigning a quality direction."""
    import numpy as np

    if left.shape != right.shape:
        raise ValueError("paired outputs have different shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired outputs contain non-finite values")
    delta = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    absolute = np.abs(delta)
    return {
        "bitwise_equal": bool(np.array_equal(left, right)),
        "different_elements": int(np.count_nonzero(delta)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the same pair_ftt2 callbacks in repeated C++/Rust ABBA order"
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--first-group", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=None, help="pin to one allowed logical CPU")
    return parser.parse_args()


def _pin_and_describe_cpu(requested_cpu: int | None) -> dict[str, Any]:
    """Optionally pin this process and report observed host CPU identity."""
    if not hasattr(os, "sched_getaffinity"):
        if requested_cpu is not None:
            raise RuntimeError("--cpu requires Linux CPU-affinity support")
        affinity: list[int] = []
    else:
        before = sorted(os.sched_getaffinity(0))
        if requested_cpu is not None:
            if requested_cpu not in before:
                raise ValueError(f"requested CPU {requested_cpu} is outside current affinity")
            os.sched_setaffinity(0, {requested_cpu})
        affinity = sorted(os.sched_getaffinity(0))

    cpu_records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not line.strip():
                if current:
                    cpu_records.append(current)
                    current = {}
                continue
            key, separator, value = line.partition(":")
            if not separator:
                continue
            key = key.strip()
            if key == "processor":
                current["processor"] = int(value.strip())
            elif key in ("model name", "flags"):
                current[key] = value.strip()
    except OSError:
        cpu_records = []
    selected = next(
        (record for record in cpu_records if record.get("processor") in affinity),
        cpu_records[0] if cpu_records else {},
    )
    first_flags = next((record.get("flags") for record in cpu_records if record.get("flags")), None)
    return {
        "requested_cpu": requested_cpu,
        "affinity_before": before if hasattr(os, "sched_getaffinity") else None,
        "affinity_after": affinity,
        "model_name": selected.get("model name", platform.processor()),
        "selected_processor": selected.get("processor"),
        "selected_cpu_flags": selected.get("flags"),
        "first_proc_cpu_flags": first_flags,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
    }


def _load_campaign(campaign: Path):
    """Load the exact campaign replay adapter after checking its admission hash."""
    source_path = campaign / "score_archive.py"
    inventory_path = campaign / "INVENTORY.json"
    admission_path = campaign / "FULL_REPLAY_ADMISSION_R1.json"
    for path in (source_path, inventory_path, admission_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    admission = json.loads(admission_path.read_text())
    source_sha256 = digest(source_path)
    if source_sha256 != admission.get("source_sha256"):
        raise ValueError("frozen campaign replay source differs from its admission receipt")
    spec = importlib.util.spec_from_file_location("team_runtime_campaign_replay", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load campaign replay source {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.HERE).resolve() != campaign.resolve():
        raise ValueError("campaign replay adapter resolved a different campaign directory")
    return module, admission, source_sha256, digest(inventory_path)


def _read_two_sequences(score: Any, data_path: Path, first_group: int):
    """Read two complete raw-feature sequences, retaining only their required arrays."""
    import numpy as np
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(data_path)
    if first_group < 0 or first_group + 2 > parquet.num_row_groups:
        raise ValueError("two-sequence selection is outside the parquet row groups")
    sequences = []
    for group in range(first_group, first_group + 2):
        table = parquet.read_row_group(group, use_threads=False)
        sequence, need = score.validate_sequence(table)
        features = np.column_stack([table[column].to_numpy() for column in score.FEATURE_COLUMNS])
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.shape != (20_000, 112) or not np.isfinite(features).all():
            raise ValueError(f"row group {group} has invalid raw feature data")
        if need.shape != (20_000,) or np.any(need[:99]) or not np.all(need[99:]):
            raise ValueError(f"row group {group} has invalid causal need_prediction mask")
        sequences.append(
            {
                "row_group": group,
                "sequence": int(sequence),
                "features": features,
                "need": np.ascontiguousarray(need, dtype=np.bool_),
                "feature_sha256": array_hash(features),
                "need_sha256": array_hash(np.ascontiguousarray(need, dtype=np.bool_)),
            }
        )
        del table
    return sequences


def _runtime_identity(runtime: str, model: Any, packages: dict[str, Any]) -> dict[str, Any]:
    """Bind loaded callbacks and their actual native library to inventory entries."""
    import sys

    if runtime == "cpp":
        package = packages["pair_ftt2"]
        module_name = "wnn_cpp"
        quant_backend = getattr(model, "quant_backend", None)
        if quant_backend is None:
            raise RuntimeError("pair_ftt2 callback did not report its actual quant_backend")
        expected_precision: dict[str, Any] = {
            "engine": "official pair_ftt2 callback",
            "fast_activations": True,
            "half_weights": False,
            "dynamic_int8": True,
            "quant_backend_actual": quant_backend,
            "member_count": model.member_count,
        }
    else:
        package = packages["E"]
        module_name = "wnn_native"
        precision_path = Path(package["package"]) / "precision.txt"
        precision_record = package["files"].get("precision.txt")
        if precision_record is None or digest(precision_path) != precision_record["sha256"]:
            raise ValueError("Rust precision file differs from inventory")
        precision_file = precision_path.read_text().strip()
        precision_actual = getattr(model, "precision", None)
        if precision_file != "q8vnni":
            raise ValueError("campaign Rust conversion must remain aligned to inventory Q8VNNI")
        if not precision_actual:
            raise RuntimeError("Rust callback did not expose its actual precision")
        expected_precision = {
            "engine": "campaign Rust conversion using pair_ftt2 members",
            "precision_requested": "q8vnni",
            "precision_file": precision_file,
            "precision_actual": precision_actual,
            "coefficients_actual": list(getattr(model, "coefficients", [])),
            "fast_activations": True,
            "budget_s": None,
            "active_members": getattr(model, "active_members", None),
        }
    native_module = sys.modules.get(module_name)
    native_path = Path(native_module.__file__).resolve() if native_module else None
    if native_path is None:
        raise RuntimeError(f"loaded callback did not expose {module_name} in sys.modules")
    relative = native_path.relative_to(Path(package["package"]).resolve()).as_posix()
    file_record = package["files"].get(relative)
    if file_record is None or digest(native_path) != file_record["sha256"]:
        raise ValueError(f"loaded native library is not inventory-bound: {native_path}")
    return {
        "runtime": runtime,
        "native_module_path": str(native_path),
        "native_module_sha256": file_record["sha256"],
        "native_module_inventory_path": relative,
        "runtime_settings": expected_precision,
    }


def _reset_before_replay(score: Any, model: Any, sample: dict[str, Any]) -> None:
    """Exercise the official sequence-change reset path outside the timed interval."""
    score.replay(model, RESET_SEQUENCE, sample["features"][:99], sample["need"][:99])


def _run_sample(score: Any, runtime: str, model: Any, sample: dict[str, Any]):
    """Time one complete sequence and return its output and separate clocks."""
    _reset_before_replay(score, model, sample)
    process_start = time.process_time()
    wall_start = time.perf_counter()
    prediction = score.replay(model, sample["sequence"], sample["features"], sample["need"])
    wall_seconds = time.perf_counter() - wall_start
    process_seconds = time.process_time() - process_start
    import numpy as np

    if prediction.shape != (20_000, 2) or not np.isfinite(prediction).all():
        raise ValueError(f"{runtime} produced invalid predictions for group {sample['row_group']}")
    return prediction, {
        "runtime": runtime,
        "row_group": sample["row_group"],
        "seq_ix": sample["sequence"],
        "input_sha256": sample["feature_sha256"],
        "need_sha256": sample["need_sha256"],
        "prediction_sha256": array_hash(prediction),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "finite": True,
        "process_time_seconds": process_seconds,
        "perf_counter_seconds": wall_seconds,
    }


def _load_average() -> list[float] | None:
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def main() -> None:
    """Run bounded same-weight C++/Rust paired callback comparisons."""
    args = _parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("the imported original C++ callback requires Python 3.11")
    if args.out.exists():
        raise FileExistsError(args.out)
    worker_path = Path(__file__).resolve()
    worker_sha256 = digest(worker_path)
    schedule = abba_order(args.cycles)
    cpu = _pin_and_describe_cpu(args.cpu)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    score, admission, campaign_source_sha256, inventory_sha256 = _load_campaign(
        args.campaign.resolve()
    )
    inventory_path = args.campaign / "INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    utils_path = Path(score.ROOT) / "utils.py"
    utils_sha256 = digest(utils_path)
    inventory_pair = inventory["packages"]["pair_ftt2"]
    inventory_rust = inventory["packages"]["E"]
    if args.data.stat().st_size != inventory["data"]["bytes"]:
        raise ValueError("valid.parquet size differs from the campaign inventory")
    data_stat_before = args.data.stat()
    data_sha256 = digest(args.data)
    if data_sha256 != inventory["data"]["sha256"]:
        raise ValueError("valid.parquet content differs from the campaign inventory")

    sequences = _read_two_sequences(score, args.data, args.first_group)
    models = {
        "cpp": score.load_model("pair_ftt2", "original", None),
        "rust": score.load_model("pair_ftt2", "rust", None),
    }
    runtime_identities = {
        runtime: _runtime_identity(runtime, model, inventory["packages"])
        for runtime, model in models.items()
    }
    expected_coefficients = [member["coef"] for member in inventory_pair["members"]]
    actual_members = json.loads((Path(inventory_pair["package"]) / "members.json").read_text())
    if [member["weights"] for member in actual_members] != [
        member["weights"] for member in inventory_pair["members"]
    ] or [member["coef"] for member in actual_members] != expected_coefficients:
        raise ValueError("loaded pair_ftt2 members/coefficients differ from inventory")
    expected_member_count = len(actual_members)
    if runtime_identities["cpp"]["runtime_settings"]["member_count"] != expected_member_count:
        raise ValueError("C++ callback loaded a different number of members")
    rust_settings = runtime_identities["rust"]["runtime_settings"]
    if rust_settings["active_members"] != expected_member_count:
        raise ValueError("Rust callback has a different active member count")
    import numpy as np

    rust_coefficients = np.asarray(rust_settings["coefficients_actual"], dtype=np.float32)
    expected_rust_coefficients = np.asarray(expected_coefficients, dtype=np.float32)
    if not np.array_equal(rust_coefficients, expected_rust_coefficients):
        raise ValueError("Rust callback coefficients differ from the pair_ftt2 member manifest")

    args.out.mkdir(parents=True)
    progress_path = args.out / "progress.jsonl"

    # Unmeasured one-sequence priming prevents first-call setup from dominating A's first slot.
    warmup_records = []
    warmup_hashes: dict[tuple[int, str], str] = {}
    for runtime in RUNTIMES:
        model = models[runtime]
        sample = sequences[0]
        _reset_before_replay(score, model, sample)
        warmup = score.replay(model, sample["sequence"], sample["features"], sample["need"])
        prediction_sha256 = array_hash(warmup)
        warmup_hashes[(sample["row_group"], runtime)] = prediction_sha256
        warmup_records.append(
            {
                "runtime": runtime,
                "row_group": sample["row_group"],
                "prediction_sha256": prediction_sha256,
            }
        )

    started_wall = time.perf_counter()
    started_process = time.process_time()
    load_start = _load_average()
    expected_hashes: dict[tuple[int, str], str] = {}
    samples: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    with progress_path.open("x") as progress:
        for sequence in sequences:
            for cycle in range(args.cycles):
                predictions: list[np.ndarray] = []
                for phase, runtime in enumerate(ABBA_ORDER):
                    prediction, record = _run_sample(score, runtime, models[runtime], sequence)
                    key = (sequence["row_group"], runtime)
                    prediction_sha = record["prediction_sha256"]
                    previous_hash = expected_hashes.setdefault(
                        key, warmup_hashes.get(key, prediction_sha)
                    )
                    record.update(
                        {
                            "cycle": cycle,
                            "phase": phase,
                            "abba_order": list(ABBA_ORDER),
                            "reset_before_timing": True,
                            "reset_replay_bitwise": previous_hash == prediction_sha,
                        }
                    )
                    if previous_hash != prediction_sha:
                        raise ValueError(
                            f"{runtime} changed output after sequence reset for group "
                            f"{sequence['row_group']}"
                        )
                    samples.append(record)
                    predictions.append(prediction)
                    progress.write(json.dumps(record, allow_nan=False) + "\n")
                    progress.flush()
                    print(json.dumps(record), flush=True)
                comparisons.append(
                    {
                        "row_group": sequence["row_group"],
                        "seq_ix": sequence["sequence"],
                        "cycle": cycle,
                        "cpp_first_vs_rust_first": parity_summary(predictions[0], predictions[1]),
                        "cpp_second_vs_rust_second": parity_summary(predictions[3], predictions[2]),
                        "cpp_repeat": parity_summary(predictions[0], predictions[3]),
                        "rust_repeat": parity_summary(predictions[1], predictions[2]),
                    }
                )
                del predictions

    data_stat_after = args.data.stat()
    stable_data_identity = (
        data_stat_before.st_ino,
        data_stat_before.st_size,
        data_stat_before.st_mtime_ns,
        data_stat_before.st_ctime_ns,
    ) == (
        data_stat_after.st_ino,
        data_stat_after.st_size,
        data_stat_after.st_mtime_ns,
        data_stat_after.st_ctime_ns,
    )
    if not stable_data_identity:
        raise ValueError("data file stat identity changed while the benchmark was running")
    if digest(worker_path) != worker_sha256:
        raise ValueError("benchmark worker changed while running")
    if digest(args.campaign / "score_archive.py") != campaign_source_sha256:
        raise ValueError("campaign replay source changed while running")
    if digest(inventory_path) != inventory_sha256:
        raise ValueError("campaign inventory changed while running")
    if digest(utils_path) != utils_sha256:
        raise ValueError("campaign callback utility source changed while running")
    result = {
        "status": "complete",
        "candidate": "pair_ftt2",
        "members": actual_members,
        "coefficients": expected_coefficients,
        "command": sys.argv,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign_path": str(args.campaign.resolve()),
        "campaign_source_sha256": campaign_source_sha256,
        "benchmark_worker_path": str(worker_path),
        "benchmark_worker_sha256": worker_sha256,
        "campaign_utils_path": str(utils_path.resolve()),
        "campaign_utils_sha256": utils_sha256,
        "campaign_admission": admission,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "package_archive_sha256": inventory_pair["sha256"],
        "pair_package_source_sha256": inventory_pair["files"]["solution.py"]["sha256"],
        "pair_members_sha256": inventory_pair["files"]["members.json"]["sha256"],
        "pair_package_path": inventory_pair["package"],
        "pair_package_source_path": str(Path(inventory_pair["package"]) / "solution.py"),
        "pair_members_path": str(Path(inventory_pair["package"]) / "members.json"),
        "rust_runtime_adapter": str((args.campaign / "score_archive.py").resolve()),
        "rust_precision_file_sha256": inventory_rust["files"]["precision.txt"]["sha256"],
        "rust_precision_file_path": str(Path(inventory_rust["package"]) / "precision.txt"),
        "rust_precision_file": "q8vnni",
        "weights": {
            member["weights"]: {
                "path": str(Path(inventory_pair["package"]) / member["weights"]),
                "sha256": inventory_pair["files"][member["weights"]]["sha256"],
            }
            for member in actual_members
        },
        "native_runtimes": runtime_identities,
        "data": {
            "path": str(args.data.resolve()),
            "bytes": data_stat_after.st_size,
            "sha256": data_sha256,
            "sha256_preflight_count": 1,
            "stable_stat_during_benchmark": stable_data_identity,
        },
        "groups": [
            {
                key: value
                for key, value in sample.items()
                if key in ("row_group", "sequence", "feature_sha256", "need_sha256")
            }
            for sample in sequences
        ],
        "schedule": [
            {
                "row_group": sequence["row_group"],
                "seq_ix": sequence["sequence"],
                "cycle": cycle,
                "phase": phase,
                "runtime": runtime,
            }
            for sequence in sequences
            for cycle, phase, runtime in schedule
        ],
        "warmup_records_unmeasured": warmup_records,
        "samples": samples,
        "paired_output_differences": comparisons,
        "cpu": cpu,
        "load_average_start": load_start,
        "load_average_end": _load_average(),
        "process_time_seconds": time.process_time() - started_process,
        "perf_counter_seconds": time.perf_counter() - started_wall,
        "timing_scope": (
            "score_archive.replay full-sequence callback including per-row DataPoint creation and "
            "model.predict; excludes data/model loading and the pre-timing reset sentinel."
        ),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": sys.version,
            "numpy": __import__("numpy").__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "scope": (
            "Two complete supplied validation sequences, same four pair_ftt2 NPZs and fixed "
            "coefficients; callback timing only. Output differences are precision/arithmetic "
            "treatment evidence, not an optimization or quality result. Host-local only; no EPYC claim."
        ),
    }
    result_path = args.out / "RESULT.json"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(result_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cycles": args.cycles,
                "samples": len(samples),
                "perf_counter_seconds": result["perf_counter_seconds"],
                "process_time_seconds": result["process_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
