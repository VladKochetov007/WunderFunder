"""Randomized, paired, full-sequence CPU panel for the frozen xteam_h4 Q8 policies.

The parent controls the order and launches one persistent, single-CPU worker per
package. Workers perform the same row-wise ``solution.PredictionModel.predict``
callback used by the competition adapter. Only NumPy and the package's native
extension are needed on the benchmark host; the fixture contains no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

POLICY_BY_LABEL = {
    "A": "q8vnni",
    "B": "q8vnni256",
    "C": "q8vnni_blocked",
    "D": "q8vnni_fused",
}
LABELS = tuple(POLICY_BY_LABEL)
CANDIDATES = LABELS[1:]
N_ROWS = 20_000
N_FEATURES = 112
N_OUTPUTS = 2
WARMUP_ROWS = 99
RESET_SEQUENCE = -987_654_321
DEFAULT_SEED = 20261015
DEFAULT_MIN_BLOCKS = 6
DEFAULT_MAX_BLOCKS = 12
DEFAULT_TIME_BUDGET_SECONDS = 1_000.0
EXPECTED_FIXTURE_MANIFEST_SHA256 = (
    "941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05"
)
EXPECTED_PACKAGE_BINDING_SHA256 = "61a11ab4ea6bf89df3f2789ae0a64db6e5064c12e1d554150ea55f118a0f2367"
EXPECTED_SOURCE_IDENTITY_SHA256 = "956724a5f1e695ffdd33d7b7629ec3ae06abf9c647465e2d531574754741d9ad"
EXPECTED_SAMPLER_MANIFEST_SHA256 = (
    "c4f597956172d65eec9da579bfa4011b1349b58632de8b2eecf23dcbbfbf2041"
)
COMPETITION_DATAPOINT_SOURCE_SHA256 = (
    "e8e1034d61be359ebe943015e4e6bd2f76716303eff520857b7d07eeeb6144a6"
)
_OUTPUT_CREATED = False
T95 = {6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}


@dataclass(frozen=True)
class CallbackPoint:
    """Attribute-compatible copy of the competition ``utils.DataPoint`` value."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: object


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sha256_array(array: Any) -> str:
    """Hash a contiguous NumPy array without creating a second payload copy."""
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def block_schedule(seed: int, block: int, slots: int = 8) -> list[dict[str, object]]:
    """Return one block with a six-block balanced B/C/D permutation cycle."""
    if slots < 1:
        raise ValueError("slots must be positive")
    rng = random.Random(seed ^ ((block + 1) * 0x9E3779B1))
    slot_order = list(range(slots))
    rng.shuffle(slot_order)
    schedule: list[dict[str, object]] = []
    for slot in slot_order:
        cycle = block // math.factorial(len(CANDIDATES))
        position = block % math.factorial(len(CANDIDATES))
        order_rng = random.Random(seed ^ (slot * 0x85EBCA6B) ^ (cycle * 0xC2B2AE35))
        candidate_orders = list(itertools.permutations(CANDIDATES))
        order_rng.shuffle(candidate_orders)
        candidates = candidate_orders[position]
        schedule.extend(
            [
                {"slot": slot, "label": "A", "control": "first"},
                *[{"slot": slot, "label": label, "control": None} for label in candidates],
                {"slot": slot, "label": "A", "control": "last"},
            ]
        )
    return schedule


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize repeated measurements with sample variation."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("timing samples must be finite and nonempty")
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_log_ci(log_ratios: list[float]) -> dict[str, float | int]:
    """Compute a descriptive paired t interval for block-mean log ratios."""
    n = len(log_ratios)
    if n < DEFAULT_MIN_BLOCKS or n > DEFAULT_MAX_BLOCKS:
        raise ValueError("paired interval needs 6 through 12 completed blocks")
    center = mean(log_ratios)
    half_log = T95[n] * stdev(log_ratios) / math.sqrt(n) if n > 1 else 0.0
    return {
        "n_blocks": n,
        "geometric_mean_ratio": math.exp(center),
        "relative_change_pct": (math.exp(center) - 1.0) * 100.0,
        "ci95_low_pct": (math.exp(center - half_log) - 1.0) * 100.0,
        "ci95_high_pct": (math.exp(center + half_log) - 1.0) * 100.0,
        "ci95_halfwidth_pct": max(
            (math.exp(center + half_log) - math.exp(center)) * 100.0,
            (math.exp(center) - math.exp(center - half_log)) * 100.0,
        ),
    }


def planned_blocks(
    pilot_seconds: list[float],
    time_budget_seconds: float,
    fixed_overhead_seconds: float,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    safety_factor: float = 1.35,
) -> int:
    """Choose an a-priori block cap from a five-call complete-sequence pilot."""
    if len(pilot_seconds) != 5 or any(value <= 0 for value in pilot_seconds):
        raise ValueError("pilot must contain the five A/B/C/D/A full callbacks")
    if min_blocks < 6 or max_blocks > 12 or min_blocks > max_blocks:
        raise ValueError("block bounds must satisfy 6 <= minimum <= maximum <= 12")
    per_slot = sum(pilot_seconds) / len(pilot_seconds)
    estimated_block = 8 * len(pilot_seconds) * per_slot * safety_factor
    available = time_budget_seconds - fixed_overhead_seconds
    budget_blocks = int(available // estimated_block)
    if budget_blocks < min_blocks:
        return 0
    return min(max_blocks, budget_blocks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--package-binding", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-blocks", type=int, default=DEFAULT_MIN_BLOCKS)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--precision-halfwidth-pct", type=float, default=1.0)
    parser.add_argument("--time-budget-s", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-pair", action="store_true")
    parser.add_argument("--smoke-slot", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", choices=LABELS)
    parser.add_argument("--worker-package", type=Path)
    parser.add_argument("--worker-fixture", type=Path)
    parser.add_argument("--worker-fixture-manifest", type=Path)
    parser.add_argument("--worker-manifest", type=Path)
    parser.add_argument("--worker-cpu", type=int)
    return parser.parse_args()


def _read_fixture_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate the frozen metadata schema before mapping any input bytes."""
    if sha256_file(manifest_path) != EXPECTED_FIXTURE_MANIFEST_SHA256:
        raise ValueError("fixture manifest SHA-256 differs from the producer-frozen manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "wnn.vast-cpu-sequence-panel" or manifest.get("version") != 1:
        raise ValueError("unknown CPU fixture format")
    source = manifest.get("source", {})
    if source.get("fold") != 0 or source.get("seed") != 3:
        raise ValueError("fixture must be the source-qualified F0 seed-3 training panel")
    expected_scope = (
        "frozen fold-0 Parquet footer SHA-256, file byte length and mtime_ns; "
        "the 29GB parquet body was not rescanned"
    )
    if source.get("source_identity_scope") != expected_scope:
        raise ValueError("fixture source identity must use the frozen footer/stat binding")
    identity = {
        key: source[key]
        for key in ("dataset_path", "bytes", "mtime_ns", "footer_bytes", "footer_sha256")
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    expected_identity = hashlib.sha256(canonical).hexdigest()
    if source.get("source_identity_sha256") != expected_identity:
        raise ValueError("canonical source_identity_sha256 mismatch")
    if expected_identity != EXPECTED_SOURCE_IDENTITY_SHA256:
        raise ValueError("fixture source identity differs from the source-qualified F0 producer")
    if source.get("sampler_manifest_sha256") != EXPECTED_SAMPLER_MANIFEST_SHA256:
        raise ValueError("fixture does not bind the frozen train sampler membership")
    arrays = manifest.get("arrays", {})
    features = arrays.get("features", {})
    need = arrays.get("need_prediction", {})
    if (features.get("file"), features.get("dtype"), features.get("shape")) != (
        "features.bin",
        "<f4",
        [8, N_ROWS, N_FEATURES],
    ):
        raise ValueError("raw feature fixture must be little-endian float32 [8,20000,112]")
    if (need.get("file"), need.get("dtype"), need.get("shape")) != (
        "need_prediction.bin",
        "|u1",
        [8, N_ROWS],
    ):
        raise ValueError("request vector must be uint8 [8,20000]")
    if features.get("bytes") != 8 * N_ROWS * N_FEATURES * 4 or need.get("bytes") != 8 * N_ROWS:
        raise ValueError("fixture byte lengths do not match the declared fixed shapes")
    request_schedule = manifest.get("request_schedule", {})
    if request_schedule.get("kind") != "fixed runtime request vector, not an official scoring mask":
        raise ValueError("need_prediction must be explicitly identified as runtime-only")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 8:
        raise ValueError("fixture must contain eight source-qualified sequences")
    if [item.get("slot") for item in sequences] != list(range(8)):
        raise ValueError("fixture slots must be ordered 0 through 7")
    if len({item.get("component_id") for item in sequences}) != 8:
        raise ValueError("fixture slots must cover eight distinct fitting components")
    if RESET_SEQUENCE in {item.get("sequence_id") for item in sequences}:
        raise ValueError("reset sentinel collides with a fixture sequence ID")
    excluded = set(manifest.get("excluded_payloads", []))
    required_excluded = {
        "targets",
        "official_scoring_mask",
        "teacher_outputs",
        "distillation_cache_values",
        "propensity_values",
    }
    if not required_excluded.issubset(excluded):
        raise ValueError("fixture manifest does not declare the excluded labels/masks/caches")
    for forbidden in ("targets", "scoring_mask", "teacher_cache", "labels"):
        if forbidden in manifest:
            raise ValueError(f"fixture manifest unexpectedly contains {forbidden}")
    return manifest


def _validate_fixture_payload(fixture_dir: Path, manifest: dict[str, Any]) -> tuple[Any, Any]:
    """Hash and memory-map the compact eight-sequence, label-free input fixture."""
    import numpy as np

    features_spec = manifest["arrays"]["features"]
    need_spec = manifest["arrays"]["need_prediction"]
    features_path = fixture_dir / features_spec["file"]
    need_path = fixture_dir / need_spec["file"]
    if features_path.stat().st_size != 8 * N_ROWS * N_FEATURES * 4:
        raise ValueError("raw feature fixture has the wrong exact byte length")
    if need_path.stat().st_size != 8 * N_ROWS:
        raise ValueError("request vector has the wrong exact byte length")
    if sha256_file(features_path) != features_spec["sha256"]:
        raise ValueError("whole raw-feature fixture SHA-256 mismatch")
    if sha256_file(need_path) != need_spec["sha256"]:
        raise ValueError("whole request-vector SHA-256 mismatch")
    features = np.memmap(features_path, dtype="<f4", mode="r", shape=(8, N_ROWS, N_FEATURES))
    need = np.memmap(need_path, dtype="u1", mode="r", shape=(8, N_ROWS))
    if not np.isfinite(features).all() or not np.isin(need, (0, 1)).all():
        raise ValueError("feature/request payload contains non-finite or invalid values")
    expected_need = np.arange(N_ROWS) >= WARMUP_ROWS
    for slot, item in enumerate(manifest["sequences"]):
        if sha256_array(features[slot]) != item["feature_sha256"]:
            raise ValueError(f"feature hash mismatch in fixture slot {slot}")
        if sha256_array(need[slot]) != item["need_prediction_sha256"]:
            raise ValueError(f"request hash mismatch in fixture slot {slot}")
        if not np.array_equal(need[slot].astype(bool), expected_need):
            raise ValueError(f"request schedule is not the fixed first-99 warmup for slot {slot}")
    return features, need


def _verify_package_root(package_root: Path, binding_path: Path) -> dict[str, Any]:
    """Check extracted package files and fixed weights before any model import."""
    if sha256_file(binding_path) != EXPECTED_PACKAGE_BINDING_SHA256:
        raise ValueError("package binding differs from the frozen A/B/C/D package inventory")
    binding = json.loads(binding_path.read_text())
    if binding.get("format") != "wnn.cpu-policy-package-binding" or binding.get("version") != 1:
        raise ValueError("unknown policy package binding format")
    if (
        binding.get("source_archive_sha256")
        != "f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27"
    ):
        raise ValueError("package binding does not point to original xteam_h4")
    for label in LABELS:
        entry = binding["packages"][label]
        package = package_root / entry["directory"]
        if (package / "precision.txt").read_text().strip() != POLICY_BY_LABEL[label]:
            raise ValueError(f"package {label} does not assert the expected policy")
        expected_files = entry["files"]
        actual_files = {
            path.relative_to(package).as_posix()
            for path in package.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if actual_files != set(expected_files):
            raise ValueError(f"package {label} file inventory differs from the frozen binding")
        for relative, expected_sha in expected_files.items():
            if sha256_file(package / relative) != expected_sha:
                raise ValueError(f"package {label} hash mismatch: {relative}")
        if entry["archive_sha256"] != binding["package_archive_sha256"][label]:
            raise ValueError(f"package {label} archive hash does not match campaign manifest")
        for relative, fixed_sha in binding["fixed_weight_and_member_sha256"].items():
            if expected_files.get(relative) != fixed_sha:
                raise ValueError(f"package binding does not hold fixed weights/members in {label}")
    return binding


def _cpu_identity(requested_cpu: int | None) -> dict[str, object]:
    """Pin this process to one allowed logical CPU and report its NUMA node."""
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("the portable CPU packet requires Linux CPU-affinity support")
    before = sorted(os.sched_getaffinity(0))
    selected = requested_cpu if requested_cpu is not None else min(before)
    if selected not in before:
        raise ValueError(f"CPU {selected} is outside the currently allowed affinity")
    os.sched_setaffinity(0, {selected})
    nodes = sorted(
        int(path.name.removeprefix("node"))
        for path in Path(f"/sys/devices/system/cpu/cpu{selected}").glob("node[0-9]*")
    )
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip() == "processor":
                current["processor"] = int(value.strip())
            elif key.strip() in ("model name", "flags"):
                current[key.strip()] = value.strip()
    except OSError:
        records = []
    selected_record = next(
        (record for record in records if record.get("processor") == selected),
        records[0] if records else {},
    )
    return {
        "requested_cpu": requested_cpu,
        "selected_cpu": selected,
        "affinity_before": before,
        "affinity_after": sorted(os.sched_getaffinity(0)),
        "numa_nodes_for_cpu": nodes,
        "cpu_model": selected_record.get("model name", platform.processor()),
        "cpu_flags": selected_record.get("flags"),
        "platform": platform.platform(),
        "python": sys.version,
    }


def _cgroup_snapshot() -> dict[str, object]:
    """Read available cgroup limits and throttling counters without changing them."""
    result: dict[str, object] = {"available": False}
    try:
        cgroup_line = next(
            line
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        path = Path("/sys/fs/cgroup") / cgroup_line[3:].lstrip("/")
    except (OSError, StopIteration):
        return result
    if not path.is_dir():
        return result
    result = {"available": True, "path": str(path)}
    for name in (
        "cpu.max",
        "memory.current",
        "memory.max",
        "memory.swap.current",
        "memory.swap.max",
    ):
        file_path = path / name
        if file_path.exists():
            result[name] = file_path.read_text().strip()
    for name in ("cpu.stat", "memory.events"):
        file_path = path / name
        if file_path.exists():
            result[name] = {
                key: int(value)
                for key, value in (line.split() for line in file_path.read_text().splitlines())
            }
    return result


def _set_single_thread_environment() -> None:
    """Disable threaded math runtimes before importing NumPy or the extension."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("WNN_PRECISION", None)
    os.environ.pop("WNN_NATIVE_LEVEL", None)


def _load_package_model(
    package: Path, label: str, expected_native_hashes: dict[str, str]
) -> tuple[Any, dict[str, object]]:
    """Import the unchanged competition solution and assert its selected native policy."""
    import numpy as np

    expected_policy = POLICY_BY_LABEL[label]
    if (package / "precision.txt").read_text().strip() != expected_policy:
        raise ValueError(f"package {label} precision assertion changed")
    module_path = package / "solution.py"
    spec = importlib.util.spec_from_file_location(f"cpu_policy_{label}_{os.getpid()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import the frozen solution in {package}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructor_start = time.perf_counter()
    constructor_cpu_start = time.process_time()
    model = module.PredictionModel()
    constructor_seconds = time.perf_counter() - constructor_start
    constructor_process_seconds = time.process_time() - constructor_cpu_start
    if model.model.precision != expected_policy:
        raise AssertionError(
            f"selected-policy assertion failed: expected {expected_policy}, got {model.model.precision}"
        )
    native_module = sys.modules.get("wnn_native")
    if native_module is None or not getattr(native_module, "__file__", None):
        raise RuntimeError("original package callback did not load its native module")
    native_path = Path(native_module.__file__).resolve()
    if not native_path.is_relative_to(package.resolve()):
        raise RuntimeError("loaded native library is outside the selected package")
    native_sha = sha256_file(native_path)
    native_relative = native_path.relative_to(package.resolve()).as_posix()
    if expected_native_hashes.get(native_relative) != native_sha:
        raise ValueError(f"loaded {label} native library is not package-inventory bound")
    return model, {
        "label": label,
        "selected_policy": model.model.precision,
        "native_level": getattr(module, "_LEVEL", None),
        "constructor_seconds_including_weight_load_and_pack": constructor_seconds,
        "constructor_process_seconds": constructor_process_seconds,
        "native_library": str(native_path),
        "native_library_sha256": native_sha,
        "package_solution_sha256": sha256_file(module_path),
        "max_rss_kib_after_load": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "numpy": np.__version__,
    }


def _replay_callback(model: Any, sequence: int, features: Any, need: Any) -> Any:
    """Replay the full competition row callback, including each model.predict call."""
    import numpy as np

    prediction = np.zeros((N_ROWS, N_OUTPUTS), dtype=np.float32)
    for step, row in enumerate(features):
        request = bool(need[step])
        value = model.predict(CallbackPoint(sequence, step, request, row))
        if not request:
            if value is not None:
                raise ValueError(f"warmup row {step} unexpectedly produced a prediction")
        elif value is None or np.shape(value) != (N_OUTPUTS,):
            raise ValueError(f"requested row {step} did not produce two predictions")
        else:
            prediction[step] = value
    if not np.isfinite(prediction).all():
        raise ValueError("full callback returned non-finite predictions")
    return prediction


def _reset_replay(model: Any, features: Any) -> None:
    """Exercise the native sequence-change reset path outside the timed callback."""

    for step, row in enumerate(features[:WARMUP_ROWS]):
        value = model.predict(CallbackPoint(RESET_SEQUENCE, step, False, row))
        if value is not None:
            raise ValueError("reset sentinel unexpectedly returned a prediction")


def _worker_main(args: argparse.Namespace) -> None:
    """Serve serialized complete-sequence requests over JSON lines."""
    _set_single_thread_environment()
    started = time.perf_counter()
    if not hasattr(os, "sched_getaffinity") or args.worker_cpu not in os.sched_getaffinity(0):
        raise RuntimeError("worker did not inherit the selected one-CPU affinity")
    os.sched_setaffinity(0, {args.worker_cpu})
    import numpy as np

    manifest = _read_fixture_manifest(args.worker_fixture_manifest)
    feature_spec = manifest["arrays"]["features"]
    need_spec = manifest["arrays"]["need_prediction"]
    features = np.memmap(
        args.worker_fixture / feature_spec["file"],
        dtype="<f4",
        mode="r",
        shape=(8, N_ROWS, N_FEATURES),
    )
    need = np.memmap(
        args.worker_fixture / need_spec["file"], dtype="u1", mode="r", shape=(8, N_ROWS)
    )
    binding = json.loads(args.worker_manifest.read_text())
    file_hashes = binding["packages"][args.worker_label]["files"]
    native_hashes = {
        name: value for name, value in file_hashes.items() if name.endswith("/wnn_native.abi3.so")
    }
    expected_native_hashes = native_hashes
    model, identity = _load_package_model(
        args.worker_package, args.worker_label, expected_native_hashes
    )
    ready = {
        "event": "ready",
        "worker_label": args.worker_label,
        "pid": os.getpid(),
        "startup_seconds": time.perf_counter() - started,
        **identity,
    }
    print(json.dumps(ready, allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "exit":
            print(json.dumps({"event": "bye", "worker_label": args.worker_label}), flush=True)
            return
        if request.get("op") != "run":
            raise ValueError("unknown worker operation")
        slot = int(request["slot"])
        if slot < 0 or slot >= 8:
            raise ValueError("worker sequence slot is outside [0,7]")
        sequence = int(manifest["sequences"][slot]["sequence_id"])
        reset_wall_start = time.perf_counter()
        reset_cpu_start = time.process_time()
        _reset_replay(model, features[slot])
        reset_wall_seconds = time.perf_counter() - reset_wall_start
        reset_process_seconds = time.process_time() - reset_cpu_start
        callback_cpu_start = time.process_time()
        callback_wall_start = time.perf_counter()
        prediction = _replay_callback(model, sequence, features[slot], need[slot])
        callback_wall_seconds = time.perf_counter() - callback_wall_start
        callback_cpu_seconds = time.process_time() - callback_cpu_start
        output_path = Path(request["output_path"])
        with output_path.open("xb") as sink:
            sink.write(memoryview(prediction).cast("B"))
        response = {
            "event": "sample",
            "worker_label": args.worker_label,
            "block": int(request["block"]),
            "slot": slot,
            "sequence_id": sequence,
            "request_id": request["request_id"],
            "output_sha256": sha256_array(prediction),
            "callback_wall_seconds": callback_wall_seconds,
            "callback_process_seconds": callback_cpu_seconds,
            "reset_wall_seconds": reset_wall_seconds,
            "reset_process_seconds": reset_process_seconds,
            "output_bytes": prediction.nbytes,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        print(json.dumps(response, allow_nan=False), flush=True)


def _read_response(process: subprocess.Popen[str]) -> dict[str, Any]:
    """Read exactly one worker protocol record or expose a worker failure."""
    if process.stdout is None:
        raise RuntimeError("worker stdout is unavailable")
    line = process.stdout.readline()
    if not line:
        return_code = process.poll()
        raise RuntimeError(f"worker exited before replying (status={return_code})")
    return json.loads(line)


def _start_worker(
    label: str,
    package: Path,
    fixture_dir: Path,
    fixture_manifest: Path,
    binding_path: Path,
    cpu: int,
    output_dir: Path,
) -> tuple[subprocess.Popen[str], dict[str, Any], Any]:
    """Launch a package-specific worker and isolate its cold-start receipt."""
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("WNN_PRECISION", None)
    env.pop("WNN_NATIVE_LEVEL", None)
    stderr_log = (output_dir / f"worker_{label}.stderr.log").open("x", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-label",
        label,
        "--worker-package",
        str(package.resolve()),
        "--worker-fixture",
        str(fixture_dir.resolve()),
        "--worker-fixture-manifest",
        str(fixture_manifest.resolve()),
        "--worker-manifest",
        str(binding_path.resolve()),
        "--worker-cpu",
        str(cpu),
    ]
    wall_start = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
            bufsize=1,
            env=env,
        )
        ready = _read_response(process)
        if ready.get("event") != "ready" or ready.get("worker_label") != label:
            raise RuntimeError(f"invalid worker-ready record: {ready}")
        ready["parent_spawn_to_ready_seconds"] = time.perf_counter() - wall_start
        return process, ready, stderr_log
    except BaseException:
        if "process" in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stderr_log.close()
        raise


def _request_sample(
    process: subprocess.Popen[str],
    label: str,
    block: int,
    slot: int,
    request_id: str,
    output_dir: Path,
) -> tuple[dict[str, Any], Any]:
    """Run one reset-and-replay request and read its transient prediction array."""
    import numpy as np

    if process.stdin is None or process.stdout is None:
        raise RuntimeError("worker process pipes are unavailable")
    output_path = output_dir / f"transient_{request_id}.f32"
    request = {
        "op": "run",
        "label": label,
        "block": block,
        "slot": slot,
        "request_id": request_id,
        "output_path": str(output_path),
    }
    roundtrip_start = time.perf_counter()
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    response = _read_response(process)
    response["worker_roundtrip_seconds"] = time.perf_counter() - roundtrip_start
    if response.get("event") != "sample" or response.get("request_id") != request_id:
        raise RuntimeError(f"worker sample response does not match its request: {response}")
    if output_path.stat().st_size != N_ROWS * N_OUTPUTS * 4:
        raise ValueError("worker output has an incorrect byte length")
    predictions = np.fromfile(output_path, dtype="<f4").reshape(N_ROWS, N_OUTPUTS)
    if sha256_array(predictions) != response["output_sha256"]:
        raise ValueError("worker output hash differs from its receipt")
    output_path.unlink()
    return response, predictions


def _stop_workers(workers: dict[str, subprocess.Popen[str]], worker_logs: list[Any]) -> None:
    """Ask live workers to exit, then close pipes and logs even after an error."""
    for process in workers.values():
        if process.poll() is not None:
            continue
        try:
            if process.stdin is not None:
                process.stdin.write('{"op":"exit"}\n')
                process.stdin.flush()
            if process.stdout is not None:
                _read_response(process)
            process.wait(timeout=10)
        except (BrokenPipeError, OSError, RuntimeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
    for log in worker_logs:
        log.close()


def _run_smoke_pair(args: argparse.Namespace) -> None:
    """Verify fixture transport, package selection, reset and full A/B callback parity."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("the frozen original packages require Python 3.11")
    required = (
        args.output,
        args.package_root,
        args.package_binding,
        args.fixture_dir,
        args.fixture_manifest,
    )
    if any(value is None for value in required):
        raise ValueError(
            "smoke pair requires package root/binding, fixture directory/manifest and output"
        )
    if not 0 <= args.smoke_slot < 8:
        raise ValueError("smoke slot must be in [0,7]")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    global _OUTPUT_CREATED
    _OUTPUT_CREATED = True
    _set_single_thread_environment()
    host = _cpu_identity(args.cpu)
    import numpy as np

    fixture_check_start = time.perf_counter()
    manifest = _read_fixture_manifest(args.fixture_manifest)
    features, need = _validate_fixture_payload(args.fixture_dir, manifest)
    fixture_check_seconds = time.perf_counter() - fixture_check_start
    package_check_start = time.perf_counter()
    binding = _verify_package_root(args.package_root, args.package_binding)
    package_check_seconds = time.perf_counter() - package_check_start
    workers: dict[str, subprocess.Popen[str]] = {}
    worker_logs: list[Any] = []
    worker_ready: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    predictions: dict[str, list[Any]] = {"A": [], "B": []}
    slot = args.smoke_slot
    try:
        for label in ("A", "B"):
            process, ready, log = _start_worker(
                label,
                args.package_root / binding["packages"][label]["directory"],
                args.fixture_dir,
                args.fixture_manifest,
                args.package_binding,
                int(host["selected_cpu"]),
                args.output,
            )
            workers[label] = process
            worker_ready[label] = ready
            worker_logs.append(log)

        for label in ("A", "B"):
            for repetition in range(2):
                request_id = f"smoke_{label}_r{repetition}_s{slot}"
                response, values = _request_sample(
                    workers[label], label, -3, slot, request_id, args.output
                )
                response.update(
                    {
                        "phase": "compatibility_smoke",
                        "cpu_policy": POLICY_BY_LABEL[label],
                        "input_sha256": manifest["sequences"][slot]["feature_sha256"],
                        "need_prediction_sha256": manifest["sequences"][slot][
                            "need_prediction_sha256"
                        ],
                        "reset_before_callback": True,
                    }
                )
                samples[label].append(response)
                predictions[label].append(values)
            if not np.array_equal(predictions[label][0], predictions[label][1]):
                raise ValueError(f"{label} callback changed across complete reset replays")

        delta = _prediction_delta(predictions["A"][0], predictions["B"][0])
        result = {
            "format": "wnn.cpu-policy-smoke-pair",
            "version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "source_commit": os.environ.get("WNN_BENCHMARK_SOURCE_COMMIT", "unset"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "command": sys.argv,
            "host": host,
            "single_cpu_affinity_only": True,
            "cpu_quota_asserted": False,
            "cgroup": _cgroup_snapshot(),
            "fixture_manifest_path": str(args.fixture_manifest.resolve()),
            "fixture_manifest_sha256": sha256_file(args.fixture_manifest),
            "fixture_hash_and_map_seconds": fixture_check_seconds,
            "fixture_arrays": manifest["arrays"],
            "source": manifest["source"],
            "sequence": manifest["sequences"][slot],
            "datapoint_contract": {
                "source_path": "utils.py",
                "source_sha256": COMPETITION_DATAPOINT_SOURCE_SHA256,
                "fields_in_order": ["seq_ix", "step_in_seq", "need_prediction", "state"],
                "adapter_imports_pyarrow": False,
            },
            "package_binding_sha256": sha256_file(args.package_binding),
            "package_verification_seconds": package_check_seconds,
            "worker_cold_start_and_constructor": worker_ready,
            "samples": samples,
            "same_policy_complete_reset_replay_bitwise_equal": {"A": True, "B": True},
            "A_vs_B": delta,
            "callback_scope": (
                "Two full 20,000-row solution.PredictionModel.predict replays per policy from raw112 "
                "features using the synthetic runtime request vector; each replay follows a 99-row "
                "sentinel reset. Includes per-row DataPoint creation and native preprocessing/inference."
            ),
            "interpretation": (
                "Compatibility smoke only; one source-qualified sequence does not establish a timing "
                "ranking, performance variance or a multi-sequence parity claim."
            ),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "numpy": np.__version__,
            "python": sys.version,
        }
        (args.output / "SMOKE_PAIR_R1.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    finally:
        _stop_workers(workers, worker_logs)


def _pairwise_interval(
    records: list[dict[str, Any]], left: str, right: str, time_key: str
) -> dict[str, Any]:
    """Compare policies using one mean log-ratio per paired block and sequence."""
    by_block: dict[int, dict[str, dict[int, list[float]]]] = {}
    for record in records:
        label = record["label"]
        if label not in (left, right):
            continue
        block = int(record["block"])
        slot = int(record["slot"])
        by_block.setdefault(block, {left: {}, right: {}})[label].setdefault(slot, []).append(
            float(record[time_key])
        )
    log_ratios: list[float] = []
    per_block: list[dict[str, object]] = []
    for block in sorted(by_block):
        pair_logs: list[float] = []
        for slot in sorted(set(by_block[block][left]) & set(by_block[block][right])):
            lvalue = mean(by_block[block][left][slot])
            rvalue = mean(by_block[block][right][slot])
            if lvalue <= 0 or rvalue <= 0:
                raise ValueError("callback timings must be positive")
            pair_logs.append(math.log(lvalue / rvalue))
        if len(pair_logs) != 8:
            raise ValueError(f"block {block} does not have eight paired sequence timings")
        log_ratios.append(mean(pair_logs))
        per_block.append({"block": block, "log_ratio_left_over_right": mean(pair_logs)})
    return {
        **paired_log_ci(log_ratios),
        "left": left,
        "right": right,
        "time_key": time_key,
        "blocks": per_block,
    }


def _compute_results(
    records: list[dict[str, Any]], min_blocks: int, max_blocks: int
) -> dict[str, Any]:
    """Summarize per-policy costs and paired panel-level uncertainty."""
    completed_blocks = sorted(
        {int(record["block"]) for record in records if int(record["block"]) >= 0}
    )
    if not (min_blocks <= len(completed_blocks) <= max_blocks):
        raise ValueError("timed panel block count is outside the admitted bounds")
    by_label = {
        label: [
            float(record["callback_wall_seconds"]) for record in records if record["label"] == label
        ]
        for label in LABELS
    }
    by_label_cpu = {
        label: [
            float(record["callback_process_seconds"])
            for record in records
            if record["label"] == label
        ]
        for label in LABELS
    }
    pairwise = {
        f"{left}_over_{right}": {
            "wall": _pairwise_interval(records, left, right, "callback_wall_seconds"),
            "process": _pairwise_interval(records, left, right, "callback_process_seconds"),
        }
        for left in LABELS
        for right in LABELS
        if left != right
        and (left, right)
        in (("B", "A"), ("C", "A"), ("D", "A"), ("B", "C"), ("B", "D"), ("C", "D"))
    }
    timing = {
        label: {
            "wall_seconds": summarize(values),
            "process_seconds": summarize(by_label_cpu[label]),
        }
        for label, values in by_label.items()
    }
    per_sequence: dict[str, Any] = {}
    for label in CANDIDATES:
        per_sequence[label] = {}
        for slot in range(8):
            candidate = [
                float(record["callback_wall_seconds"])
                for record in records
                if record["label"] == label and int(record["slot"]) == slot
            ]
            controls = [
                float(record["callback_wall_seconds"])
                for record in records
                if record["label"] == "A" and int(record["slot"]) == slot
            ]
            paired = min(len(candidate), len(controls) // 2)
            log_changes = []
            for block in range(len(completed_blocks)):
                candidate_record = next(
                    record
                    for record in records
                    if record["label"] == label
                    and int(record["slot"]) == slot
                    and int(record["block"]) == block
                )
                controls_in_block = [
                    float(record["callback_wall_seconds"])
                    for record in records
                    if record["label"] == "A"
                    and int(record["slot"]) == slot
                    and int(record["block"]) == block
                ]
                log_changes.append(
                    math.log(
                        float(candidate_record["callback_wall_seconds"]) / mean(controls_in_block)
                    )
                )
            per_sequence[label][str(slot)] = {
                "paired_repetitions": paired,
                "candidate_wall": summarize(candidate),
                "A_control_wall": summarize(controls),
                "log_ratio_candidate_over_bracketed_A_by_block": log_changes,
            }
    between_sequence_policy_means = {
        label: summarize(
            [
                mean(
                    float(record["callback_wall_seconds"])
                    for record in records
                    if record["label"] == label and int(record["slot"]) == slot
                )
                for slot in range(8)
            ]
        )
        for label in LABELS
    }
    return {
        "completed_blocks": len(completed_blocks),
        "timing_by_policy": timing,
        "pairwise_block_ci95": pairwise,
        "within_sequence_repeat_variation": per_sequence,
        "between_sequence_mean_wall_variation": between_sequence_policy_means,
    }


def _run_panel(args: argparse.Namespace) -> None:
    """Run the balanced, time-bounded panel on already staged packages and fixture."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("the frozen original packages require Python 3.11")
    if any(
        value is None
        for value in (
            args.output,
            args.package_root,
            args.package_binding,
            args.fixture_dir,
            args.fixture_manifest,
        )
    ):
        raise ValueError(
            "package root, binding, fixture directory, fixture manifest and output are required"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    global _OUTPUT_CREATED
    _OUTPUT_CREATED = True
    if args.min_blocks < 6 or args.max_blocks > 12 or args.min_blocks > args.max_blocks:
        raise ValueError("block bounds must satisfy 6 <= minimum <= maximum <= 12")
    if args.time_budget_s <= 0 or args.precision_halfwidth_pct <= 0:
        raise ValueError("time budget and interval target must be positive")
    _set_single_thread_environment()
    host = _cpu_identity(args.cpu)
    import numpy as np

    manifest = _read_fixture_manifest(args.fixture_manifest)
    features, need = _validate_fixture_payload(args.fixture_dir, manifest)
    binding = _verify_package_root(args.package_root, args.package_binding)
    start_wall = time.perf_counter()
    start_process = time.process_time()
    load_before = _load_average()
    cgroup_before = _cgroup_snapshot()
    workers: dict[str, subprocess.Popen[str]] = {}
    worker_logs: list[Any] = []
    worker_ready: dict[str, Any] = {}
    try:
        for label in LABELS:
            process, ready, log = _start_worker(
                label,
                args.package_root / binding["packages"][label]["directory"],
                args.fixture_dir,
                args.fixture_manifest,
                args.package_binding,
                int(host["selected_cpu"]),
                args.output,
            )
            workers[label] = process
            worker_logs.append(log)
            worker_ready[label] = ready
        records: list[dict[str, Any]] = []
        predictions: dict[tuple[str, int], Any] = {}
        output_index = 0

        def sample(label: str, slot: int, block: int, phase: str) -> tuple[dict[str, Any], Any]:
            nonlocal output_index
            request_id = f"{output_index:05d}_{label}_b{block}_s{slot}_{phase}"
            output_index += 1
            response, values = _request_sample(
                workers[label], label, block, slot, request_id, args.output
            )
            response["phase"] = phase
            response["cpu_policy"] = POLICY_BY_LABEL[label]
            response["input_sha256"] = manifest["sequences"][slot]["feature_sha256"]
            response["need_prediction_sha256"] = manifest["sequences"][slot][
                "need_prediction_sha256"
            ]
            response["reset_before_callback"] = True
            return response, values

        # Prime all package/sequence pairs. These complete callbacks are not timed evidence.
        warmup_records = []
        for slot in range(8):
            baseline = None
            for label in LABELS:
                response, values = sample(label, slot, -1, "unmeasured_prime")
                previous = predictions.setdefault((label, slot), values.copy())
                stable = bool(np.array_equal(previous, values))
                if not stable:
                    raise ValueError(f"{label} reset replay changed on warmup slot {slot}")
                if label == "A":
                    baseline = values.copy()
                elif baseline is not None:
                    response["bitwise_equal_to_slot_A"] = bool(np.array_equal(baseline, values))
                response["bitwise_repeat_after_reset"] = stable
                warmup_records.append(response)

        # One uncounted full A/B/C/D/A pilot selects an a-priori maximum block count.
        pilot_records = []
        pilot_order = [
            "A",
            *random.Random(args.seed).sample(list(CANDIDATES), len(CANDIDATES)),
            "A",
        ]
        pilot_times = []
        pilot_reference = None
        for label in pilot_order:
            response, values = sample(label, 0, -2, "pilot")
            if pilot_reference is None:
                pilot_reference = values.copy()
            response["bitwise_equal_to_pilot_A"] = bool(np.array_equal(pilot_reference, values))
            response["bitwise_repeat_after_reset"] = bool(
                np.array_equal(predictions[(label, 0)], values)
            )
            pilot_times.append(float(response["callback_wall_seconds"]))
            pilot_records.append(response)
        minimum_panel_estimate = 8 * 5 * args.min_blocks * max(pilot_times) * 1.35
        planned_max = planned_blocks(
            pilot_times,
            args.time_budget_s,
            fixed_overhead_seconds=time.perf_counter() - start_wall,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
        )
        if planned_max == 0:
            raise TimeoutError(
                "pilot does not leave room for six full paired blocks within the time budget"
            )
        if minimum_panel_estimate > args.time_budget_s:
            raise TimeoutError("conservative pilot estimate exceeds the full-panel budget")

        formal_block_records: list[dict[str, Any]] = []
        ci_history: list[dict[str, Any]] = []
        for block in range(planned_max):
            schedule = block_schedule(args.seed, block)
            slot_a_first: dict[int, Any] = {}
            slot_a_last: dict[int, Any] = {}
            current_block: list[dict[str, Any]] = []
            for item in schedule:
                label = str(item["label"])
                slot = int(item["slot"])
                phase = (
                    "A_first"
                    if item["control"] == "first"
                    else ("A_last" if item["control"] == "last" else "candidate")
                )
                response, values = sample(label, slot, block, phase)
                a_reference = predictions[("A", slot)]
                response["bitwise_equal_to_slot_A"] = bool(np.array_equal(a_reference, values))
                previous = predictions[(label, slot)]
                response["bitwise_repeat_after_reset"] = bool(np.array_equal(previous, values))
                response["delta_vs_slot_A"] = _prediction_delta(a_reference, values)
                if item["control"] == "first":
                    slot_a_first[slot] = values.copy()
                elif item["control"] == "last":
                    slot_a_last[slot] = values.copy()
                if not response["bitwise_repeat_after_reset"]:
                    response["same_policy_replay_changed"] = True
                predictions[(label, slot)] = values.copy()
                current_block.append(response)
                formal_block_records.append(response)
                records.append(response)
                if label == "A" and item["control"] == "first":
                    predictions[("A", slot)] = values.copy()
            if set(slot_a_first) != set(range(8)) or set(slot_a_last) != set(range(8)):
                raise ValueError("paired block does not contain two A controls for all eight slots")
            block_path = args.output / "progress.jsonl"
            with block_path.open("a", encoding="utf-8") as progress:
                for response in current_block:
                    progress.write(json.dumps(response, allow_nan=False) + "\n")
            ci = (
                _compute_results(formal_block_records, args.min_blocks, args.max_blocks)
                if block + 1 >= args.min_blocks
                else None
            )
            if ci is not None:
                estimates = ci["pairwise_block_ci95"]
                halfwidths = [
                    float(entry["wall"]["ci95_halfwidth_pct"]) for entry in estimates.values()
                ]
                ci_history.append(
                    {"completed_blocks": block + 1, "comparison_intervals": estimates}
                )
                if max(halfwidths, default=0.0) <= args.precision_halfwidth_pct:
                    break

        elapsed_wall = time.perf_counter() - start_wall
        if elapsed_wall > args.time_budget_s:
            raise TimeoutError("panel exceeded its admitted time budget")
        results = _compute_results(records, args.min_blocks, args.max_blocks)
        results["adaptive_stopping"] = {
            "target_halfwidth_pct": args.precision_halfwidth_pct,
            "blocks_available_from_pilot": planned_max,
            "blocks_completed": results["completed_blocks"],
            "precision_target_met": all(
                float(entry["wall"]["ci95_halfwidth_pct"]) <= args.precision_halfwidth_pct
                for entry in results["pairwise_block_ci95"].values()
            ),
            "intervals_are_descriptive_after_precision_stopping": True,
        }
        result = {
            "format": "wnn.cpu-policy-panel-result",
            "version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "source_commit": os.environ.get("WNN_BENCHMARK_SOURCE_COMMIT", "unset"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "command": sys.argv,
            "host": host,
            "single_cpu_affinity_only": True,
            "cpu_quota_asserted": False,
            "cgroup_before": cgroup_before,
            "cgroup_after": _cgroup_snapshot(),
            "load_average_start": load_before,
            "load_average_end": _load_average(),
            "source": manifest["source"],
            "fixture_manifest_path": str(args.fixture_manifest.resolve()),
            "fixture_manifest_sha256": sha256_file(args.fixture_manifest),
            "fixture_arrays": manifest["arrays"],
            "sequence_panel": manifest["sequences"],
            "datapoint_contract": {
                "source_path": "utils.py",
                "source_sha256": COMPETITION_DATAPOINT_SOURCE_SHA256,
                "fields_in_order": ["seq_ix", "step_in_seq", "need_prediction", "state"],
                "adapter_imports_pyarrow": False,
            },
            "package_binding_sha256": sha256_file(args.package_binding),
            "package_binding": binding,
            "worker_cold_start_and_constructor": worker_ready,
            "warmup_callbacks_unmeasured": warmup_records,
            "pilot_callbacks_unmeasured": pilot_records,
            "pilot_minimum_panel_estimate_seconds": minimum_panel_estimate,
            "pilot_planned_max_blocks": planned_max,
            "all_timing_blocks_are_exploratory": True,
            "confirmatory_winner_claim": False,
            "ci_history": ci_history,
            "samples": records,
            "results": results,
            "callback_scope": (
                "full 20,000-row solution.PredictionModel.predict loop from raw112 features, "
                "with the frozen synthetic need_prediction vector; each timed replay follows "
                "an untimed 99-row sentinel sequence reset. Includes per-row DataPoint creation "
                "and native preprocessing/inference; excludes process startup, model construction, "
                "fixture hashing and reset sentinel."
            ),
            "limits": [
                "Eight previously exposed F0 seed-3 fitting components; runtime only, no targets or grade.",
                "One CPU affinity is enforced; cgroup CPU quota is recorded but not asserted.",
                "Paired t intervals use completed randomized blocks and are host-conditional.",
                "Adaptive precision stopping is descriptive and not sequentially adjusted.",
                "No comparison establishes relative CPU capacity across different machine models.",
            ],
            "elapsed_wall_seconds": elapsed_wall,
            "controller_elapsed_process_seconds": time.process_time() - start_process,
            "summed_worker_callback_process_seconds": {
                label: sum(
                    float(sample["callback_process_seconds"])
                    for sample in records
                    if sample["label"] == label
                )
                for label in LABELS
            },
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "numpy": np.__version__,
        }
        (args.output / "RESULT_R1.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    finally:
        _stop_workers(workers, worker_logs)


def _prediction_delta(reference: Any, candidate: Any) -> dict[str, object]:
    """Describe possible numeric changes without converting them into quality claims."""
    import numpy as np

    delta = reference.astype(np.float64, copy=False) - candidate.astype(np.float64, copy=False)
    absolute = np.abs(delta)
    return {
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
        "different_elements": int(np.count_nonzero(delta)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def _load_average() -> list[float] | None:
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def main() -> None:
    """Dispatch a package worker, show a dry-run schedule, or execute a panel."""
    args = _parse_args()
    if args.worker:
        if any(
            value is None
            for value in (
                args.worker_label,
                args.worker_package,
                args.worker_fixture,
                args.worker_fixture_manifest,
                args.worker_manifest,
                args.worker_cpu,
            )
        ):
            raise ValueError("internal worker arguments are incomplete")
        _worker_main(args)
        return
    if args.dry_run:
        plan = {
            "seed": args.seed,
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "callbacks_per_block": 8 * 5,
            "candidate_orders_for_first_slot": [
                [
                    item["label"]
                    for item in block_schedule(args.seed, block)
                    if item["slot"] == 0 and item["label"] != "A"
                ]
                for block in range(math.factorial(len(CANDIDATES)))
            ],
            "schedule_block_0": block_schedule(args.seed, 0),
            "schedule_block_1": block_schedule(args.seed, 1),
            "pairwise_interval_example": paired_log_ci([0.0] * DEFAULT_MIN_BLOCKS),
        }
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    try:
        if args.smoke_pair:
            _run_smoke_pair(args)
        else:
            _run_panel(args)
    except BaseException as error:
        if _OUTPUT_CREATED and args.output is not None:
            failed_path = args.output / "FAILED_ATTEMPT_R1.json"
            if not failed_path.exists():
                failure = {
                    "format": "wnn.cpu-policy-benchmark-failure",
                    "version": 1,
                    "created_utc": datetime.now(UTC).isoformat(),
                    "source_commit": os.environ.get("WNN_BENCHMARK_SOURCE_COMMIT", "unset"),
                    "runner_sha256": sha256_file(Path(__file__).resolve()),
                    "command": sys.argv,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                    "cgroup": _cgroup_snapshot(),
                }
                failed_path.write_text(
                    json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
        raise


if __name__ == "__main__":
    main()
