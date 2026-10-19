"""One fixed R13 CPU run with create-only records and scoped resource checks."""

from __future__ import annotations

import datetime as dt
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r1.sample_data import load_sample
from idea_screens_r7.model_math import state_fingerprint
from idea_screens_r7.runtime_support import append_json_line, physical_output, write_json

from .input_panel import ARM_NAMES, prepare_fit_panel
from .model_arms import assert_initial_prediction_parity, initialize_models
from .result_payload import archive_models, build_result, build_source_record
from .scoring import score_models
from .training import train_arm

SOURCE_REVISION = "a266b036"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
HIGH_BYTES = 1_400_000_000
RESERVE_BYTES = 256 * 1024**2
SERVICE = "computer-vision-homework-eda19-r13-v1.service"


def _cgroup_path(membership: str) -> Path:
    """Resolve only the exact R13 service below the campaign slice."""
    entries = []
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and not fields[1]:
            relative = Path(fields[2].lstrip("/"))
            if ".." in relative.parts:
                raise ValueError("invalid self-cgroup membership")
            entries.append(relative)
    if len(entries) != 1:
        raise RuntimeError("expected one unified self-cgroup membership")
    relative = entries[0]
    if "wnn-campaign.slice" not in relative.parts or relative.name != SERVICE:
        raise RuntimeError("process is not inside the exact R13 campaign service")
    return Path("/sys/fs/cgroup") / relative


def memory_guard(stage: str) -> dict[str, int | str]:
    """Require the fixed 256 MiB reserve under the R13 service memory high."""
    cgroup = _cgroup_path(Path("/proc/self/cgroup").read_text())
    current = int((cgroup / "memory.current").read_text().strip())
    if current > HIGH_BYTES - RESERVE_BYTES:
        raise MemoryError(f"R13 memory reserve unavailable before {stage}: {current} bytes")
    rss = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
            break
    if rss is None:
        raise RuntimeError("cannot read process RSS")
    return {
        "stage": stage,
        "own_cgroup_memory_current_bytes": current,
        "process_rss_bytes": rss,
        "process_peak_rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "high_bytes": HIGH_BYTES,
        "reserved_bytes": RESERVE_BYTES,
    }


def _configure_runtime() -> dict[str, object]:
    """Set and report the fixed CPU numerical environment before model work."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cuda_autocast = bool(torch.is_autocast_enabled("cuda"))
        cpu_autocast = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:
        cuda_autocast = bool(torch.is_autocast_enabled())
        cpu_autocast = bool(torch.is_autocast_cpu_enabled())
    cuda_initialized = bool(torch.cuda.is_initialized())
    cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if cuda_autocast or cpu_autocast or cuda_initialized or cuda_tf32 or cudnn_tf32:
        raise RuntimeError("R13 CPU runtime is not in its fixed no-autocast/no-TF32 state")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autocast_cuda_enabled": cuda_autocast,
        "autocast_cpu_enabled": cpu_autocast,
        "cuda_initialized": cuda_initialized,
        "cuda_matmul_allow_tf32": cuda_tf32,
        "cudnn_allow_tf32": cudnn_tf32,
    }


def _initialize_output(output: Path, scratch: Path, identities: dict[str, object]) -> Path:
    """Create the new physical output folder and an initialized JSONL stream."""
    physical_output(output, scratch)
    output.mkdir()
    write_json(
        output / "START.json",
        {
            "schema": "nf19.r13.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "output_path": str(output),
            **identities,
        },
    )
    progress = output / "PROGRESS.jsonl"
    with progress.open("xb"):
        pass
    append_json_line(progress, {"event": "initialized"})
    return progress


def _sample_counts(sequences: list[dict]) -> tuple[int, int]:
    """Validate that the reader mapped exactly the 288 panel NPY payloads."""
    count = 0
    for item in sequences:
        if not all(key in item for key in ("x", "y", "scored")):
            raise ValueError("R13 sample item is missing an x/y/scored array")
        count += 3
    if len(sequences) != 96 or count != 288:
        raise ValueError("R13 requires all 96 sample sequences and 288 bound NPY files")
    fit_components = {int(item["component"]) for item in sequences if item["role"] == "fit"}
    eval_components = {int(item["component"]) for item in sequences if item["role"] == "eval"}
    if len(fit_components) != 64 or len(eval_components) != 32 or fit_components & eval_components:
        raise ValueError("R13 sample component roles are incomplete or overlapping")
    return count, len(eval_components)


def execute(
    *,
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> None:
    """Load the accepted sample, train the three frozen arms, and score once."""
    if output.parent != sample_path.parent or sample_path.parent != base / "research_tmp":
        raise ValueError("R13 sample/output must be direct children of B/research_tmp")
    runtime = _configure_runtime()
    progress = _initialize_output(
        output,
        sample_path.parent,
        {
            "sample_sha256": SAMPLE_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "preparation_manifest_sha256": preparation_sha256,
            "source_revision": SOURCE_REVISION,
        },
    )
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        guards = [memory_guard("before_sample_hash_and_mapping")]
        sequences, sample_manifest = load_sample(sample_path)
        array_count, eval_components = _sample_counts(sequences)
        if len(sample_manifest["records"]) != 96:
            raise ValueError("R13 sample manifest must list 96 complete sequences")
        guards.append(memory_guard("after_sample_hash_and_mapping"))
        fit = prepare_fit_panel(sequences)
        guards.append(memory_guard("after_full_history_fit_feature_preparation"))

        source_record = build_source_record(
            preparation_sha256=preparation_sha256,
            source_pins=source_pins,
            runtime_imports=runtime_imports,
            sample_manifest=sample_manifest,
            sample_sha256=SAMPLE_SHA256,
            membership_sha256=MEMBERSHIP_SHA256,
            sample_array_count=array_count,
            panel_stats=fit.panel_stats,
        )
        source_record["runtime"] = runtime
        source_record["evaluation_components"] = eval_components
        source_sha256 = write_json(output / "SOURCE.json", source_record)
        append_json_line(
            progress,
            {"event": "source_bound_and_fit_rows_prepared", "source_sha256": source_sha256},
        )

        models, fingerprints = initialize_models()
        dtypes = sorted(
            {str(parameter.dtype) for model in models.values() for parameter in model.parameters()}
        )
        devices = sorted(
            {str(parameter.device) for model in models.values() for parameter in model.parameters()}
        )
        if dtypes != ["torch.float32"] or devices != ["cpu"]:
            raise RuntimeError("R13 models must initialize with FP32 CPU parameters")
        runtime["model_parameter_dtypes"] = dtypes
        runtime["model_parameter_devices"] = devices
        runtime["cuda_initialized_after_model_initialization"] = bool(torch.cuda.is_initialized())
        first = min(64, len(fit.targets))
        parity = assert_initial_prediction_parity(
            models,
            fit.features_by_arm[ARM_NAMES[0]][:first, :224],
            {arm: fit.features_by_arm[arm][:first, 224:] for arm in ARM_NAMES},
        )
        if len({state_fingerprint(model) for model in models.values()}) != 1:
            raise RuntimeError("R13 initialized arms are not identical")
        initialization = {
            "seed": 19_061,
            "state_fingerprints": fingerprints,
            "initial_prediction_parity": parity,
            "added_side_weights_zero": True,
            "logvariance_heads_zero_and_unused": True,
        }

        training: dict[str, dict[str, object]] = {}
        for arm in ARM_NAMES:

            def on_epoch(record: dict[str, object]) -> None:
                append_json_line(progress, record)

            training[arm] = train_arm(
                models[arm],
                fit.features_by_arm[arm],
                fit.targets,
                arm,
                on_epoch,
                memory_guard,
            )
            guards.append(memory_guard(f"{arm}_complete"))

        append_json_line(progress, {"event": "all_arms_complete_before_evaluation"})
        scores = score_models(models, fit.evaluation, fit.mean, fit.scale)
        archive = archive_models(output / "MODELS.npz", models, fit.mean, fit.scale)
        result = build_result(
            fit=fit.__dict__
            if hasattr(fit, "__dict__")
            else {
                "targets": fit.targets,
                "panel_stats": fit.panel_stats,
            },
            initialization=initialization,
            training=training,
            scores=scores,
            archive=archive,
            runtime=runtime,
            source_sha256=source_sha256,
        )
        result["resource_guards"] = guards
        result["elapsed_wall_seconds"] = time.perf_counter() - wall_start
        result["elapsed_cpu_seconds"] = time.process_time() - cpu_start
        result_sha256 = write_json(output / "RESULT.json", result)
        result_bytes = (output / "RESULT.json").stat().st_size
        done = {
            "schema": "nf19.r13.done.v1",
            "result_sha256": result_sha256,
            "result_bytes": result_bytes,
            "model_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - wall_start,
            "elapsed_cpu_seconds": time.process_time() - cpu_start,
            "successful_updates": sum(
                int(item["successful_updates"]) for item in training.values()
            ),
            "nonfinite_count": sum(int(item["nonfinite_count"]) for item in training.values()),
        }
        write_json(output / "DONE.json", done)
        append_json_line(progress, {"event": "done", **done})
    except Exception as exc:
        failure = {
            "schema": "nf19.r13.failure.v1",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_wall_seconds": time.perf_counter() - wall_start,
            "elapsed_cpu_seconds": time.process_time() - cpu_start,
        }
        write_json(output / "FAILURE.json", failure)
        append_json_line(progress, {"event": "failure", **failure})
        raise
