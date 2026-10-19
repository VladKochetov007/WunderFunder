"""Fixed R7 panel training, scoring, and create-only evidence output."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r1.sample_data import load_sample

from .model_math import (
    apply_normalizer,
    fit_normalizer,
    initialize_pair,
    state_fingerprint,
)
from .model_training import train_arm
from .panel import collect_fit_rows, score_pair
from .result_payload import (
    EXPECTED_FIT_ROWS,
    EXPECTED_UPDATES_PER_ARM,
    build_result,
)
from .runtime_support import (
    append_json_line,
    archive_models,
    initialize_output,
    memory_guard,
    write_json,
)


def execute(
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    identities: dict[str, str],
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> None:
    """Train the two fixed R7 arms and write scalar results plus a B-only archive."""
    scratch = base / "research_tmp"
    progress_path, started_wall, started_cpu = initialize_output(
        output, scratch, sample_path, identities, preparation_sha256
    )
    state: dict[str, object] = {"stage": "sample_load"}
    try:
        cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visibility not in ("", "-1"):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must hide accelerators for R7")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        sample_started = time.perf_counter()
        sequences, sample_manifest = load_sample(sample_path)
        sample_seconds = time.perf_counter() - sample_started
        state["stage"] = "fit_feature_collection"
        collection_started = time.perf_counter()
        fit_features_raw, fit_targets, evaluation, panel_summary = collect_fit_rows(sequences)
        mean, scale = fit_normalizer(fit_features_raw)
        fit_features = apply_normalizer(fit_features_raw, mean, scale)
        del fit_features_raw
        collection_seconds = time.perf_counter() - collection_started
        if fit_features.shape[0] != EXPECTED_FIT_ROWS or len(evaluation) != 32:
            raise ValueError("R7 realized fit/evaluation row counts differ from frozen panel")

        control, candidate, initialization = initialize_pair()
        runtime = {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "model_dtype": "float32",
            "cuda_visible_devices": cuda_visibility,
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
            "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
            "blas_thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
        initial_memory = memory_guard("initialized_before_first_fit_step")
        source_record = {
            "schema": "nf19.r7_mean_variance.source.v1",
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "source_file_pins": source_pins,
            "runtime_import_inventory": runtime_imports,
            "sample_metadata": {
                "sample_sha256": identities["sample_sha256"],
                "membership_sha256": identities["membership_sha256"],
                "split_sha256": sample_manifest.get("split_sha256"),
                "sources": sample_manifest.get("sources"),
                "records": len(sample_manifest.get("records", [])),
            },
            "panel": panel_summary,
            "initialization_sha256": initialization,
            "normalizer": {
                "fit_rows_only": True,
                "mean_min": float(mean.min()),
                "mean_max": float(mean.max()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "unit_scale_columns": int((scale == 1.0).sum()),
            },
            "runtime": runtime,
            "memory_before_fit": initial_memory,
            "phase_seconds": {
                "sample_verify_and_map": sample_seconds,
                "full_history_fit_collection_and_normalization": collection_seconds,
            },
        }
        write_json(output / "SOURCE.json", source_record)

        def progress(record: dict[str, object]) -> None:
            """Persist one compact epoch record and running wall/CPU clocks."""
            append_json_line(
                progress_path,
                {
                    **record,
                    "elapsed_wall_s": time.perf_counter() - started_wall,
                    "elapsed_cpu_s": time.process_time() - started_cpu,
                },
            )

        control_counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
        candidate_counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
        state["stage"] = "control_fit"
        control_fit = train_arm(
            control,
            fit_features,
            fit_targets,
            "control",
            progress,
            memory_guard,
            control_counters,
        )
        state["stage"] = "heteroskedastic_fit"
        candidate_fit = train_arm(
            candidate,
            fit_features,
            fit_targets,
            "heteroskedastic",
            progress,
            memory_guard,
            candidate_counters,
        )
        if (
            control_fit["successful_updates"] != EXPECTED_UPDATES_PER_ARM
            or candidate_fit["successful_updates"] != EXPECTED_UPDATES_PER_ARM
            or control_fit["attempted_updates"] != EXPECTED_UPDATES_PER_ARM
            or candidate_fit["attempted_updates"] != EXPECTED_UPDATES_PER_ARM
        ):
            raise RuntimeError("R7 pair did not complete exactly 756 successful updates per arm")

        state["stage"] = "final_evaluation"
        evaluation_started = time.perf_counter()
        score = score_pair(control, candidate, evaluation, mean, scale)
        evaluation_seconds = time.perf_counter() - evaluation_started
        state["stage"] = "model_archive"
        archive = archive_models(
            output / "FINAL_MODELS_AND_NORMALIZER.npz", control, candidate, mean, scale
        )
        result = build_result(
            identities=identities,
            preparation_sha256=preparation_sha256,
            initialization_sha256=initialization,
            control_fit=control_fit,
            candidate_fit=candidate_fit,
            score=score,
            archive=archive,
            runtime=runtime,
            source_phase_seconds=source_record["phase_seconds"],
            evaluation_seconds=evaluation_seconds,
            started_wall=started_wall,
            started_cpu=started_cpu,
        )
        state["stage"] = "result_write"
        result_sha = write_json(output / "RESULT.json", result)
        write_json(
            output / "DONE.json",
            {
                "schema": "nf19.r7_mean_variance.done.v1",
                "status": "completed",
                "result_sha256": result_sha,
                "result_bytes": (output / "RESULT.json").stat().st_size,
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
                "attempted_updates": EXPECTED_UPDATES_PER_ARM * 2,
                "successful_updates": EXPECTED_UPDATES_PER_ARM * 2,
                "nonfinite_count": 0,
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
                "final_model_state_sha256": {
                    "control": state_fingerprint(control),
                    "heteroskedastic": state_fingerprint(candidate),
                },
            },
        )
    except Exception as error:
        write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r7_mean_variance.failure.v1",
                "stage": state["stage"],
                "exception_type": type(error).__name__,
                "message": str(error),
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
                "state": state,
            },
        )
        raise
