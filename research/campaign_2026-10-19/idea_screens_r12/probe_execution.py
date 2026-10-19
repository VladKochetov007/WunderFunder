"""Execute the fixed R12 full-fit objective pair and write compact evidence."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r1.sample_data import load_sample
from idea_screens_r7.model_math import (
    apply_normalizer,
    fit_normalizer,
    state_fingerprint,
)
from idea_screens_r7.panel import collect_fit_rows, score_pair
from idea_screens_r7.runtime_support import append_json_line, memory_guard, write_json

from .result_payload import create_output, fit_prediction_diagnostics, write_model_archive
from .training_pair import initial_gradient_norms, initialize_pair, scale_gamma, train_arm

SERVICE_NAME = "computer-vision-homework-eda19-r12-v1.service"
CAMPAIGN_SLICE = "wnn-campaign.slice"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"


def _unified_cgroup_parts(membership: str) -> tuple[str, ...]:
    """Parse the one exact unified cgroup path without guessing a fallback."""
    paths = []
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[:2] == ["0", ""]:
            path = Path(fields[2])
            if not path.is_absolute() or ".." in path.parts or path == Path("/"):
                raise RuntimeError("invalid or unscoped unified cgroup path")
            paths.append(path.parts)
    if len(paths) != 1:
        raise RuntimeError("expected one non-root unified cgroup membership")
    return paths[0]


def _require_r12_service(membership: str) -> None:
    """Require the campaign slice and exact R12 systemd unit before resource reads."""
    parts = _unified_cgroup_parts(membership)
    if CAMPAIGN_SLICE not in parts or SERVICE_NAME not in parts:
        raise RuntimeError("process is outside the exact admitted R12 cgroup")


def _r12_memory_guard(stage: str) -> dict[str, int | str]:
    """Check the exact service identity, then use the frozen 1.4 GB guard."""
    _require_r12_service(Path("/proc/self/cgroup").read_text())
    return memory_guard(stage)


def execute(
    *,
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    identities: dict[str, str],
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> None:
    """Load the verified R3 panel, train both fixed arms, and save aggregate results."""
    scratch = base / "research_tmp"
    progress, started_wall, started_cpu = create_output(
        output, scratch, sample_path, identities, preparation_sha256
    )
    state: dict[str, str] = {"stage": "environment"}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
            raise RuntimeError("R12 requires CUDA hidden")
        _require_r12_service(Path("/proc/self/cgroup").read_text())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state["stage"] = "sample_load"
        phase_start = time.perf_counter()
        sequences, sample_manifest = load_sample(sample_path)
        sample_seconds = time.perf_counter() - phase_start
        state["stage"] = "fit_collection"
        phase_start = time.perf_counter()
        raw_features, fit_targets, evaluation, panel = collect_fit_rows(sequences)
        mean, scale = fit_normalizer(raw_features)
        fit_features = apply_normalizer(raw_features, mean, scale)
        del raw_features
        collection_seconds = time.perf_counter() - phase_start
        if fit_features.shape != (31_808, 224) or len(evaluation) != 32:
            raise ValueError("R12 fit/evaluation realization differs from the frozen panel")
        wmse, pooled_corr, initialization = initialize_pair()
        state["stage"] = "initial_scale_prepass"
        memory_before_prepass = _r12_memory_guard("before_initial_scale_prepass")
        phase_start, prepass_cpu = time.perf_counter(), time.process_time()
        norms = initial_gradient_norms(wmse, fit_features, fit_targets)
        gamma = scale_gamma(norms)
        prepass = {
            "wall_seconds": time.perf_counter() - phase_start,
            "cpu_seconds": time.process_time() - prepass_cpu,
            "wmse_gradient_l2": norms["wmse"],
            "pooled_corr_gradient_l2": norms["pooled_corr"],
            "gamma": gamma,
            "fit_rows_only": True,
            "model_state_unchanged": True,
            "parameter_grad_fields_none": True,
            "variance_head_excluded": True,
            "optimizer_decay_excluded": True,
        }
        runtime = {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "model_dtype": "float32",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
            "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
        }
        source = {
            "schema": "nf19.r12_objective_pair.source.v1",
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "source_file_pins": source_pins,
            "runtime_import_inventory": runtime_imports,
            "sample_metadata": {
                "sample_sha256": SAMPLE_SHA256,
                "membership_sha256": MEMBERSHIP_SHA256,
                "split_sha256": sample_manifest.get("split_sha256"),
                "records": len(sample_manifest.get("records", [])),
            },
            "panel": panel,
            "initialization_sha256": initialization,
            "gamma_prepass": prepass,
            "fit_normalizer": {
                "fit_rows_only": True,
                "mean_min": float(mean.min()),
                "mean_max": float(mean.max()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "unit_scale_columns": int((scale == 1).sum()),
            },
            "runtime": runtime,
            "memory_before_gamma_prepass": memory_before_prepass,
            "phase_seconds": {
                "sample_verify_and_map": sample_seconds,
                "full_history_fit_collection_and_normalization": collection_seconds,
            },
            "memory_before_optimization": _r12_memory_guard("before_first_optimizer_update"),
        }
        write_json(output / "SOURCE.json", source)

        def report_update(record: dict[str, object]) -> None:
            """Append bounded update-block timing and finite-gradient evidence."""
            append_json_line(
                progress,
                {
                    **record,
                    "elapsed_wall_s": time.perf_counter() - started_wall,
                    "elapsed_cpu_s": time.process_time() - started_cpu,
                },
            )

        state["stage"] = "wmse_fit"
        wmse_fit = train_arm(
            wmse, fit_features, fit_targets, "wmse", gamma, report_update, _r12_memory_guard
        )
        state["stage"] = "pooled_corr_fit"
        pooled_fit = train_arm(
            pooled_corr,
            fit_features,
            fit_targets,
            "pooled_corr",
            gamma,
            report_update,
            _r12_memory_guard,
        )
        if any(
            result["successful_updates"] != 128
            or result["attempted_updates"] != 128
            or result["nonfinite_count"] != 0
            for result in (wmse_fit, pooled_fit)
        ):
            raise RuntimeError(
                "R12 requires exactly128 successful updates and zero nonfinite events per arm"
            )
        fit_diagnostics = {
            "wmse": fit_prediction_diagnostics(wmse, fit_features),
            "pooled_corr": fit_prediction_diagnostics(pooled_corr, fit_features),
        }
        state["stage"] = "evaluation"
        evaluation_start = time.perf_counter()
        legacy_score = score_pair(wmse, pooled_corr, evaluation, mean, scale)
        evaluation_seconds = time.perf_counter() - evaluation_start
        arms = legacy_score["arms"]
        delta = (
            arms["heteroskedastic"]["official_metric"]["wp"]
            - arms["control"]["official_metric"]["wp"]
        )
        score = {
            "metric_id": legacy_score["metric_id"],
            "components": legacy_score["components"],
            "rows": legacy_score["rows"],
            "arms": {"wmse": arms["control"], "pooled_corr": arms["heteroskedastic"]},
            "primary_delta_pooled_corr_minus_wmse_wp": delta,
            "component_sensitivity_is_not_independent_replication": True,
        }
        state["stage"] = "model_archive"
        archive = write_model_archive(
            output / "FINAL_MODELS_AND_NORMALIZER.npz", wmse, pooled_corr, mean, scale
        )
        result = {
            "schema": "nf19.r12_objective_pair.result.v1",
            "status": "completed",
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "objective_control": "canonical weighted_mse_loss on q=clamp(raw_mean,-2,2)",
            "objective_candidate": "gamma times canonical weighted_pooled_corr_loss on same clipped q",
            "gamma": gamma,
            "initial_gradient_norms": norms,
            "fit_rows": 31_808,
            "evaluation_scored_rows": 59_436,
            "updates_per_arm": 128,
            "row_exposures_per_arm": 4_071_424,
            "total_row_exposures": 8_142_848,
            "budget_interpretation": "equal rows and optimizer updates; not equal FLOPs or wall time",
            "optimizer": {
                "name": "AdamW",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0001,
                "foreach": False,
                "fused": False,
                "gradient_clip_global_norm": 1.0,
            },
            "initialization_sha256": initialization,
            "fit_prediction_clipping_diagnostics": fit_diagnostics,
            "fit_arm_summaries": {"wmse": wmse_fit, "pooled_corr": pooled_fit},
            "final_evaluation": score,
            "archive": archive,
            "runtime": runtime,
            "phase_seconds": {
                **source["phase_seconds"],
                "gamma_prepass_wall": prepass["wall_seconds"],
                "wmse_wall": wmse_fit["wall_seconds"],
                "wmse_cpu": wmse_fit["cpu_seconds"],
                "pooled_corr_wall": pooled_fit["wall_seconds"],
                "pooled_corr_cpu": pooled_fit["cpu_seconds"],
                "evaluation_wall": evaluation_seconds,
                "total_wall": time.perf_counter() - started_wall,
                "total_cpu": time.process_time() - started_cpu,
            },
            "scope_limit": "one exposed adaptive R3 panel; no fresh holdout, replication, or general objective claim",
            "predictions_in_json_or_archive": False,
        }
        state["stage"] = "result_write"
        result_sha = write_json(output / "RESULT.json", result)
        write_json(
            output / "DONE.json",
            {
                "schema": "nf19.r12_objective_pair.done.v1",
                "status": "completed",
                "result_sha256": result_sha,
                "result_bytes": (output / "RESULT.json").stat().st_size,
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
                "attempted_updates": 256,
                "successful_updates": 256,
                "nonfinite_count": 0,
                "final_state_sha256": {
                    "wmse": state_fingerprint(wmse),
                    "pooled_corr": state_fingerprint(pooled_corr),
                },
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
            },
        )
    except Exception as error:
        write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r12_objective_pair.failure.v1",
                "stage": state["stage"],
                "exception_type": type(error).__name__,
                "message": str(error),
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
            },
        )
        raise
