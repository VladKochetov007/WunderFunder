"""Run the single fixed R8 constant-variance arm on the accepted panel."""

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
from idea_screens_r7.model_training import train_arm
from idea_screens_r7.panel import collect_fit_rows
from idea_screens_r7.runtime_support import (
    append_json_line,
    memory_guard,
    physical_output,
    write_json,
)

from .model import ConstantVarianceMLP, initialize_constant_variance_model
from .reporting import (
    TRAINING_RECIPE,
    archive_one_model,
    reference_contrasts,
    result_head_record,
    runtime_record,
)
from .scoring import score_endpoint

EXPECTED_FIT_ROWS = 31_808
EXPECTED_UPDATES = 756
EXPECTED_EVAL_ROWS = 59_436
EXPECTED_EPOCHS = 12
EXPECTED_R7_INIT = "649b11f31a011e43aac99f4e420c596fb191abb68f5daa502e4693764fce1db3"


def _complete_run(
    *,
    output: Path,
    progress_path: Path,
    started_wall: float,
    started_cpu: float,
    state: dict[str, str],
    model: ConstantVarianceMLP,
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    evaluation: list[dict[str, object]],
    fit_mean: np.ndarray,
    fit_scale: np.ndarray,
    reference: dict[str, object],
    identities: dict[str, str],
    preparation_sha256: str,
    runtime: dict[str, object],
    sample_seconds: float,
    collection_seconds: float,
    initial_model_sha256: str,
) -> None:
    """Train, score once, archive parameters, and record the bounded result."""

    def progress(record: dict[str, object]) -> None:
        """Persist R7 epoch diagnostics under the explicit R8 arm name."""
        normalized = dict(record)
        normalized["r7_train_arm_alias"] = normalized.pop("arm", "heteroskedastic")
        normalized["arm"] = "constant_variance"
        append_json_line(
            progress_path,
            {
                **normalized,
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
            },
        )

    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    state["stage"] = "constant_variance_fit"
    fit_record = train_arm(
        model,
        fit_features,
        fit_targets,
        "heteroskedastic",
        progress,
        memory_guard,
        counters,
    )
    if (
        fit_record["epochs_completed"] != EXPECTED_EPOCHS
        or fit_record["attempted_updates"] != EXPECTED_UPDATES
        or fit_record["successful_updates"] != EXPECTED_UPDATES
        or fit_record["nonfinite_count"] != 0
    ):
        raise RuntimeError("R8 arm did not complete its fixed 756 finite updates")
    head_state = result_head_record(model)
    if not head_state["weight_is_exactly_zero"]:
        raise RuntimeError("R8 constant logvariance weights changed from exact zero")
    weight_gradient = model.logvar_head.weight.grad
    if weight_gradient is None or torch.count_nonzero(weight_gradient).item() != 0:
        raise RuntimeError("R8 constant logvariance weights must have zero final gradient")

    state["stage"] = "single_final_endpoint"
    evaluation_started = time.perf_counter()
    score = score_endpoint(model, evaluation, fit_mean, fit_scale)
    evaluation_seconds = time.perf_counter() - evaluation_started
    contrasts = reference_contrasts(score, reference)

    state["stage"] = "model_archive"
    archive = archive_one_model(
        output / "FINAL_MODEL_AND_NORMALIZER.npz", model, fit_mean, fit_scale
    )
    result = {
        "schema": "nf19.r8_constant_variance.result.v1",
        "status": "completed",
        **identities,
        "preparation_manifest_sha256": preparation_sha256,
        "stored_r7_result_sha256": reference["result_sha256"],
        "model_initial_state_sha256": initial_model_sha256,
        "model_final_state_sha256": state_fingerprint(model),
        "training_branch": "idea_screens_r7.model_training.train_arm(arm='heteroskedastic')",
        "training_recipe": TRAINING_RECIPE,
        "arm": "constant_variance",
        "constant_logvariance_head": head_state,
        "updates": {
            "epochs_completed": fit_record["epochs_completed"],
            "expected_and_observed_updates": fit_record["successful_updates"],
            "attempted_updates": fit_record["attempted_updates"],
            "nonfinite_count": fit_record["nonfinite_count"],
        },
        "fit_curves": fit_record["epochs"],
        "evaluation": score,
        "contrasts_vs_stored_r7": contrasts,
        "archive": archive,
        "runtime": runtime,
        "phase_seconds": {
            "sample_verify_and_map": sample_seconds,
            "complete_history_fit_collection_and_normalization": collection_seconds,
            "single_final_endpoint": evaluation_seconds,
            "wall_elapsed": time.perf_counter() - started_wall,
            "cpu_elapsed": time.process_time() - started_cpu,
        },
        "budget_interpretation": "one finite seed and matched examples/updates; not equal FLOPs or wall time",
        "scope_limit": "one adaptively exposed R3 panel pilot; not a fresh full-fold estimate or latent-noise recovery claim",
    }
    state["stage"] = "result_write"
    result_sha = write_json(output / "RESULT.json", result)
    write_json(
        output / "DONE.json",
        {
            "schema": "nf19.r8_constant_variance.done.v1",
            "status": "completed",
            "result_sha256": result_sha,
            "result_bytes": (output / "RESULT.json").stat().st_size,
            "archive_sha256": archive["sha256"],
            "archive_bytes": archive["bytes"],
            "attempted_updates": EXPECTED_UPDATES,
            "successful_updates": EXPECTED_UPDATES,
            "nonfinite_count": 0,
            "elapsed_wall_s": time.perf_counter() - started_wall,
            "elapsed_cpu_s": time.process_time() - started_cpu,
            "final_model_state_sha256": result["model_final_state_sha256"],
        },
    )


def execute(
    sample_path: Path,
    output: Path,
    scratch: Path,
    preparation_sha256: str,
    identities: dict[str, str],
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
    reference: dict[str, object],
    sample_metadata: dict[str, str],
) -> None:
    """Train one R8 arm, query one final endpoint, and write scalar evidence."""
    physical_output(output, scratch)
    output.mkdir()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    progress_path = output / "PROGRESS.jsonl"
    start_record = {
        "schema": "nf19.r8_constant_variance.start.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **identities,
        "preparation_manifest_sha256": preparation_sha256,
        "stored_r7_result_sha256": reference["result_sha256"],
        "sample_path": str(sample_path),
        "output_path": str(output),
    }
    write_json(output / "START.json", start_record)
    with progress_path.open("xb"):
        pass
    append_json_line(
        progress_path,
        {"event": "initialized", "elapsed_wall_s": 0.0, "elapsed_cpu_s": 0.0},
    )
    state: dict[str, str] = {"stage": "pre_sample_guard"}
    try:
        cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        runtime = runtime_record(cuda_visibility)
        memory_before_sample = memory_guard("before_sample_load")
        state["stage"] = "sample_load"
        sample_started = time.perf_counter()
        sequences, sample_manifest = load_sample(sample_path)
        sample_seconds = time.perf_counter() - sample_started

        state["stage"] = "fit_feature_collection"
        collection_started = time.perf_counter()
        fit_raw, fit_targets, evaluation, panel = collect_fit_rows(sequences)
        fit_mean, fit_scale = fit_normalizer(fit_raw)
        fit_features = apply_normalizer(fit_raw, fit_mean, fit_scale)
        del fit_raw
        collection_seconds = time.perf_counter() - collection_started
        if fit_features.shape != (EXPECTED_FIT_ROWS, 224) or len(fit_targets) != EXPECTED_FIT_ROWS:
            raise ValueError("R8 fit rows differ from the frozen R3 panel")
        if panel["eval_original_scored_rows"] != EXPECTED_EVAL_ROWS:
            raise ValueError("R8 original evaluation mask count differs from R3")

        state["stage"] = "initialization"
        model, initialization = initialize_constant_variance_model(EXPECTED_R7_INIT)
        initial_model_sha256 = state_fingerprint(model)
        memory_before_fit = memory_guard("after_normalization_before_fit")
        source_record = {
            "schema": "nf19.r8_constant_variance.source.v1",
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "stored_r7_result_sha256": reference["result_sha256"],
            "source_file_pins": source_pins,
            "runtime_import_inventory": runtime_imports,
            "sample_metadata": {
                **sample_metadata,
                "split_sha256": sample_manifest.get("split_sha256"),
                "records": len(sample_manifest.get("records", [])),
                "sources": sample_manifest.get("sources"),
            },
            "panel": panel,
            "initialization_sha256": initialization,
            "model_initial_state_sha256": initial_model_sha256,
            "constant_head_initial": result_head_record(model),
            "normalizer": {
                "fit_rows_only": True,
                "mean_min": float(fit_mean.min()),
                "mean_max": float(fit_mean.max()),
                "scale_min": float(fit_scale.min()),
                "scale_max": float(fit_scale.max()),
                "unit_scale_columns": int((fit_scale == 1.0).sum()),
            },
            "runtime": runtime,
            "memory_before_sample": memory_before_sample,
            "memory_before_fit": memory_before_fit,
            "phase_seconds": {
                "sample_verify_and_map": sample_seconds,
                "complete_history_fit_collection_and_normalization": collection_seconds,
            },
            "trainer_alias": {
                "implementation": "idea_screens_r7.model_training.train_arm",
                "arm_argument": "heteroskedastic",
                "reason": "reuses the frozen R7 bounded-Gaussian non-control branch on the constant-output subclass",
            },
            "training_recipe": TRAINING_RECIPE,
        }
        write_json(output / "SOURCE.json", source_record)

        _complete_run(
            output=output,
            progress_path=progress_path,
            started_wall=started_wall,
            started_cpu=started_cpu,
            state=state,
            model=model,
            fit_features=fit_features,
            fit_targets=fit_targets,
            evaluation=evaluation,
            fit_mean=fit_mean,
            fit_scale=fit_scale,
            reference=reference,
            identities=identities,
            preparation_sha256=preparation_sha256,
            runtime=runtime,
            sample_seconds=sample_seconds,
            collection_seconds=collection_seconds,
            initial_model_sha256=initial_model_sha256,
        )
    except Exception as error:
        write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r8_constant_variance.failure.v1",
                "stage": state["stage"],
                "exception_type": type(error).__name__,
                "message": str(error),
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
            },
        )
        raise
