"""Execute one fixed NLL fit on the restored R15 sparse+EMA representation."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r15.pipeline_reports import validate_loaded_panel
from idea_screens_r15.representation import fingerprint

from .panel import prepare_fit_panel
from .reference import (
    PREDICTOR_INITIAL_SHA256,
    REFERENCE_ARCHIVE_PATH,
    SPARSE_DICTIONARY_SHA256,
    read_reference_result,
    restore_sparse_reference,
)
from .runtime_support import (
    append_durable_line,
    create_model_archive,
    memory_guard,
    physical_output,
    sha256_file,
    write_durable_json,
)
from .scoring import score_candidate
from .training import EXPECTED_UPDATES, initialize_candidate, train_candidate


def execute(
    sample_path: Path,
    output: Path,
    support_root: Path,
    preparation_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
    reference_result_path: Path,
    runtime_state: dict[str, object],
) -> dict[str, object]:
    """Restore fixed sparse state, fit NLL once, score once, and persist receipts."""
    from idea_screens_r1.sample_data import load_sample

    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError("combo requires float32 Torch model initialization")
    if (
        runtime_state.get("torch_threads") != 1
        or runtime_state.get("torch_interop_threads") != 1
        or runtime_state.get("cuda_initialized")
        or runtime_state.get("autocast_cuda_enabled")
        or runtime_state.get("autocast_cpu_enabled")
        or runtime_state.get("tf32_matmul_enabled")
        or runtime_state.get("tf32_cudnn_enabled")
    ):
        raise RuntimeError("combo CPU runtime state differs from its fixed execution contract")
    physical_output(output, sample_path.parent)
    output.mkdir()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    guards: list[dict[str, object]] = []

    def guard(stage: str) -> dict[str, object]:
        snapshot = memory_guard(stage)
        guards.append(snapshot)
        return snapshot

    sample_sha = sha256_file(sample_path / "SAMPLE.json")
    membership_sha = sha256_file(sample_path / "MEMBERSHIP.json")
    sample_meta = json.loads((sample_path / "SAMPLE.json").read_bytes())
    start = {
        "schema": "nf19.combo.start.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "sample_sha256": sample_sha,
        "membership_sha256": membership_sha,
        "split_sha256": sample_meta["split_sha256"],
        "preparation_manifest_sha256": preparation_sha256,
        "output_path": str(output),
    }
    write_durable_json(output / "START.json", start)
    progress = output / "PROGRESS.jsonl"
    with progress.open("xb"):
        pass
    append_durable_line(progress, {"event": "initialized"})
    try:
        guard("before_reference_archive_hash_and_load")
        expected_result_path = (
            support_root / "idea_combo_r1" / "REFERENCE_R15_RESULT.json"
        ).resolve()
        if reference_result_path.resolve() != expected_result_path:
            raise ValueError("R15 compact reference result escaped the pinned combo source bundle")
        reference_result = read_reference_result(reference_result_path)
        archive_path = (sample_path.parent / REFERENCE_ARCHIVE_PATH).resolve()
        expected_archive_parent = (sample_path.parent / "nf19-r15-v1").resolve()
        if archive_path.parent != expected_archive_parent or not archive_path.is_relative_to(
            sample_path.parent.resolve()
        ):
            raise ValueError("R15 archive path escapes the B research_tmp reference directory")
        dictionary, restored, archive_meta = restore_sparse_reference(
            archive_path, reference_result
        )
        guard("after_reference_archive_restore")

        sequences, sample_manifest = load_sample(sample_path)
        counts = validate_loaded_panel(sequences, sample_manifest)
        if counts["fit_components"] != 64 or counts["eval_components"] != 32:
            raise ValueError("combo sample component roles differ from the fixed R3 panel")
        source_record = {
            "schema": "nf19.combo.source.v1",
            "source_pins": source_pins,
            "runtime_imports": runtime_imports,
            "preparation_manifest_sha256": preparation_sha256,
            "sample_sha256": sample_sha,
            "membership_sha256": membership_sha,
            "split_sha256": sample_manifest["split_sha256"],
            "mapped_npy_file_count": 288,
            "target_files_hashed_and_mapped_for_integrity": True,
            "fit_target_values_used_for_supervised_training": True,
            "eval_target_values_used_only_for_the_new_final_endpoint_score": True,
            "fit_components": counts["fit_components"],
            "fit_stride_rows": counts["fit_stride_rows"],
            "eval_components": counts["eval_components"],
            "eval_scored_rows": counts["eval_scored_rows"],
            "r15_reference_result_sha256": sha256_file(reference_result_path),
            "r15_reference_archive": archive_meta,
            "r15_old_predictors_reinferred_or_regraded": False,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "torch_runtime": runtime_state,
            "blas_thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "all_model_parameters_cpu_float32": True,
        }
        source_sha = write_durable_json(output / "SOURCE.json", source_record)
        guard("after_sample_mapping")

        features, targets, normalizers, feature_meta = prepare_fit_panel(
            sequences, sample_manifest, dictionary, restored, guard
        )
        guard("after_fit_feature_matrix")
        model, initial_sha = initialize_candidate()
        if initial_sha != PREDICTOR_INITIAL_SHA256:
            raise RuntimeError("combo predictor initialization does not match stored R15 init")
        fit_stats = train_candidate(
            model,
            features,
            targets,
            lambda event: append_durable_line(progress, event),
            guard,
        )
        if fit_stats["attempted_updates"] != EXPECTED_UPDATES:
            raise RuntimeError("combo predictor attempted-update count differs")
        if fingerprint(dictionary) != SPARSE_DICTIONARY_SHA256:
            raise RuntimeError("frozen R15 sparse dictionary changed during predictor training")
        for parameter in dictionary.parameters():
            parameter.grad = None
        del features, targets
        guard("before_new_endpoint_scoring")
        scored = score_candidate(
            sequences,
            model,
            dictionary,
            restored["raw_input_mean"],
            restored["raw_input_scale"],
            normalizers["sparse_downstream_mean"],
            normalizers["sparse_downstream_scale"],
            reference_result,
            guard,
        )
        guard("after_new_endpoint_scoring")
        arrays = {
            **restored,
            **{
                f"candidate_{key.replace('.', '_')}": value.detach().cpu().numpy()
                for key, value in model.state_dict().items()
            },
        }
        archive = create_model_archive(output / "model_parameters.npz", arrays)
        result: dict[str, object] = {
            "schema": "nf19.combo.result.v1",
            "source_sha256": source_sha,
            "preparation_manifest_sha256": preparation_sha256,
            "sample_sha256": sample_sha,
            "membership_sha256": membership_sha,
            "split_sha256": sample_manifest["split_sha256"],
            "reference_r15_result_sha256": source_record["r15_reference_result_sha256"],
            "reference_r15_archive": archive_meta,
            "feature_pipeline": "frozen R15 signed top4/16 codes + retained residual + causal signed-delta EMA32",
            "fit_normalizers_exactly_match_stored_r15": feature_meta["normalizer_exact_match"],
            "fit_normalizer_metadata": {
                "raw_scale_min": float(normalizers["raw_input_scale"].min()),
                "raw_scale_max": float(normalizers["raw_input_scale"].max()),
                "downstream_scale_min": float(normalizers["sparse_downstream_scale"].min()),
                "downstream_scale_max": float(normalizers["sparse_downstream_scale"].max()),
            },
            "fit_features": feature_meta,
            "training": fit_stats,
            "optimizer": "AdamW(lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0001,foreach=false,fused=false)",
            "objective": "unchanged R7 bounded Gaussian NLL; mean scored with global_wp_clip2_v1",
            "evaluation": scored,
            "expected_updates": EXPECTED_UPDATES,
            "batch_size": 512,
            "epochs": 12,
            "row_exposures": 381_696,
            "all_model_parameters_cpu_float32": True,
            "nonfinite_count": fit_stats["nonfinite_count"],
            "memory_guards": guards,
            "model_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
            "component_sensitivity_is_not_independent_replication": True,
            "interpretation_scope": "one fixed exposed-panel sparse+EMA NLL endpoint; not an interaction estimate",
        }
        result_sha = write_durable_json(output / "RESULT.json", result)
        done = {
            "schema": "nf19.combo.done.v1",
            "result_sha256": result_sha,
            "result_bytes": (output / "RESULT.json").stat().st_size,
            "source_sha256": source_sha,
            "model_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
        }
        write_durable_json(output / "DONE.json", done)
        append_durable_line(progress, {"event": "complete", "result_sha256": result_sha})
        return result
    except Exception as exc:
        failure = {
            "schema": "nf19.combo.failure.v1",
            "exception_type": type(exc).__name__,
            "message": str(exc)[:1000],
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
        }
        try:
            write_durable_json(output / "FAILURE.json", failure)
            append_durable_line(progress, {"event": "failure", **failure})
        finally:
            raise
