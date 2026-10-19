"""One fixed bounded R15 sparse/dense representation pipeline execution."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from .panel import collect_fit_inputs, fit_features_for_arm, score_endpoints
from .pipeline_reports import fit_dictionary_diagnostics, model_arrays, validate_loaded_panel
from .representation import (
    DictionaryAutoencoder,
    fingerprint,
    fit_normalizer,
    initialize_dictionary_pair,
)
from .runtime_support import (
    append_json_line,
    archive_arrays,
    assert_cpu_runtime,
    memory_guard,
    physical_output,
    sha256_file,
    write_json,
)
from .training import initialize_predictor_pair, train_dictionary, train_predictor


def _check_dictionaries_frozen(
    dictionaries: dict[str, DictionaryAutoencoder],
    frozen_fingerprints: dict[str, str],
) -> dict[str, str]:
    """Require dictionary weights to match the post-training frozen state."""
    final = {name: fingerprint(model) for name, model in dictionaries.items()}
    if final != frozen_fingerprints:
        raise RuntimeError("R15 dictionary changed during downstream predictor fitting")
    return final


def execute(
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> dict[str, object]:
    """Run the four fixed fits, score once, and persist compact B-side receipts."""
    assert_cpu_runtime()
    from idea_screens_r1.sample_data import load_sample

    physical_output(output, sample_path.parent)
    output.mkdir()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    sample_meta = json.loads((sample_path / "SAMPLE.json").read_bytes())
    membership_sha = sha256_file(sample_path / "MEMBERSHIP.json")
    write_json(
        output / "START.json",
        {
            "schema": "nf19.r15.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "sample_sha256": sha256_file(sample_path / "SAMPLE.json"),
            "membership_sha256": membership_sha,
            "split_sha256": sample_meta["split_sha256"],
            "preparation_manifest_sha256": preparation_sha256,
            "output_path": str(output),
        },
    )
    progress = output / "PROGRESS.jsonl"
    with progress.open("xb"):
        pass
    append_json_line(progress, {"event": "initialized", "elapsed_wall_s": 0.0})
    guards: list[dict[str, object]] = []

    def observed_memory_guard(stage: str) -> dict[str, object]:
        snapshot = memory_guard(stage)
        guards.append(snapshot)
        return snapshot

    try:
        observed_memory_guard("before_sample_mapping")
        sequences, manifest = load_sample(sample_path)
        counts = validate_loaded_panel(sequences, manifest)
        runtime_state = assert_cpu_runtime()
        if torch.get_default_dtype() is not torch.float32:
            raise RuntimeError("R15 requires float32 model initialization")
        source_record = {
            "schema": "nf19.r15.source.v1",
            "source_pins": source_pins,
            "runtime_imports": runtime_imports,
            "sample_sha256": sha256_file(sample_path / "SAMPLE.json"),
            "membership_sha256": membership_sha,
            "split_sha256": manifest["split_sha256"],
            "sample_records": len(manifest["records"]),
            "mapped_npy_file_count": 288,
            "target_files_hashed_and_mapped_for_integrity": True,
            "eval_target_values_used_only_for_final_score": True,
            "fit_target_values_used_only_for_supervised_predictors": True,
            "fit_components": counts["fit_components"],
            "fit_rows": 31_808,
            "eval_components": counts["eval_components"],
            "eval_scored_rows": 59_436,
            "torch_threads": runtime_state["torch_threads"],
            "torch_interop_threads": runtime_state["torch_interop_threads"],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_initialized": runtime_state["cuda_initialized"],
            "autocast_cuda_enabled": runtime_state["autocast_cuda_enabled"],
            "autocast_cpu_enabled": runtime_state["autocast_cpu_enabled"],
            "tf32_matmul_enabled": runtime_state["tf32_matmul_enabled"],
            "tf32_cudnn_enabled": runtime_state["tf32_cudnn_enabled"],
            "model_dtype": "float32",
        }
        source_sha = write_json(output / "SOURCE.json", source_record)
        observed_memory_guard("after_sample_mapping")
        fit_raw = collect_fit_inputs(sequences)
        raw_mean, raw_scale = fit_normalizer(fit_raw)
        fit_normalized = np.ascontiguousarray(
            ((fit_raw.astype(np.float64) - raw_mean) / raw_scale).astype(np.float32)
        )
        if not np.isfinite(fit_normalized).all():
            raise FloatingPointError("R15 raw input normalization produced nonfinite values")
        observed_memory_guard("after_raw_fit_normalizer")
        dictionaries = dict(
            zip(("dense", "sparse"), initialize_dictionary_pair(19071), strict=True)
        )
        dict_initial = {name: fingerprint(model) for name, model in dictionaries.items()}
        if dict_initial["dense"] != dict_initial["sparse"]:
            raise RuntimeError("R15 dictionaries do not share the frozen initialization")
        dict_stats: dict[str, object] = {}
        for name in ("dense", "sparse"):
            model = dictionaries[name]
            counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
            fit_record = train_dictionary(
                model,
                fit_normalized,
                19072,
                lambda event, arm=name: append_json_line(progress, {**event, "arm": arm}),
                counters,
            )
            if counters != {
                "attempted_updates": 756,
                "successful_updates": 756,
                "nonfinite_count": 0,
            }:
                raise RuntimeError(f"R15 {name} dictionary update budget differs: {counters}")
            for parameter in model.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            if any(
                parameter.grad is not None or parameter.requires_grad
                for parameter in model.parameters()
            ):
                raise RuntimeError("R15 dictionary did not enter the fixed no-gradient state")
            fit_dict_meta = fit_dictionary_diagnostics(model, fit_normalized)
            dict_stats[name] = {**fit_record, **fit_dict_meta}
            append_json_line(progress, {"event": "dictionary_fit_complete", "arm": name})
            observed_memory_guard(f"after_{name}_dictionary_fit")
        dict_frozen = {name: fingerprint(model) for name, model in dictionaries.items()}
        predictor_dense, predictor_sparse, predictor_initial = initialize_predictor_pair()
        predictors = {"dense": predictor_dense, "sparse": predictor_sparse}
        for model in (*dictionaries.values(), *predictors.values()):
            if any(
                parameter.device.type != "cpu" or parameter.dtype is not torch.float32
                for parameter in model.parameters()
            ):
                raise RuntimeError("R15 requires CPU float32 model parameters")
        feature_means: dict[str, np.ndarray] = {}
        feature_scales: dict[str, np.ndarray] = {}
        predictor_stats: dict[str, object] = {}
        for name in ("dense", "sparse"):
            features, targets, mean, scale, feature_meta = fit_features_for_arm(
                sequences, dictionaries[name], raw_mean, raw_scale, observed_memory_guard
            )
            feature_means[name], feature_scales[name] = mean, scale
            counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
            model_initial = fingerprint(predictors[name])
            trained = train_predictor(
                predictors[name],
                features,
                targets,
                19074,
                lambda event, arm=name: append_json_line(progress, {**event, "arm": arm}),
                counters,
            )
            if counters != {
                "attempted_updates": 756,
                "successful_updates": 756,
                "nonfinite_count": 0,
            }:
                raise RuntimeError(f"R15 {name} predictor update budget differs: {counters}")
            predictor_stats[name] = {
                **trained,
                "initialization_sha256": model_initial,
                **feature_meta,
            }
            del features, targets
            observed_memory_guard(f"after_{name}_predictor_fit")
        dict_final = _check_dictionaries_frozen(dictionaries, dict_frozen)
        for model in dictionaries.values():
            for parameter in model.parameters():
                parameter.grad = None
        observed_memory_guard("before_final_endpoint_scoring")
        scored = score_endpoints(
            sequences,
            predictors,
            dictionaries,
            {name: raw_mean for name in dictionaries},
            {name: raw_scale for name in dictionaries},
            feature_means,
            feature_scales,
            observed_memory_guard,
        )
        arrays = model_arrays(
            dictionaries, predictors, raw_mean, raw_scale, feature_means, feature_scales
        )
        archive = archive_arrays(output / "model_parameters.npz", arrays)
        result = {
            "schema": "nf19.r15.result.v1",
            "source_sha256": source_sha,
            "preparation_manifest_sha256": preparation_sha256,
            "sample_sha256": source_record["sample_sha256"],
            "membership_sha256": membership_sha,
            "split_sha256": manifest["split_sha256"],
            "metric_id": scored["metric_id"],
            "primary_sparse_minus_dense_wp": scored["primary_sparse_minus_dense_wp"],
            "target_deltas_sparse_minus_dense": scored["target_deltas_sparse_minus_dense"],
            "prospective_gate": {
                "minimum_pooled_delta": 0.005,
                "both_target_deltas_positive": True,
                "passed": bool(
                    scored["primary_sparse_minus_dense_wp"] >= 0.005
                    and all(value > 0 for value in scored["target_deltas_sparse_minus_dense"])
                ),
            },
            "evaluation": scored,
            "fit": {"dictionaries": dict_stats, "predictors": predictor_stats},
            "dictionary_initial_sha256": dict_initial,
            "dictionary_frozen_sha256": dict_frozen,
            "dictionary_final_sha256": dict_final,
            "predictor_initialization_sha256": predictor_initial,
            "expected_updates_per_fit": 756,
            "fit_count": 4,
            "fit_total_updates": sum(
                int(dict_stats[name]["successful_updates"])
                + int(predictor_stats[name]["successful_updates"])
                for name in ("dense", "sparse")
            ),
            "optimizer": "AdamW(lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0001,foreach=false,fused=false)",
            "batch_size": 512,
            "epochs_per_fit": 12,
            "all_model_parameters_cpu_float32": True,
            "nonfinite_count": sum(
                int(v["nonfinite_count"])
                for part in (dict_stats, predictor_stats)
                for v in part.values()
            ),
            "memory_guards": guards,
            "model_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
            "component_sensitivity_is_not_independent_replication": True,
            "interpretation_scope": "one fixed exposed-panel sparse-vs-dense retained-residual pipeline",
        }
        result_sha = write_json(output / "RESULT.json", result)
        done = {
            "schema": "nf19.r15.done.v1",
            "result_sha256": result_sha,
            "result_bytes": (output / "RESULT.json").stat().st_size,
            "source_sha256": source_sha,
            "model_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
        }
        write_json(output / "DONE.json", done)
        append_json_line(progress, {"event": "complete", "result_sha256": result_sha})
        return result
    except Exception as exc:
        failure = {
            "schema": "nf19.r15.failure.v1",
            "exception_type": type(exc).__name__,
            "message": str(exc)[:1000],
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_cpu_seconds": time.process_time() - started_cpu,
        }
        try:
            write_json(output / "FAILURE.json", failure)
            append_json_line(progress, {"event": "failure", **failure})
        finally:
            raise
