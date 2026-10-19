"""Hash-bound R11 data preparation, two-arm fit, scoring, and evidence writing."""

from __future__ import annotations

import hashlib
import os
import posixpath
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r1.sample_data import load_sample
from idea_screens_r7.model_math import apply_normalizer, fit_normalizer, state_fingerprint
from idea_screens_r7.panel import collect_fit_rows, score_pair
from idea_screens_r7.runtime_support import (
    append_json_line,
    archive_models,
    initialize_output,
    write_json,
)

from .importance_math import (
    DRAW_ROUNDS,
    UNIFORM_SEED,
    analytic_joint_gradient_norms,
    clipped_weight_table,
    correction_factors,
    importance_probabilities,
    uncentered_gradient_second_moment,
)
from .importance_training import EXPECTED_UPDATES, initialize_pair, train_arm
from .result_payload import build_result, named_sampler_score

MEMORY_HIGH_BYTES = 1_400_000_000
MEMORY_RESERVE_BYTES = 256 * 1024**2
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
EXPECTED_CGROUP_PARTS = {
    "wnn-campaign.slice",
    "computer-vision-homework-eda19-r11-v1.service",
}


def scoped_cgroup_memory_path(cgroup_text: str) -> Path:
    """Resolve only the exact R11 service memory counter from self-cgroup text."""
    entries: list[str] = []
    for line in cgroup_text.splitlines():
        hierarchy, separator, remainder = line.partition(":")
        if not separator:
            continue
        controllers, separator, relative = remainder.partition(":")
        if hierarchy == "0" and separator and not controllers:
            entries.append(relative)
    if len(entries) != 1:
        raise RuntimeError("expected exactly one cgroup-v2 membership for this process")
    relative = entries[0]
    if not relative.startswith("/") or relative == "/" or ".." in relative.split("/"):
        raise RuntimeError("R11 requires a non-root absolute cgroup-v2 membership")
    normalized = posixpath.normpath(relative)
    parts = set(normalized.split("/"))
    if not EXPECTED_CGROUP_PARTS.issubset(parts):
        raise RuntimeError("process is outside the exact R11 campaign service cgroup")
    return Path("/sys/fs/cgroup") / normalized.lstrip("/") / "memory.current"


def _process_rss_bytes() -> int | None:
    """Read process RSS in bytes when procfs exposes it."""
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def memory_guard(stage: str, memory_path: Path) -> dict[str, int | str | None]:
    """Reserve 256 MiB under the fixed R11 memory-high limit."""
    if not memory_path.is_file():
        raise RuntimeError(f"R11 service cgroup memory.current is unavailable: {memory_path}")
    current = int(memory_path.read_text().strip())
    if current + MEMORY_RESERVE_BYTES > MEMORY_HIGH_BYTES:
        raise MemoryError(f"R11 memory reserve is unavailable at {stage}: current={current}")
    return {
        "stage": stage,
        "own_cgroup_current_bytes": current,
        "process_rss_bytes": _process_rss_bytes(),
        "conservative_extra_budget_bytes": MEMORY_RESERVE_BYTES,
        "memory_high_bytes": MEMORY_HIGH_BYTES,
    }


def _array_sha256(array: np.ndarray) -> str:
    """Hash one contiguous numeric array without serializing its values."""
    values = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _runtime_settings() -> dict[str, object]:
    """Set and record the fixed CPU FP32 runtime configuration."""
    cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visibility not in ("", "-1") or torch.cuda.is_available():
        raise RuntimeError("R11 requires CUDA to be hidden and unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("R11 requires one Torch intra-op and inter-op thread")
    return {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "model_dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": cuda_visibility,
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
        "blas_thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def execute(
    *,
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    identities: dict[str, str],
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> dict[str, object]:
    """Run the fixed R11 pair after all source and sample metadata gates pass."""
    scratch = sample_path.parent
    progress_path, started_wall, started_cpu = initialize_output(
        output, scratch, sample_path, identities, preparation_sha256
    )
    stage = "scope_gate"
    state: dict[str, object] = {"stage": stage}
    try:
        memory_path = scoped_cgroup_memory_path(Path("/proc/self/cgroup").read_text())
        runtime = _runtime_settings()
        memory_before_sample = memory_guard("before_sample_mapping", memory_path)

        stage = "sample_verify_and_map"
        sample_started = time.perf_counter()
        sequences, sample_manifest = load_sample(sample_path)
        sample_seconds = time.perf_counter() - sample_started
        mapped_files = sum(len(record.get("files", {})) for record in sample_manifest["records"])
        if mapped_files != 288 or len(sequences) != 96:
            raise ValueError(
                "R11 requires the accepted 96-sequence sample with 288 bound NPY files"
            )

        stage = "fit_feature_collection"
        collection_started = time.perf_counter()
        raw_fit, fit_targets, evaluation, panel_summary = collect_fit_rows(sequences)
        mean, scale = fit_normalizer(raw_fit)
        fit_features = apply_normalizer(raw_fit, mean, scale)
        del raw_fit
        clipped_targets, weights, mu_w = clipped_weight_table(fit_targets)
        if fit_features.shape != (31_808, 224) or len(evaluation) != 32:
            raise ValueError(
                "realized R11 fit/evaluation panel differs from the accepted R3 counts"
            )
        collection_seconds = time.perf_counter() - collection_started

        stage = "initialization_and_prepass"
        uniform_model, importance_model, initialization = initialize_pair()
        memory_before_prepass = memory_guard("before_initial_gradient_norm_prepass", memory_path)
        norm_prepass_started, norm_prepass_cpu = time.perf_counter(), time.process_time()
        norms = analytic_joint_gradient_norms(
            fit_features, clipped_targets, weights, mu_w, uniform_model
        )
        norm_prepass_seconds = time.perf_counter() - norm_prepass_started
        norm_prepass_cpu_seconds = time.process_time() - norm_prepass_cpu
        sampler_started, sampler_cpu = time.perf_counter(), time.process_time()
        probabilities = importance_probabilities(norms)
        uniform_probabilities = np.full(len(norms), 1.0 / len(norms), dtype=np.float64)
        corrections = correction_factors(probabilities)
        draw_rng = np.random.default_rng(UNIFORM_SEED)
        uniforms = draw_rng.random(DRAW_ROUNDS * len(norms), dtype=np.float64)
        sampler_setup_seconds = time.perf_counter() - sampler_started
        sampler_setup_cpu_seconds = time.process_time() - sampler_cpu
        uniforms_sha256 = _array_sha256(uniforms)
        uniform_second_moment = uncentered_gradient_second_moment(norms, uniform_probabilities)
        importance_second_moment = uncentered_gradient_second_moment(norms, probabilities)

        fit_features_tensor = torch.from_numpy(np.ascontiguousarray(fit_features, dtype=np.float32))
        clipped_targets_tensor = torch.from_numpy(clipped_targets)
        weights_tensor = torch.from_numpy(weights)
        phases = {
            "sample_verify_and_map": sample_seconds,
            "full_history_fit_collection_and_normalization": collection_seconds,
            "initial_gradient_norm_prepass_wall": norm_prepass_seconds,
            "initial_gradient_norm_prepass_cpu": norm_prepass_cpu_seconds,
            "fixed_probability_correction_and_uniform_setup_wall": sampler_setup_seconds,
            "fixed_probability_correction_and_uniform_setup_cpu": sampler_setup_cpu_seconds,
        }
        source_record = {
            "schema": "nf19.r11_initial_gradient_importance.source.v1",
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "source_file_pins": source_pins,
            "runtime_import_inventory": runtime_imports,
            "sample_metadata": {
                "sample_sha256": identities["sample_sha256"],
                "membership_sha256": identities["membership_sha256"],
                "split_sha256": sample_manifest.get("split_sha256"),
                "records": len(sample_manifest.get("records", [])),
                "npy_files_hashed_and_mapped": mapped_files,
                "all_96_complete_sequences_verified": True,
            },
            "panel": panel_summary,
            "fit_only_normalizer": {
                "fit_rows_only": True,
                "mean_min": float(mean.min()),
                "mean_max": float(mean.max()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "unit_scale_columns": int((scale == 1.0).sum()),
            },
            "fit_only_objective_weights": {
                "mu_w": mu_w,
                "fit_target_rows": len(clipped_targets),
                "evaluation_targets_used_for_fit_or_sampling": False,
                "evaluation_target_values_accessed_only_for_final_score": True,
            },
            "initialization_sha256": initialization,
            "initial_gradient_sampling": {
                "score": "FP64 closed-form exact-GELU real-valued gradient-norm estimate from stored FP32 parameters/features",
                "not_bit_identical_to_fp32_autograd_machine_gradient": True,
                "probabilities_fixed_once_before_training": True,
                "shared_uniform_seed": UNIFORM_SEED,
                "shared_uniform_draw_count": int(len(uniforms)),
                "shared_uniform_draws_sha256": uniforms_sha256,
                "memory_before_prepass": memory_before_prepass,
            },
            "runtime": runtime,
            "memory_before_sample_mapping": memory_before_sample,
            "phase_seconds": phases,
        }
        write_json(output / "SOURCE.json", source_record)

        def progress(record: dict[str, object]) -> None:
            """Append one compact, durable draw-round record."""
            append_json_line(
                progress_path,
                {
                    **record,
                    "elapsed_wall_s": time.perf_counter() - started_wall,
                    "elapsed_cpu_s": time.process_time() - started_cpu,
                },
            )

        stage = "uniform_fit"
        uniform_started = time.perf_counter()
        uniform_fit = train_arm(
            name="uniform",
            model=uniform_model,
            features=fit_features_tensor,
            clipped_targets=clipped_targets_tensor,
            weights=weights_tensor,
            mu_w=mu_w,
            probabilities=uniform_probabilities,
            uniforms=uniforms,
            on_round=progress,
            memory_guard=lambda label: memory_guard(label, memory_path),
        )
        uniform_seconds = time.perf_counter() - uniform_started

        stage = "importance_fit"
        importance_started = time.perf_counter()
        importance_fit = train_arm(
            name="importance",
            model=importance_model,
            features=fit_features_tensor,
            clipped_targets=clipped_targets_tensor,
            weights=weights_tensor,
            mu_w=mu_w,
            probabilities=probabilities,
            uniforms=uniforms,
            on_round=progress,
            memory_guard=lambda label: memory_guard(label, memory_path),
        )
        importance_seconds = time.perf_counter() - importance_started
        if any(
            fit["successful_updates"] != EXPECTED_UPDATES
            or fit["attempted_updates"] != EXPECTED_UPDATES
            for fit in (uniform_fit, importance_fit)
        ):
            raise RuntimeError("R11 arms did not complete exactly 756 successful updates each")

        stage = "final_evaluation"
        memory_before_evaluation = memory_guard("before_final_evaluation", memory_path)
        evaluation_started = time.perf_counter()
        reused_score = score_pair(uniform_model, importance_model, evaluation, mean, scale)
        score = named_sampler_score(reused_score)
        evaluation_seconds = time.perf_counter() - evaluation_started
        stage = "model_archive"
        archive = archive_models(
            output / "FINAL_MODELS_AND_NORMALIZER.npz",
            uniform_model,
            importance_model,
            mean,
            scale,
        )
        phases.update(
            {
                "uniform_fit_wall": uniform_seconds,
                "importance_fit_wall": importance_seconds,
                "final_evaluation": evaluation_seconds,
            }
        )
        result = build_result(
            identities=identities,
            preparation_sha256=preparation_sha256,
            mu_w=mu_w,
            initialization=initialization,
            norms=norms,
            probabilities=probabilities,
            corrections=corrections,
            uniform_second_moment=uniform_second_moment,
            importance_second_moment=importance_second_moment,
            uniform_fit=uniform_fit,
            importance_fit=importance_fit,
            score=score,
            archive=archive,
            runtime={**runtime, "memory_before_final_evaluation": memory_before_evaluation},
            phase_seconds=phases,
            started_wall=started_wall,
            started_cpu=started_cpu,
        )
        result_sha256 = write_json(output / "RESULT.json", result)
        done = {
            "schema": "nf19.r11_initial_gradient_importance.done.v1",
            "status": "completed",
            "result_sha256": result_sha256,
            "result_bytes": (output / "RESULT.json").stat().st_size,
            "archive_sha256": archive["sha256"],
            "archive_bytes": archive["bytes"],
            "attempted_updates_per_arm": EXPECTED_UPDATES,
            "successful_updates_per_arm": EXPECTED_UPDATES,
            "total_attempted_updates": 2 * EXPECTED_UPDATES,
            "total_successful_updates": 2 * EXPECTED_UPDATES,
            "nonfinite_count": int(uniform_fit["nonfinite_count"])
            + int(importance_fit["nonfinite_count"]),
            "memory_before_final_evaluation": memory_before_evaluation,
            "elapsed_wall_s": time.perf_counter() - started_wall,
            "elapsed_cpu_s": time.process_time() - started_cpu,
            "final_model_state_sha256": {
                "uniform": state_fingerprint(uniform_model),
                "importance": state_fingerprint(importance_model),
            },
        }
        write_json(output / "DONE.json", done)
        return result
    except Exception as error:
        write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r11_initial_gradient_importance.failure.v1",
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
                "state": state,
                "elapsed_wall_s": time.perf_counter() - started_wall,
                "elapsed_cpu_s": time.process_time() - started_cpu,
            },
        )
        raise
