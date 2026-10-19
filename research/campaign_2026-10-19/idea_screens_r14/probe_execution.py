"""One fixed R14 analytic fit, invariant check, and post-fit panel score."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np

from .history_features import coordinate_map, direct_map_check
from .linear_math import build_systems, mapped_systems, steepest_descent
from .panel_data import FIT_ROWS, prepare_fit_panel, validate_sample
from .result_payload import archive_parameters, build_result, runtime_record
from .runtime_support import (
    append_json_line,
    memory_guard,
    physical_output,
    write_json,
)
from .score_endpoints import score_endpoints

SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"


def _invariants(
    design_a: np.ndarray,
    transform: np.ndarray,
    whitener: np.ndarray,
    systems_a: tuple,
    systems_b: tuple,
    trajectories_a: list,
    trajectories_b: list,
) -> dict[str, object]:
    """Check mapped feature/objective/optimum identities and the fixed gate."""
    direct_errors = [
        direct_map_check(design_a, trajectories_b[target].beta, transform, whitener)
        for target in range(2)
    ]
    mapped_optima = [transform.T @ item.beta_star for item in trajectories_b]
    optimum_errors = [
        float(
            np.linalg.norm(mapped_optima[target] - trajectories_a[target].beta_star)
            / max(np.linalg.norm(trajectories_a[target].beta_star), 1e-15)
        )
        for target in range(2)
    ]
    if not np.isfinite(direct_errors + optimum_errors).all() or max(optimum_errors) > 1e-6:
        raise ValueError("R14 mapped analytic optima differ beyond tolerance")
    hessian_errors = [
        float(np.max(np.abs(systems_b[k].hessian - transform @ systems_a[k].hessian @ transform.T)))
        for k in range(2)
    ]
    rhs_errors = [
        float(np.max(np.abs(systems_b[k].rhs - transform @ systems_a[k].rhs))) for k in range(2)
    ]
    if max(hessian_errors + rhs_errors) > 1e-10:
        raise ValueError("R14 mapped objective congruence check failed")
    gates = []
    for target in range(2):
        gap_a = trajectories_a[target].checkpoints[-1]["relative_gap"]
        gap_b = trajectories_b[target].checkpoints[-1]["relative_gap"]
        gates.append(
            trajectories_a[target].initial_relative_gap is not None
            and trajectories_a[target].initial_gap > 0.0
            and trajectories_b[target].initial_gap > 0.0
            and gap_a is not None
            and float(gap_a) > 0.0
            and gap_b is not None
            and float(gap_b) <= 0.8 * float(gap_a)
        )
    return {
        "direct_errors": direct_errors,
        "mapped_optima": mapped_optima,
        "optimum_errors": optimum_errors,
        "hessian_errors": hessian_errors,
        "rhs_errors": rhs_errors,
        "spending_gates": gates,
    }


def execute(
    *,
    sample_path: Path,
    output: Path,
    base: Path,
    preparation_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> None:
    """Run the bounded sequence-streaming analytic comparison and score once."""
    from idea_screens_r1.sample_data import load_sample

    scratch = base / "research_tmp"
    if sample_path.parent != scratch or output.parent != scratch:
        raise ValueError("R14 sample/output must be direct children of B/research_tmp")
    physical_output(output, scratch)
    output.mkdir()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    progress = output / "PROGRESS.jsonl"
    with progress.open("xb"):
        pass
    append_json_line(progress, {"event": "initialized"})
    write_json(
        output / "START.json",
        {
            "schema": "nf19.r14.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "sample_sha256": SAMPLE_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "split_sha256": SPLIT_SHA256,
            "preparation_manifest_sha256": preparation_sha256,
            "output_path": str(output),
        },
    )
    try:
        guards = [memory_guard("before_sample_hash_and_mapping")]
        sequences, sample_manifest = load_sample(sample_path)
        mapped_files, eval_components = validate_sample(sequences, sample_manifest)
        guards.append(memory_guard("after_sample_hash_and_mapping"))
        source = {
            "schema": "nf19.r14.source.v1",
            "sample_sha256": SAMPLE_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "split_sha256": SPLIT_SHA256,
            "preparation_manifest_sha256": preparation_sha256,
            "source_revision": "a266b036",
            "source_pins": source_pins,
            "runtime_imports": runtime_imports,
            "input_npy_files_hashed_and_mapped": mapped_files,
            "complete_sequences": len(sequences),
            "fit_components": 64,
            "fit_stride_rows": FIT_ROWS,
            "eval_components": eval_components,
            "eval_scored_rows": 59_436,
            "fit_target_values_used_for_objective": True,
            "eval_target_values_used_for_fit_or_normalization": False,
            "eval_target_values_used_for_secondary_scores_after_fit": True,
            "arrays_or_model_payloads_transferred_from_B": False,
            "runtime": runtime_record(),
        }
        source_sha256 = write_json(output / "SOURCE.json", source)
        append_json_line(progress, {"event": "source_bound", "source_sha256": source_sha256})

        guards.append(memory_guard("before_fit_history_and_design"))
        panel = prepare_fit_panel(sequences, memory_guard)
        guards.extend(panel.memory_guards)
        guards.append(memory_guard("after_preallocated_fit_design"))
        gram, whitener, transform, gram_error = coordinate_map()
        guards.append(memory_guard("before_normal_equation_matrices"))
        systems_a = build_systems(panel.features_a, panel.targets)
        systems_b = mapped_systems(systems_a, transform)
        guards.append(memory_guard("after_mapped_normal_equation_matrices"))
        trajectories_a = [steepest_descent(system) for system in systems_a]
        trajectories_b = [steepest_descent(system) for system in systems_b]
        guards.append(memory_guard("after_both_128_step_paths"))
        checks = _invariants(
            panel.features_a,
            transform,
            whitener,
            systems_a,
            systems_b,
            trajectories_a,
            trajectories_b,
        )
        append_json_line(progress, {"event": "both_fixed_optimization_paths_complete"})

        beta_a = np.column_stack([item.beta for item in trajectories_a])
        beta_b = np.column_stack([item.beta for item in trajectories_b])
        beta_b_as_a = transform.T @ beta_b
        beta_star_a = np.column_stack([item.beta_star for item in trajectories_a])
        mean, scale = panel.mean, panel.scale
        del panel
        guards.append(memory_guard("after_fit_design_release_before_evaluation"))
        scores = score_endpoints(
            sequences,
            mean,
            scale,
            {
                "arm_a_128": beta_a,
                "arm_b_128": beta_b_as_a,
                "common_optimum": beta_star_a,
            },
            memory_guard,
        )
        guards.extend(scores.pop("resource_guards"))
        guards.append(memory_guard("after_secondary_eval_scoring"))
        guards.append(memory_guard("before_parameter_archive"))
        archive = archive_parameters(
            output / "PARAMETERS.npz",
            mean,
            scale,
            gram,
            whitener,
            transform,
            systems_a,
            systems_b,
            beta_a,
            beta_b,
            beta_star_a,
            checks["mapped_optima"],
        )
        guards.append(memory_guard("after_parameter_archive"))
        result = build_result(
            source_sha256=source_sha256,
            gram_error=gram_error,
            hessian_errors=checks["hessian_errors"],
            rhs_errors=checks["rhs_errors"],
            direct_errors=checks["direct_errors"],
            optimum_errors=checks["optimum_errors"],
            systems_a=systems_a,
            systems_b=systems_b,
            trajectories_a=trajectories_a,
            trajectories_b=trajectories_b,
            spending_gates=checks["spending_gates"],
            scores=scores,
            archive=archive,
            resource_guards=guards,
            elapsed_wall_seconds=time.perf_counter() - wall_start,
            elapsed_cpu_seconds=time.process_time() - cpu_start,
        )
        result_sha256 = write_json(output / "RESULT.json", result)
        result_bytes = (output / "RESULT.json").stat().st_size
        done = {
            "schema": "nf19.r14.done.v1",
            "result_sha256": result_sha256,
            "result_bytes": result_bytes,
            "parameter_archive": archive,
            "elapsed_wall_seconds": time.perf_counter() - wall_start,
            "elapsed_cpu_seconds": time.process_time() - cpu_start,
            "target_iterations_attempted": 512,
        }
        write_json(output / "DONE.json", done)
        append_json_line(progress, {"event": "done", **done})
    except Exception as error:
        failure = {
            "schema": "nf19.r14.failure.v1",
            "exception_type": type(error).__name__,
            "message": str(error),
            "elapsed_wall_seconds": time.perf_counter() - wall_start,
            "elapsed_cpu_seconds": time.process_time() - cpu_start,
        }
        write_json(output / "FAILURE.json", failure)
        append_json_line(progress, {"event": "failure", **failure})
        raise
