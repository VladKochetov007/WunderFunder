"""Run the single fixed NumPy input-innovation screen and write receipts."""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import numpy as np

from .estimator import GROUPS, edge_is_stable, fit_direction
from .panel import FIT_ROWS, load_fit_panel
from .runtime_support import (
    append_json_line,
    archive_parameters,
    initialize_progress,
    memory_guard,
    physical_output,
    sha256_file,
    write_json,
)


def _runtime_provenance() -> dict[str, object]:
    """Describe observed NumPy-only runtime state without probing CUDA APIs."""
    import sys

    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_imported_observed": "torch" in sys.modules,
        "pyarrow_imported_observed": "pyarrow" in sys.modules,
        "cuda_visibility_environment": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        },
        "blas_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _paired_edges(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pair the two fixed component-half scores for each ordered group edge."""
    by_key = {(item["source"], item["destination"], item["direction"]): item for item in reports}
    edges: list[dict[str, object]] = []
    for source in GROUPS:
        for destination in GROUPS:
            if source == destination:
                continue
            forward = by_key[(source, destination, "L_to_R")]
            reverse = by_key[(source, destination, "R_to_L")]
            edges.append(
                {
                    "source": source,
                    "destination": destination,
                    "L_to_R_relative_sse_reduction": forward["relative_sse_reduction"],
                    "R_to_L_relative_sse_reduction": reverse["relative_sse_reduction"],
                    "L_to_R_component_median_relative_sse_reduction": forward[
                        "component_median_relative_sse_reduction"
                    ],
                    "R_to_L_component_median_relative_sse_reduction": reverse[
                        "component_median_relative_sse_reduction"
                    ],
                    "stable": edge_is_stable(forward, reverse),
                }
            )
    return edges


def execute(
    sample_dir: Path,
    output: Path,
    expected_sample_sha256: str,
    preparation_manifest_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
) -> dict[str, object]:
    """Validate, fit both fixed directions, and persist compact scalar evidence."""
    physical_output(output, sample_dir.parent)
    output.mkdir()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    start = {
        "schema": "nf19.r16.start.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "sample_sha256_expected": expected_sample_sha256,
        "preparation_manifest_sha256": preparation_manifest_sha256,
        "output_path": str(output),
        "operation": "input_only_conditional_innovation_screen",
    }
    write_json(output / "START.json", start)
    progress_path = output / "PROGRESS.jsonl"
    initialize_progress(progress_path)
    append_json_line(progress_path, {"event": "initialized", "elapsed_wall_s": 0.0})
    memory: list[dict[str, int | str]] = []

    def observe(stage: str) -> dict[str, int | str]:
        snapshot = memory_guard(stage)
        memory.append(snapshot)
        if (
            stage.endswith("_input_map_0")
            or stage.endswith("_input_map_31")
            or stage.endswith("_input_map_63")
        ):
            append_json_line(
                progress_path,
                {
                    "event": "fit_input_mapping",
                    "stage": stage,
                    "elapsed_wall_s": time.perf_counter() - wall_start,
                },
            )
        return snapshot

    source_sha: str | None = None
    try:
        observe("before_metadata_and_input_verification")
        input_start = time.perf_counter()
        panel = load_fit_panel(sample_dir, expected_sample_sha256, observe)
        input_elapsed = time.perf_counter() - input_start
        if (
            len(panel.input_receipts) != 128
            or len(panel.left.components) != 32
            or len(panel.right.components) != 32
        ):
            raise RuntimeError("R16 fit input or component counts differ from the frozen design")
        source_record = {
            "schema": "nf19.r16.source.v1",
            "source_pins": source_pins,
            "runtime_imports": runtime_imports,
            "sample_sha256": sha256_file(sample_dir / "SAMPLE.json"),
            "membership_sha256": sha256_file(sample_dir / "MEMBERSHIP.json"),
            "split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
            "sample_metadata_records": 96,
            "fit_components": 64,
            "evaluation_components_metadata_only": 32,
            "fit_rows": FIT_ROWS,
            "rows_per_fit_component": 497,
            "fit_input_files_hashed_and_mapped": 128,
            "input_file_receipts": panel.input_receipts,
            "target_values_accessed": False,
            "y_payloads_opened": 0,
            "evaluation_array_payloads_opened": 0,
            "selected_raw_past_and_current_finite_checks_passed": True,
            "official_scorer_called": False,
            "runtime": _runtime_provenance(),
        }
        source_sha = write_json(output / "SOURCE.json", source_record)
        observe("after_fit_input_assembly")
        append_json_line(
            progress_path,
            {
                "event": "fit_inputs_ready",
                "fit_rows": FIT_ROWS,
                "elapsed_wall_s": time.perf_counter() - wall_start,
            },
        )

        analysis_start = time.perf_counter()
        reports: list[dict[str, object]] = []
        parameters: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for direction, train_rows, held_rows in (
            ("L_to_R", panel.left, panel.right),
            ("R_to_L", panel.right, panel.left),
        ):
            observe(f"before_{direction}")
            edge_reports, edge_parameters, direction_diagnostics = fit_direction(
                train_rows, held_rows, direction
            )
            reports.extend(edge_reports)
            parameters.update(edge_parameters)
            diagnostics[direction] = direction_diagnostics
            observe(f"after_{direction}")
            append_json_line(
                progress_path,
                {
                    "event": "direction_complete",
                    "direction": direction,
                    "edge_count": len(edge_reports),
                    "elapsed_wall_s": time.perf_counter() - wall_start,
                },
            )
        if len(reports) != 12 or len(parameters) == 0:
            raise RuntimeError("R16 did not retain all six edges in both directions")
        edges = _paired_edges(reports)
        analysis_elapsed = time.perf_counter() - analysis_start
        observe("before_parameter_archive")
        archive_start = time.perf_counter()
        archive = archive_parameters(output / "PARAMETERS.npz", parameters)
        archive_elapsed = time.perf_counter() - archive_start
        result = {
            "schema": "nf19.r16.result.v1",
            "experiment": "E-NF19-IDEA-SCREENS-R16",
            "hypothesis": "H-NF19-10",
            "source_sha256": source_sha,
            "analysis": {
                "fit_components": 64,
                "fit_rows_per_direction": 15_904,
                "held_rows_per_direction": 15_904,
                "ordered_edges": 6,
                "directional_evaluations": 12,
                "baseline_ridge_fits": 6,
                "source_projection_fits": 12,
                "correction_ridge_fits": 12,
                "direction_diagnostics": diagnostics,
                "edge_scores": reports,
                "paired_edges": edges,
                "stable_edge_count": sum(bool(edge["stable"]) for edge in edges),
                "spending_gate_passed": any(bool(edge["stable"]) for edge in edges),
                "gate": {
                    "minimum_relative_sse_reduction_each_direction": 0.01,
                    "component_median_relative_sse_reduction_each_direction": ">0",
                    "all_32_component_denominators_required": True,
                },
                "interpretation_scope": "input-prediction proxy on the exposed R3 fit sample; not a target score or causal graph",
            },
            "parameter_archive": archive,
            "phase_elapsed_seconds": {
                "fit_input_hash_map_and_assembly": input_elapsed,
                "both_directional_fit_and_held_summaries": analysis_elapsed,
                "parameter_archive_write": archive_elapsed,
            },
            "elapsed_wall_s": time.perf_counter() - wall_start,
            "elapsed_process_cpu_s": time.process_time() - cpu_start,
            "memory_observations": memory,
        }
        result_sha = write_json(output / "RESULT.json", result)
        done = {
            "schema": "nf19.r16.done.v1",
            "result_sha256": result_sha,
            "result_bytes": (output / "RESULT.json").stat().st_size,
            "parameter_archive": archive,
            "input_file_count_hashed": 128,
            "elapsed_wall_s": time.perf_counter() - wall_start,
            "elapsed_process_cpu_s": time.process_time() - cpu_start,
            "status": "complete",
        }
        write_json(output / "DONE.json", done)
        append_json_line(progress_path, {"event": "done", "result_sha256": result_sha})
        return result
    except Exception as exc:
        failure = {
            "schema": "nf19.r16.failure.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_sha256": source_sha,
            "elapsed_wall_s": time.perf_counter() - wall_start,
            "elapsed_process_cpu_s": time.process_time() - cpu_start,
            "memory_observations": memory,
        }
        append_json_line(progress_path, {"event": "failure", **failure})
        write_json(output / "FAILURE.json", failure)
        raise
