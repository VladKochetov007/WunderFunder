"""Run the fixed R3 ridge screen on a source-bound sample panel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel, fit_candidates

from .analysis_fit import collect_fit_rows, cross_validate
from .analysis_report import build_result, score_outer
from .analysis_views import CASE_BUDGET, CASE_NAMES


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_panel(sequences: list[dict], low32: list[int], full64_order: list[int]) -> list[int]:
    fit = [item for item in sequences if item["role"] == "fit"]
    evaluation = [item for item in sequences if item["role"] == "eval"]
    fit_components = [int(item["component"]) for item in fit]
    eval_components = [int(item["component"]) for item in evaluation]
    if len(fit) != 64 or len(set(fit_components)) != 64:
        raise ValueError("R3 requires one fit sequence from each of 64 components")
    if len(evaluation) != 32 or len(set(eval_components)) != 32:
        raise ValueError("R3 requires one evaluation sequence from each of 32 components")
    if set(fit_components) & set(eval_components):
        raise ValueError("fit and evaluation components overlap")
    if (
        len(set(low32)) != 32
        or len(set(full64_order)) != 64
        or full64_order[:32] != low32
        or set(full64_order) != set(fit_components)
    ):
        raise ValueError("sample component order does not bind the nested 32/64 fit panel")
    fit_groups = [(item["source"], item["group"]) for item in fit]
    eval_groups = [(item["source"], item["group"]) for item in evaluation]
    if len(set(fit_groups)) != 64 or len(set(eval_groups)) != 32:
        raise ValueError("duplicate selected group in the fit or evaluation panel")
    if any(item["source"] != "train" for item in fit) or any(
        item["source"] != "valid" for item in evaluation
    ):
        raise ValueError("R3 fit/evaluation roles must use canonical train/valid sources")
    return full64_order


def _model_payload(model: RidgeModel, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": model.mean,
        f"{prefix}_scale": model.scale,
        f"{prefix}_coefficients": model.coefficients,
        f"{prefix}_intercepts": model.intercepts,
        f"{prefix}_norms": np.asarray(model.norms, dtype=np.float64),
    }


def _save_models(output: Path, case: str, selected: RidgeModel, baseline: RidgeModel) -> dict:
    path = output / f"models_{case}.npz"
    with path.open("xb") as stream:
        np.savez(
            stream,
            **_model_payload(selected, "selected"),
            **_model_payload(baseline, "alpha1_baseline"),
        )
        stream.flush()
        os.fsync(stream.fileno())
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _digest(path)}


def run(
    sequences: list[dict],
    output: Path,
    sample_binding: dict,
    source_manifest: dict,
) -> dict:
    """Run R3 cross-validation, fixed refits, full-panel score and receipts."""
    output.mkdir(parents=True, exist_ok=False)
    source_path = output / "SOURCE_MANIFEST.json"
    _write_json(source_path, source_manifest)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    _write_json(
        output / "START.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "sample_manifest_sha256": sample_binding["sha256"],
            "source_manifest_sha256": _digest(source_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sequences": len(sequences),
        },
    )
    fit_component_order = _validate_panel(
        sequences,
        [int(value) for value in sample_binding["fit_low32_components"]],
        [int(value) for value in sample_binding["fit_full64_component_order"]],
    )
    progress_path = output / "incremental_progress.jsonl"
    feature_started = time.process_time()
    fit_features, targets, row_components, fit_counts = collect_fit_rows(
        sequences, fit_component_order
    )
    feature_collection_cpu = time.process_time() - feature_started
    selected_models, model_archives, refit_seconds = {}, {}, {}
    with progress_path.open("x", encoding="utf-8") as progress_stream:

        def record(event: dict) -> None:
            progress_stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            progress_stream.flush()
            os.fsync(progress_stream.fileno())

        record({"stage": "fit_features_collected", **fit_counts})
        alpha_table, selections, cv_counts = cross_validate(
            sequences, fit_features, targets, row_components, fit_component_order, record
        )
        for name in CASE_NAMES:
            components = set(fit_component_order[: CASE_BUDGET[name]])
            train_rows = np.isin(row_components, list(components))
            selected_alpha = int(selections[name])
            alphas = (1,) if selected_alpha == 1 else (1, selected_alpha)
            started = time.process_time()
            fitted = fit_candidates(fit_features[name][train_rows], targets[train_rows], alphas)
            selected_models[name] = {
                "selected": fitted[selected_alpha],
                "alpha1_baseline": fitted[1],
            }
            model_archives[name] = _save_models(
                output,
                name,
                selected_models[name]["selected"],
                selected_models[name]["alpha1_baseline"],
            )
            refit_seconds[name] = time.process_time() - started
            record(
                {
                    "stage": "case_refit_archived",
                    "case": name,
                    "selected_alpha": selected_alpha,
                    "archive": model_archives[name],
                }
            )
        del fit_features, targets, row_components
        outer, outer_cost = score_outer(sequences, selected_models, record)
    return build_result(
        output,
        sequences,
        fit_component_order,
        fit_counts,
        alpha_table,
        selections,
        cv_counts,
        model_archives,
        refit_seconds,
        selected_models,
        outer,
        outer_cost,
        feature_collection_cpu,
        started_wall,
        started_cpu,
        sample_binding,
        source_path,
        progress_path,
    )
