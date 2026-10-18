"""Export completed fixed-epoch fits and grade only exact paired folds."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    from .postprocess_common import (
        _check_disk,
        _remaining,
        _runtime_paths,
    )
    from .postprocess_export_run import _export_one
    from .postprocess_fit import _fit_binding
    from .postprocess_grade import _grade_pair
except ImportError:  # Supports direct execution with control/ on sys.path.
    from postprocess_common import (
        _check_disk,
        _remaining,
        _runtime_paths,
    )
    from postprocess_export_run import _export_one
    from postprocess_fit import _fit_binding
    from postprocess_grade import _grade_pair


def finish_cells(root: Path, jobs: list[dict[str, Any]], deadline: float) -> dict[str, Any]:
    """Export completed fixed-epoch fits and grade only exact paired exported folds.

    `jobs` contains the frozen heterogeneous JSON records from WAVE_PLAN. This function
    never launches or retries a fit; existing failed export or grade attempts are retained.
    """
    root = root.resolve(strict=True)
    if not math.isfinite(deadline):
        raise ValueError("deadline must be a finite Unix epoch timestamp")
    _check_disk(root)
    if not jobs:
        return {
            "schema": "tabdata-b300-postprocess.v1",
            "root": str(root),
            "deadline_unix": deadline,
            "status": "partial",
            "exports": [],
            "grades": [],
            "resource_admission": False,
            "stopped_reason": "no jobs supplied",
        }
    _remaining(deadline)
    paths = _runtime_paths(root, jobs)
    _remaining(deadline)
    results: dict[str, Any] = {
        "schema": "tabdata-b300-postprocess.v1",
        "root": str(root),
        "deadline_unix": deadline,
        "status": "partial",
        "exports": [],
        "grades": [],
        "resource_admission": False,
        "stopped_reason": None,
    }
    fits: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[str, Any]] = {}
    for job in jobs:
        try:
            _remaining(deadline)
            _check_disk(root)
        except (OSError, TimeoutError) as error:
            results["stopped_reason"] = str(error)
            break
        run_name = str(job.get("id", "unknown"))
        try:
            fit = _fit_binding(root, job)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            results["exports"].append(
                {"run_name": run_name, "status": "fit_not_admitted", "error": str(error)}
            )
            continue
        fits[run_name] = fit
        entry = _export_one(root, paths, fit, deadline)
        admitted = bool(entry.pop("resource_admitted", False))
        if admitted:
            exports[run_name] = entry
            results["resource_admission"] = True
        results["exports"].append(
            {key: value for key, value in entry.items() if key != "output_root"}
        )
        if not admitted and not results["resource_admission"]:
            results["stopped_reason"] = (
                "first exact-path export resource admission failed; no further exports attempted"
            )
            break
        if not admitted:
            results["stopped_reason"] = (
                "a later export failed or was preserved; remaining cells continue"
            )
    if not results["resource_admission"]:
        results["stopped_reason"] = (
            results["stopped_reason"] or "no successful exact-path export resource admission"
        )
        results["status"] = "partial"
        return results
    by_fold: dict[int, dict[str, dict[str, Any]]] = {}
    for run_name, exported in exports.items():
        arm, fold = exported["arm"], exported["fold"]
        by_fold.setdefault(fold, {})[arm] = fits[run_name]
    for fold, pair in sorted(by_fold.items()):
        if set(pair) != {"wide", "deep"}:
            continue
        pair_exports = {arm: exports[str(pair[arm]["job"]["id"])] for arm in ("wide", "deep")}
        try:
            _remaining(deadline)
            results["grades"].append(_grade_pair(root, paths, fold, pair, pair_exports, deadline))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            results["grades"].append(
                {"fold": fold, "status": "grade_failed_or_preserved", "error": str(error)}
            )
            if time.time() >= deadline:
                results["stopped_reason"] = "total postprocess deadline expired during grading"
                break
    all_exports = len(exports) == len(jobs)
    required_folds = {
        fold
        for fold in range(4)
        if {job.get("arm") for job in jobs if job.get("fold") == fold} == {"wide", "deep"}
    }
    grade_folds = {
        grade.get("fold") for grade in results["grades"] if grade.get("status") == "graded"
    }
    expected_cells = {(arm, fold) for arm in ("wide", "deep") for fold in range(4)}
    supplied_cells = {(job.get("arm"), job.get("fold")) for job in jobs}
    if (
        len(jobs) == 8
        and supplied_cells == expected_cells
        and all_exports
        and required_folds == grade_folds
        and len(grade_folds) == 4
    ):
        results["status"] = "complete"
    else:
        results["status"] = "partial"
    return results
