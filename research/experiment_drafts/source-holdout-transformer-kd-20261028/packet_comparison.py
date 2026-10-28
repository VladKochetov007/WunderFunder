"""Shared fixed-endpoint scoring and conditional component contrasts."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from packet_contract import REPO, read, require
from packet_lineage import validate_cache, validate_evaluation, validate_terminal

# Canonical score receipts contain heterogeneous JSON, checked before use.


def read_score(root: Path, cell: str) -> dict[str, Any]:
    """Verify immutable lineage and return one canonical full-valid score receipt."""
    directory = root / cell
    actual = validate_evaluation(root, cell)
    require(read(directory / "evaluation_receipt.json") == actual, "Evaluation receipt changed")
    score = read(directory / "score.json")
    metrics = score["score"]
    require(metrics.get("metric_id") == "global_wp_clip2_v1", "Official metric required")
    for name in ("wp", "t0", "t1"):
        value = metrics.get(name)
        require(type(value) in (float, int) and math.isfinite(value), "Invalid score")
        require(-1 <= value <= 1, "Correlation outside [-1,1]")
    require(abs(metrics["wp"] - (metrics["t0"] + metrics["t1"]) / 2) < 1e-12, "WP differs")
    require(type(metrics.get("selected_rows")) is int and metrics["selected_rows"] > 0, "No rows")
    require(bool(score.get("component_centered_moments")), "Missing component statistics")
    if cell.startswith("kd_"):
        teacher = validate_cache(root, cell)
        identity = validate_terminal(root, cell)["identity"]
        require(identity["data"].get("teacher") == teacher, "Student teacher lineage changed")
    return score


def paired_contrast(
    candidate: dict[str, Any], reference: dict[str, Any], bootstrap_seed: int
) -> dict[str, Any]:
    """Resample the same graph components for two already fitted endpoints."""
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from wnn.evaluation import paired_component_bootstrap

    a, b = candidate["component_centered_moments"], reference["component_centered_moments"]
    require(set(a) == set(b), "Component coverage differs")
    keys = sorted(a, key=int)
    candidate_moments = np.asarray([a[key] for key in keys], dtype=np.float64)
    reference_moments = np.asarray([b[key] for key in keys], dtype=np.float64)
    require(candidate_moments.shape == (len(keys), 2, 6), "Malformed component moments")
    require(reference_moments.shape == candidate_moments.shape, "Moment shape differs")
    require(
        np.isfinite(candidate_moments).all() and np.isfinite(reference_moments).all(), "NaN moments"
    )
    # Weight, target mean and target centered variance must identify the same labels.
    require(
        np.allclose(
            candidate_moments[..., [0, 1, 3]],
            reference_moments[..., [0, 1, 3]],
            rtol=1e-10,
            atol=1e-10,
        ),
        "Target statistics differ between paired endpoints",
    )
    interval = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        np.asarray([int(key) for key in keys]),
        n_boot=2000,
        seed=bootstrap_seed,
        centered=True,
    )
    for side, score in (("candidate_wp", candidate), ("reference_wp", reference)):
        require(abs(interval[side] - score["score"]["wp"]) < 1e-9, "Moments do not reproduce score")
    return {
        "delta": {
            name: candidate["score"][name] - reference["score"][name] for name in ("wp", "t0", "t1")
        },
        "conditional_component_bootstrap": interval,
        "bootstrap_input_grain": "component; library n_sequences counts aggregated component records",
        "interval_scope": "Fixed predictions, graph-component resampling only; not training variance, pristine holdout or independent seed/fold evidence.",
    }
