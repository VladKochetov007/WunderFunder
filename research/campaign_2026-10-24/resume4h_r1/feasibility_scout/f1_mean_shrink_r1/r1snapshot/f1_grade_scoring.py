"""Score the fixed F1 raw, parent, and 0.1-shrunk endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from f0_grade_scoring import (
    ROWS,
    TIME_BINS,
    _materialize_target_maps,
    _official_mask,
    verify_target_hashes,
)
from online_relationship_grade_core import sha256_file
from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import GlobalAccumulator

ALPHA = np.float32(0.1)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271_828
ENDPOINTS = {
    "raw_A_lag1": "candidate",
    "parent": "parent",
    "shrink_A_lag1": "shrink",
}
CONTRASTS = (
    {"name": "shrink_A_lag1-minus-parent", "candidate": "shrink_A_lag1", "reference": "parent", "role": "primary"},
    {"name": "raw_A_lag1-minus-parent", "candidate": "raw_A_lag1", "reference": "parent", "role": "secondary"},
    {"name": "shrink_A_lag1-minus-raw_A_lag1", "candidate": "shrink_A_lag1", "reference": "raw_A_lag1", "role": "secondary"},
)


def _band_masks(mask_row: np.ndarray) -> dict[str, np.ndarray]:
    """Build the official row mask and four unchanged position bands."""
    rows = np.arange(ROWS)
    return {
        name: np.asarray(mask_row, dtype=bool) & (rows >= start) & (rows < stop)
        for name, (start, stop) in TIME_BINS.items()
    }


def score_f1_endpoints(prepared: Any) -> dict[str, Any]:
    """Accumulate official metrics and paired component intervals group by group."""
    recipe = prepared.recipe
    target_record = recipe["target_cache"]
    target_path = prepared.target_cache_path
    before_sha = sha256_file(target_path)
    if before_sha != target_record["sha256"]:
        raise ValueError("official target archive differs from its frozen source hash")

    scratch = Path(recipe["resource_contract"]["scratch"]).resolve()
    if not scratch.is_dir():
        raise ValueError("approved physical scratch root is unavailable")
    inputs = SimpleNamespace(
        target_cache_path=target_path,
        evaluation_groups=prepared.evaluation_groups,
        parent_recipe={
            "target_cache": target_record,
            "resource_contract": recipe["resource_contract"],
        },
    )
    parent = np.load(prepared.parent_prediction_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(prepared.candidate_prediction_path, mmap_mode="r", allow_pickle=False)
    endpoint_names = tuple(ENDPOINTS)
    accumulators = {
        name: {band: GlobalAccumulator() for band in TIME_BINS} for name in endpoint_names
    }
    moments = {
        name: np.zeros((len(prepared.evaluation_groups), 2, 6), dtype=np.float64)
        for name in endpoint_names
    }

    with tempfile.TemporaryDirectory(prefix="f1-mean-shrink-targets-", dir=scratch) as temp:
        targets, masks, _, target_positions, _ = _materialize_target_maps(inputs, Path(temp))
        for index, group in enumerate(prepared.evaluation_groups):
            target_row = targets[target_positions[group]]
            mask_row = masks[target_positions[group]]
            parent_row = parent[index]
            candidate_row = candidate[index]
            shrink_row = parent_row + ALPHA * (candidate_row - parent_row)
            endpoint_rows = {
                "raw_A_lag1": candidate_row,
                "parent": parent_row,
                "shrink_A_lag1": shrink_row,
            }
            for band, selected in _band_masks(mask_row).items():
                for name, prediction_row in endpoint_rows.items():
                    accumulators[name][band].add(target_row, prediction_row, selected)
            scored = _official_mask(mask_row, 99, ROWS)
            for name, prediction_row in endpoint_rows.items():
                value, _ = paired_component_moments(
                    target_row[None, ...],
                    prediction_row[None, ...],
                    prediction_row[None, ...],
                    scored[None, ...],
                )
                moments[name][index] = value[0]

    endpoint_scores = {
        name: {band: accumulator.result() for band, accumulator in by_band.items()}
        for name, by_band in accumulators.items()
    }
    contrasts = []
    for definition in CONTRASTS:
        candidate_name, reference_name = definition["candidate"], definition["reference"]
        interval = paired_component_bootstrap(
            moments[candidate_name],
            moments[reference_name],
            np.asarray(prepared.components, dtype=np.int64),
            n_boot=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        band_deltas: dict[str, dict[str, float | int]] = {}
        for band in TIME_BINS:
            left = endpoint_scores[candidate_name][band]
            right = endpoint_scores[reference_name][band]
            if left["selected_rows"] != right["selected_rows"]:
                raise ValueError(f"contrast {definition['name']} has mismatched scored rows")
            band_deltas[band] = {
                metric: left[metric] - right[metric] for metric in ("wp", "t0", "t1")
            }
            band_deltas[band]["selected_rows"] = left["selected_rows"]
        contrasts.append(
            {
                **definition,
                "point_wp_delta": band_deltas["overall"]["wp"],
                "point_t0_delta": band_deltas["overall"]["t0"],
                "point_t1_delta": band_deltas["overall"]["t1"],
                "paired_component_bootstrap": interval,
                "time_bin_deltas": band_deltas,
                "uncertainty_scope": "conditional on these fixed exposed endpoints",
            }
        )

    after_sha = sha256_file(target_path)
    target_identity = verify_target_hashes(target_record["sha256"], before_sha, after_sha)
    prediction_hashes = {
        "candidate": sha256_file(prepared.candidate_prediction_path),
        "parent": sha256_file(prepared.parent_prediction_path),
    }
    expected_hashes = {
        "candidate": recipe["inputs"]["candidate"]["prediction_sha256"],
        "parent": recipe["inputs"]["parent"]["prediction_sha256"],
    }
    if prediction_hashes != expected_hashes:
        raise ValueError("prediction arrays changed during F1 scoring")
    return {
        "endpoint_scores": endpoint_scores,
        "contrasts": contrasts,
        "target_cache_identity": target_identity,
        "prediction_sha256_after_score": prediction_hashes,
    }

