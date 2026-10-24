"""Fixed-alpha scoring helpers for the F0 correction-shrink diagnostic."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any  # Recipe contrast records are runtime-defined JSON.

import numpy as np
from f0_grade_preflight import PreparedF0Grade
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
CELL_NAMES = ("A_lag1", "B_lag1", "C_lag1", "B_lag16", "C_lag16")
PREDICTION_RUNS = {
    "A_lag1": "online_rel_f0_s1001_lag1_a_r1",
    "B_lag1": "online_rel_f0_s1001_lag1_b_r1",
    "C_lag1": "online_rel_f0_s1001_lag1_c_r1",
    "B_lag16": "online_rel_f0_s1001_lag16_b_r1",
    "C_lag16": "online_rel_f0_s1001_lag16_c_r1",
}
RAW_ENDPOINTS = {cell: f"raw_{cell}" for cell in CELL_NAMES}
SHRUNK_ENDPOINTS = {cell: f"shrink_{cell}" for cell in CELL_NAMES}


def validate_alpha(value: Any) -> None:
    """Require the exact conceptual coefficient and its FP32 implementation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0.1:
        raise ValueError("recipe alpha must be exactly the fixed decimal 0.1")
    if np.float32(value) != ALPHA:
        raise ValueError("recipe alpha does not map to the scorer's FP32 coefficient")


def expected_contrasts() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return the exact five primary and thirteen secondary comparisons."""
    primary = [
        {
            "name": f"shrink_{cell}-minus-parent",
            "candidate": SHRUNK_ENDPOINTS[cell],
            "reference": "parent",
            "role": "primary",
        }
        for cell in CELL_NAMES
    ]
    secondary = [
        {
            "name": f"shrink_{cell}-minus-raw_{cell}",
            "candidate": SHRUNK_ENDPOINTS[cell],
            "reference": RAW_ENDPOINTS[cell],
            "role": "secondary",
        }
        for cell in CELL_NAMES
    ]
    for role, endpoints in (("raw", RAW_ENDPOINTS), ("shrink", SHRUNK_ENDPOINTS)):
        for lag in (1, 16):
            for candidate, reference in (("C", "B"), ("B", "A")):
                candidate_cell = f"{candidate}_lag{lag}"
                reference_cell = f"{reference}_lag{lag}"
                if reference_cell == "A_lag16":
                    reference_cell = "A_lag1"
                secondary.append(
                    {
                        "name": f"{role}_{candidate}-minus-{reference}_lag{lag}",
                        "candidate": endpoints[candidate_cell],
                        "reference": endpoints[reference_cell],
                        "role": "secondary",
                    }
                )
    return primary, secondary


def validate_contrasts(recipe: dict[str, Any]) -> list[dict[str, str]]:
    """Require the recipe to match the fixed primary and descriptive pairs."""
    expected_primary, expected_secondary = expected_contrasts()
    if recipe["primary_contrasts"] != expected_primary:
        raise ValueError("recipe primary contrasts differ from the five frozen shrink pairs")
    if recipe["secondary_contrasts"] != expected_secondary:
        raise ValueError("recipe secondary contrasts differ from the frozen raw/incremental pairs")
    return [*expected_primary, *expected_secondary]


def score_endpoints(prepared: PreparedF0Grade) -> dict[str, Any]:
    """Score raw and shrunken outputs with the original mask and metric helper."""
    expected_target_sha = prepared.parent_recipe["target_cache"]["sha256"]
    observed_before = sha256_file(prepared.target_cache_path)
    if observed_before != expected_target_sha:
        raise ValueError("official target archive differs from its frozen source hash")
    scratch = Path(prepared.parent_recipe["resource_contract"]["scratch"]).resolve()
    if not scratch.is_dir():
        raise ValueError("approved physical scratch root is unavailable")
    endpoint_names = ["parent", *RAW_ENDPOINTS.values(), *SHRUNK_ENDPOINTS.values()]
    arrays = {
        cell: np.load(prepared.candidate_prediction_paths[cell], mmap_mode="r", allow_pickle=False)
        for cell in CELL_NAMES
    }
    parent = np.load(prepared.parent_prediction_path, mmap_mode="r", allow_pickle=False)
    accumulators = {
        name: {band: GlobalAccumulator() for band in TIME_BINS} for name in endpoint_names
    }
    moments = {
        name: np.zeros((len(prepared.evaluation_groups), 2, 6), dtype=np.float64)
        for name in endpoint_names
    }
    rows = np.arange(ROWS)
    with tempfile.TemporaryDirectory(prefix="correction-shrink-f0-", dir=scratch) as temporary:
        targets, masks, row_groups, positions, _ = _materialize_target_maps(
            prepared, Path(temporary)
        )
        del row_groups
        for group_index, group in enumerate(prepared.evaluation_groups):
            target_row = targets[positions[group]]
            mask_row = masks[positions[group]]
            parent_row = parent[group_index]
            endpoint_rows: dict[str, np.ndarray] = {"parent": parent_row}
            band_masks = {
                band: np.asarray(mask_row, dtype=bool) & (rows >= start) & (rows < stop)
                for band, (start, stop) in TIME_BINS.items()
            }
            for cell, prediction_map in arrays.items():
                raw_row = prediction_map[group_index]
                shrunk_row = parent_row + ALPHA * (raw_row - parent_row)
                endpoint_rows[RAW_ENDPOINTS[cell]] = raw_row
                endpoint_rows[SHRUNK_ENDPOINTS[cell]] = shrunk_row
            scored_mask = _official_mask(mask_row, 99, ROWS)
            for name in endpoint_names:
                prediction_row = endpoint_rows[name]
                for band, selected in band_masks.items():
                    accumulators[name][band].add(target_row, prediction_row, selected)
                endpoint_moments, _ = paired_component_moments(
                    target_row[None, ...],
                    prediction_row[None, ...],
                    prediction_row[None, ...],
                    scored_mask[None, ...],
                )
                moments[name][group_index] = endpoint_moments[0]
        endpoint_scores = {
            name: {band: accumulator.result() for band, accumulator in band_acc.items()}
            for name, band_acc in accumulators.items()
        }
        observed_after = sha256_file(prepared.target_cache_path)
        target_identity = verify_target_hashes(expected_target_sha, observed_before, observed_after)
    return {
        "endpoint_scores": endpoint_scores,
        "moments": moments,
        "target_cache_identity": target_identity,
    }


def compare_raw_scores(scores: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Require freshly accumulated raw scores to reproduce the prior F0 grade."""
    for cell in CELL_NAMES:
        for band in TIME_BINS:
            observed = scores[RAW_ENDPOINTS[cell]][band]
            expected = baseline["endpoint_scores"][cell][band]
            if observed["selected_rows"] != expected["selected_rows"]:
                raise ValueError(f"raw baseline row count differs for {cell}/{band}")
            for metric in ("wp", "t0", "t1"):
                if abs(observed[metric] - expected[metric]) > 1e-12:
                    raise ValueError(
                        f"raw baseline {metric} differs from F0 grade for {cell}/{band}"
                    )


def build_result(
    recipe: dict[str, Any],
    recipe_sha256: str,
    prepared: PreparedF0Grade,
    scores: dict[str, Any],
    admission: dict[str, Any],
    admission_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    """Build the fixed-alpha report and conditional component intervals."""
    endpoint_scores = scores["endpoint_scores"]
    moments = scores["moments"]
    contrasts: list[dict[str, Any]] = []
    for definition in validate_contrasts(recipe):
        candidate, reference = definition["candidate"], definition["reference"]
        interval = paired_component_bootstrap(
            moments[candidate],
            moments[reference],
            prepared.components,
            n_boot=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        deltas: dict[str, Any] = {}
        for band in TIME_BINS:
            candidate_score = endpoint_scores[candidate][band]
            reference_score = endpoint_scores[reference][band]
            if candidate_score["selected_rows"] != reference_score["selected_rows"]:
                raise ValueError(f"contrast {definition['name']} has mismatched scored rows")
            deltas[band] = {
                "wp": candidate_score["wp"] - reference_score["wp"],
                "t0": candidate_score["t0"] - reference_score["t0"],
                "t1": candidate_score["t1"] - reference_score["t1"],
                "selected_rows": candidate_score["selected_rows"],
            }
        contrasts.append(
            {
                **definition,
                "point_wp_delta": deltas["overall"]["wp"],
                "point_t0_delta": deltas["overall"]["t0"],
                "point_t1_delta": deltas["overall"]["t1"],
                "paired_component_bootstrap": interval,
                "time_bin_deltas": deltas,
                "uncertainty_scope": "conditional on these fixed exported endpoints",
            }
        )
    return {
        "schema": "wnn.correction-shrink-f0-grade.v1",
        "status": "graded_exploratory_fixed_correction_shrink_f0",
        "recipe_sha256": recipe_sha256,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "base_recipe_sha256": recipe["base_recipe"]["sha256"],
        "base_source_manifest_sha256": recipe["base_source_manifest"]["sha256"],
        "baseline_grade_sha256": recipe["baseline_grade"]["sha256"],
        "preflight_receipt_sha256": recipe["preflight_receipt"]["sha256"],
        "admission": {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "preflight_receipt_sha256": sha256_file(preflight_path),
            "root_authorized": admission["target_quality_reads_authorized"],
        },
        "metric": "global_wp_clip2_v1",
        "metric_source": "src/wnn/gmetric.py",
        "score_mask": "official mask AND row >= 99",
        "clip": [-2.0, 2.0],
        "weight": "absolute clipped target magnitude",
        "fold": 0,
        "seed": 1001,
        "evaluation_groups": len(prepared.evaluation_groups),
        "evaluation_components": int(np.unique(prepared.components).size),
        "new_fit_count": 0,
        "shrink": {
            "alpha": 0.1,
            "operation": "parent + alpha * (raw_candidate_prediction - parent)",
            "dtype": "float32",
            "coefficient_search": False,
            "unique_candidates": list(CELL_NAMES),
            "omitted_duplicate": {
                "cell": "A_lag16",
                "alias": "A_lag1",
                "identity": "exact prediction SHA-256 equality was required by preflight",
            },
        },
        "baseline_raw_scores": {cell: endpoint_scores[RAW_ENDPOINTS[cell]] for cell in CELL_NAMES},
        "shrunken_scores": {cell: endpoint_scores[SHRUNK_ENDPOINTS[cell]] for cell in CELL_NAMES},
        "candidate_prediction_sources": {
            cell: recipe["candidate_predictions"][cell] for cell in CELL_NAMES
        },
        "target_cache_identity": {
            "path": str(prepared.target_cache_path),
            **scores["target_cache_identity"],
            "verified_before_label_reads": True,
            "verified_after_scoring": True,
        },
        "contrasts": contrasts,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "exposure": "F0 is exposed; this is a post-original-grade fixed-prediction diagnostic.",
        "limits": [
            "One exposed fold and one fitting seed do not estimate fold or training variability.",
            "Component intervals condition on fixed exported predictions and are not a kill gate.",
            "No new fit, inference parity check, public score, or coefficient optimization is performed.",
        ],
    }
