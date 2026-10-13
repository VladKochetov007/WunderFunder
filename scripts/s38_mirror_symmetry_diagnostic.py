#!/usr/bin/env python3
"""Bounded feature-only screen of schema-anchored involutive input maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wnn.data import FEATURE_COLUMNS, N_FEATURES, SequenceFile

QUANTILES = np.array([0.01, 0.05, 0.5, 0.95, 0.99], dtype=np.float64)
TAIL_QUANTILES = (0.001, 0.999)
SEED = 2026101338
SAMPLE_COMPONENTS = 32
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_DATA = {
    "bytes": 29240219011,
    "mtime_ns": 1577826000000000000,
    "footer_bytes": 60437285,
    "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
}


def build_maps() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return three exact signed permutations on the documented feature schema."""
    identity = np.arange(N_FEATURES, dtype=np.int64)
    side_perm = identity.copy()
    for base in (0, 52):
        side_perm[base : base + 11] = np.arange(base + 11, base + 22)
        side_perm[base + 11 : base + 22] = np.arange(base, base + 11)
        side_perm[base + 22 : base + 33] = np.arange(base + 33, base + 44)
        side_perm[base + 33 : base + 44] = np.arange(base + 22, base + 33)

    reflection_sign = np.ones(N_FEATURES, dtype=np.float64)
    sign_only_sign = reflection_sign.copy()
    for base in (0, 52):
        reflection_sign[base : base + 22] = -1.0
        reflection_sign[base + 44 : base + 48] = -1.0
        sign_only_sign[base : base + 22] = -1.0
        sign_only_sign[base + 44 : base + 48] = -1.0

    return {
        "side_swap": (side_perm, np.ones(N_FEATURES, dtype=np.float64)),
        "price_axis_reflection": (side_perm, reflection_sign),
        "price_sign_only_negative_control": (identity, sign_only_sign),
    }


def validate_involution(permutation: np.ndarray, sign: np.ndarray) -> None:
    """Raise when a signed permutation does not square to the identity."""
    expected = np.arange(permutation.size, dtype=np.int64)
    if not np.array_equal(permutation[permutation], expected):
        raise ValueError("permutation is not an involution")
    if not np.array_equal(sign * sign[permutation], np.ones_like(sign)):
        raise ValueError("signed permutation is not an involution")


def transform(features: np.ndarray, mapping: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply one signed coordinate permutation without changing row order."""
    permutation, sign = mapping
    return features[:, permutation] * sign[None, :]


def sha256_footer(path: Path, size: int) -> str:
    """Hash the parquet footer and 8-byte trailer with bounded-memory streaming."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(-(size + 8), 2)
        remaining = size + 8
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("train parquet ended before its declared footer")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def load_and_verify_inputs(
    data_root: Path, split_path: Path, proposal_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify frozen split, source metadata, and proposal bindings before reads."""
    proposal_bytes = proposal_path.read_bytes()
    proposal = json.loads(proposal_bytes)
    repo_root = Path(__file__).resolve().parents[1]
    for relative, expected_sha256 in proposal["source_hashes"].items():
        source = repo_root / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"proposal source hash mismatch: {relative}")
    split_bytes = split_path.read_bytes()
    if hashlib.sha256(split_bytes).hexdigest() != EXPECTED_SPLIT_SHA256:
        raise ValueError("split manifest hash differs from frozen proposal")
    split = json.loads(split_bytes)
    data_path = data_root / "datasets/train.parquet"
    stat = data_path.stat()
    observed = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "footer_bytes": EXPECTED_DATA["footer_bytes"],
        "footer_sha256": sha256_footer(data_path, EXPECTED_DATA["footer_bytes"]),
    }
    if observed != EXPECTED_DATA:
        raise ValueError(f"train parquet identity differs from frozen split: {observed}")
    if (
        proposal["source_hashes"]["research/campaign_2026-09-29/artifacts/splits/fold0.json"]
        != EXPECTED_SPLIT_SHA256
    ):
        raise ValueError("proposal does not bind the expected split")
    return (
        split,
        data_path,
        {
            "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
            "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "source_path": str(data_path.resolve()),
            **observed,
        },
    )


def choose_sample(split: dict[str, Any]) -> list[dict[str, int]]:
    """Choose one complete fit-train sequence from each of 32 components."""
    fit_train = set(map(int, split["fit"]["train"]))
    records = [row for row in split["sequences"]["train"] if int(row["row_group"]) in fit_train]
    by_component: dict[int, list[dict[str, int]]] = {}
    for row in records:
        by_component.setdefault(int(row["component"]), []).append(row)
    component_ids = np.array(sorted(by_component), dtype=np.int64)
    if component_ids.size < SAMPLE_COMPONENTS:
        raise ValueError("fewer than 32 source-qualified fitting components")
    rng = np.random.default_rng(SEED)
    selected_components = rng.choice(component_ids, size=SAMPLE_COMPONENTS, replace=False)
    selected: list[dict[str, int]] = []
    for component in selected_components:
        rows = sorted(by_component[int(component)], key=lambda row: int(row["row_group"]))
        selected.append(dict(rows[int(rng.integers(0, len(rows)))]))
    return selected


def validate_group_membership(split: dict[str, Any], row_group_count: int) -> None:
    """Check manifest uniqueness, valid parquet bounds, and complete fit membership."""
    records = split["sequences"]["train"]
    row_groups = [int(record["row_group"]) for record in records]
    if len(set(row_groups)) != len(row_groups):
        raise ValueError("duplicate train row group in split manifest")
    if any(group < 0 or group >= row_group_count for group in row_groups):
        raise ValueError("split manifest train row group is outside the parquet")
    fit_train = set(map(int, split["fit"]["train"]))
    if not fit_train <= set(row_groups):
        raise ValueError("fit-train membership contains a group absent from train records")


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize equally weighted component-level values."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def analyze_sequence(
    features: np.ndarray, mapping: tuple[np.ndarray, np.ndarray]
) -> dict[str, Any]:
    """Compare transformed rows to the same-coordinate original support."""
    if features.shape != (20_000, N_FEATURES):
        raise ValueError(f"unexpected feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("non-finite value in fitting feature sequence")
    candidate = transform(features, mapping)
    if not np.isfinite(candidate).all():
        raise ValueError("map generated non-finite fitting feature values")

    values = features.astype(np.float64, copy=False)
    transformed = candidate.astype(np.float64, copy=False)
    original_q = np.quantile(values, QUANTILES, axis=0)
    candidate_q = np.quantile(transformed, QUANTILES, axis=0)
    robust_scale = np.maximum.reduce(
        [original_q[4] - original_q[0], np.abs(original_q[2]) * 1e-6, np.full(N_FEATURES, 1e-8)]
    )
    normalized_shift = np.max(np.abs(candidate_q - original_q) / robust_scale[None, :], axis=0)
    tail_bounds = np.quantile(values, TAIL_QUANTILES, axis=0)
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    return {
        "finite_values": int(candidate.size),
        "outside_empirical_0_1_99_9_percent_fraction": float(
            np.mean((transformed < tail_bounds[0]) | (transformed > tail_bounds[1]))
        ),
        "outside_empirical_0_1_99_9_percent_by_feature": np.mean(
            (transformed < tail_bounds[0]) | (transformed > tail_bounds[1]), axis=0
        ).tolist(),
        "outside_observed_min_max_fraction": float(
            np.mean((transformed < lower) | (transformed > upper))
        ),
        "outside_observed_min_max_by_feature": np.mean(
            (transformed < lower) | (transformed > upper), axis=0
        ).tolist(),
        "normalized_quantile_displacement_by_feature": normalized_shift.tolist(),
        "features_with_displacement_gt_1": int(np.count_nonzero(normalized_shift > 1.0)),
        "median_feature_displacement": float(np.median(normalized_shift)),
        "p90_feature_displacement": float(np.quantile(normalized_shift, 0.9)),
        "p95_feature_displacement": float(np.quantile(normalized_shift, 0.95)),
        "max_feature_displacement": float(np.max(normalized_shift)),
    }


def run_diagnostic(data_root: Path, split_path: Path, proposal_path: Path) -> dict[str, Any]:
    """Run the frozen complete-sequence feature-only screen and return a receipt."""
    split, data_path, identity = load_and_verify_inputs(data_root, split_path, proposal_path)
    reader = SequenceFile(data_path)
    validate_group_membership(split, len(reader))
    sample = choose_sample(split)
    maps = build_maps()
    for mapping in maps.values():
        validate_involution(*mapping)

    accumulators: dict[str, dict[str, list[Any]]] = {
        name: {
            "components": [],
            "feature_displacements": [],
            "feature_tail_fractions": [],
            "feature_minmax_fractions": [],
        }
        for name in maps
    }
    for record in sample:
        group = int(record["row_group"])
        sequence = reader.read(group, targets=False, mask=False)
        if sequence.seq_ix != int(record["seq_ix"]):
            raise ValueError(f"sequence identity mismatch at row group {group}")
        for name, mapping in maps.items():
            metrics = analyze_sequence(sequence.features, mapping)
            accumulator = accumulators[name]
            accumulator["feature_displacements"].append(
                metrics.pop("normalized_quantile_displacement_by_feature")
            )
            accumulator["feature_tail_fractions"].append(
                metrics.pop("outside_empirical_0_1_99_9_percent_by_feature")
            )
            accumulator["feature_minmax_fractions"].append(
                metrics.pop("outside_observed_min_max_by_feature")
            )
            accumulator["components"].append(
                {
                    "component": int(record["component"]),
                    "row_group": group,
                    "seq_ix": sequence.seq_ix,
                    "rows": int(sequence.features.shape[0]),
                    **metrics,
                }
            )
        del sequence

    per_map: dict[str, Any] = {}
    for name, accumulator in accumulators.items():
        component_rows = accumulator["components"]
        feature_rows = accumulator["feature_displacements"]
        tail_feature_rows = accumulator["feature_tail_fractions"]
        minmax_feature_rows = accumulator["feature_minmax_fractions"]
        feature_array = np.asarray(feature_rows, dtype=np.float64)
        tail_feature_array = np.asarray(tail_feature_rows, dtype=np.float64)
        minmax_feature_array = np.asarray(minmax_feature_rows, dtype=np.float64)
        per_feature = []
        for index, column in enumerate(FEATURE_COLUMNS):
            values = feature_array[:, index]
            per_feature.append(
                {
                    "feature": column,
                    "normalized_quantile_displacement": summarize(values.tolist()),
                    "outside_empirical_0_1_99_9_percent_fraction": summarize(
                        tail_feature_array[:, index].tolist()
                    ),
                    "outside_observed_min_max_fraction": summarize(
                        minmax_feature_array[:, index].tolist()
                    ),
                }
            )
        component_summary = {
            key: summarize([float(row[key]) for row in component_rows])
            for key in (
                "outside_empirical_0_1_99_9_percent_fraction",
                "outside_observed_min_max_fraction",
                "features_with_displacement_gt_1",
                "median_feature_displacement",
                "p90_feature_displacement",
                "p95_feature_displacement",
                "max_feature_displacement",
            )
        }
        qualifying = sum(row["features_with_displacement_gt_1"] > 56 for row in component_rows)
        per_map[name] = {
            "component_summary": component_summary,
            "per_feature_equal_component_summary": per_feature,
            "components_with_more_than_56_features_over_threshold": qualifying,
            "predeclared_strong_marginal_mismatch": qualifying >= 16,
            "per_component": component_rows,
        }

    return {
        "experiment_id": "E-STRATEGY-S38-H317-INPUT-SYMMETRY-DIAGNOSTIC-R1",
        "proposal_id": "S38-H317-mirror-symmetry-diagnostic-r1",
        "created_utc": datetime.now(UTC).isoformat(),
        "result_type": "feature_only_input_marginal_screen",
        "data_identity": identity,
        "fit_split": {
            "manifest_sha256": identity["split_sha256"],
            "split": "train",
            "fit_train_groups": len(split["fit"]["train"]),
            "sample_components": len(sample),
            "seed": SEED,
            "rng": "numpy.random.default_rng",
            "sample": sample,
        },
        "features": {"count": N_FEATURES, "columns": FEATURE_COLUMNS},
        "maps": {name: {"involution_verified": True, **values} for name, values in per_map.items()},
        "limits": {
            "labels_read": False,
            "validation_or_evaluation_read": False,
            "training_or_model_inference": False,
            "joint_or_target_symmetry_established": False,
            "augmentation_authorized": False,
            "interpretation": "A marginal screen on one deterministic 32-component sample; matching does not establish row-level or target symmetry.",
        },
    }


def main() -> None:
    """Run the diagnostic and write a strict-JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(args.data_root, args.split_manifest, args.proposal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
