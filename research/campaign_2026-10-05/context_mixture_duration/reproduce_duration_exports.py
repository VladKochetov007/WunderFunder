"""Independently reproduce E488 scores from raw exports with shifted NumPy sums.

This checker imports neither the campaign grader nor ``wnn.evaluation``. It is
intended to verify an already-written E488 score/export pair using the
evaluation arrays in that raw export. It does not access Parquet or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (271839, 271840, 271841)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix")
EPOCHS = (1, 3)
ROWS = 20_000
COMPONENTS = 16
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 271_839
TOLERANCE = 1e-10
CONTRASTS = (
    ("recurrent_minus_static", "recurrent_mix", "static_mix"),
    ("recurrent_minus_fixed", "recurrent_mix", "fixed_mix"),
    ("static_minus_fixed", "static_mix", "fixed_mix"),
    ("fixed_minus_gelu", "fixed_mix", "gelu"),
)


def require(ok: bool, message: str) -> None:
    """Raise on malformed raw exports or score records."""
    if not ok:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    """Hash one completed result/export using bounded-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_near(actual: Any, expected: Any, label: str) -> float:
    """Reject discrepancies larger than registered summation tolerance."""
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"{label}: shape differs {actual_array.shape} != {expected_array.shape}")
    error = float(np.max(np.abs(actual_array - expected_array), initial=0.0))
    if not math.isfinite(error) or error > TOLERANCE:
        raise ValueError(f"{label}: reproduction discrepancy {error:.3g}")
    return error


def component_statistics(
    targets: np.ndarray,
    predictions: np.ndarray,
    selected_mask: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Build additive weighted-Pearson statistics around shared first-row anchors.

    Arrays use the raw export's component/time/target layout. Statistics are
    shifted before squaring, independently of the scorer's centered Welford
    component moments.
    """
    raw_y = np.asarray(targets)
    raw_p = np.asarray(predictions)
    mask = np.asarray(selected_mask)
    counts = np.asarray(counts)
    require(raw_y.ndim == 3 and raw_y.shape[-1] == 2, "target export must be [component,row,2]")
    require(raw_p.shape == raw_y.shape, "prediction export shape differs")
    require(mask.shape == raw_y.shape[:2] and mask.dtype == np.bool_, "selected mask differs")
    require(counts.shape == (raw_y.shape[0],), "component count vector differs")
    require(np.array_equal(mask.sum(axis=1), counts), "selection mask/count mismatch")
    require((counts > 0).all(), "component selection count is empty")
    require(np.isfinite(raw_y).all() and np.isfinite(raw_p).all(), "raw export is nonfinite")

    y32 = raw_y.astype(np.float32)
    p32 = raw_p.astype(np.float32)
    require(np.isfinite(y32).all() and np.isfinite(p32).all(), "raw export overflows float32")
    y = np.clip(y32[mask], -2.0, 2.0).astype(np.float64)
    p = np.clip(p32[mask], -2.0, 2.0).astype(np.float64)
    require(
        y.shape == p.shape == (int(counts.sum()), 2), "selected target/prediction shape differs"
    )
    weights = np.abs(y)
    delta_y = y - y[0]
    delta_p = p - p[0]
    rows = np.stack(
        (
            weights,
            weights * delta_y,
            weights * delta_p,
            weights * np.square(delta_y),
            weights * np.square(delta_p),
            weights * delta_y * delta_p,
        ),
        axis=-1,
    )
    boundaries = np.cumsum(np.r_[0, counts])
    return np.stack(
        [
            rows[start:stop].sum(axis=0, dtype=np.float64)
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )


def scores(statistics: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    """Pool shifted statistics and return weighted correlations by target/draw."""
    stats = np.asarray(statistics, dtype=np.float64)
    counts = np.asarray(multiplicities)
    require(stats.ndim == 3 and stats.shape[1:] == (2, 6), "moments must be [component,2,6]")
    require(counts.ndim == 2 and counts.shape[1] == len(stats), "draw multiplicities differ")
    require(
        np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all(), "invalid multiplicities"
    )
    require(np.isfinite(stats).all(), "shifted statistics are nonfinite")

    pooled = np.einsum("bc,ctm->btm", counts, stats, optimize=True)
    total, sy, sp, syy, spp, syp = np.moveaxis(pooled, -1, 0)
    safe = np.where(total > 1e-8, total, 1.0)
    var_y = np.maximum(syy - np.square(sy) / safe, 0.0)
    var_p = np.maximum(spp - np.square(sp) / safe, 0.0)
    covariance = syp - sy * sp / safe
    denom = np.sqrt(var_y * var_p)
    valid = (total >= 1e-8) & (np.sqrt(var_y / safe) > 1e-8)
    valid &= np.sqrt(var_p / safe) > 1e-8
    result = np.zeros_like(covariance)
    np.divide(covariance, denom, out=result, where=valid)
    return np.clip(result, -1.0, 1.0)


def component_draws(
    seed: int = BOOTSTRAP_SEED, n_components: int = COMPONENTS
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate all paired draws and their multiplicities from the registered seed."""
    require(n_components > 0, "component count must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_components, size=(BOOTSTRAP_COUNT, n_components))
    multiplicities = np.stack([np.bincount(row, minlength=n_components) for row in indices]).astype(
        np.int64
    )
    return indices, multiplicities


def summary(point: float, draws: np.ndarray, seed: int, target: str | None) -> dict[str, Any]:
    """Construct the grader's percentile/SD/sign summary for 2,000 draws."""
    values = np.asarray(draws, dtype=np.float64)
    require(values.shape == (BOOTSTRAP_COUNT,) and np.isfinite(values).all(), "bad bootstrap draws")
    return {
        "point_diff": float(point),
        "bootstrap_sd": float(values.std(ddof=1)),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(values <= 0)),
        "seed": seed,
        "target": target,
        "components": COMPONENTS,
        "bootstrap_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metric_id": "global_wp_clip2_v1",
    }


def expected_arm_keys() -> set[str]:
    """Enumerate the 24 seed/arm/epoch saved predictions."""
    return {
        f"seed{seed}__{arm}__epoch{epoch}" for seed in SEEDS for arm in ARMS for epoch in EPOCHS
    }


def expected_contrast_records() -> dict[str, dict[str, Any]]:
    """Enumerate every preregistered contrast and exact candidate/reference mapping."""
    records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for epoch in EPOCHS:
            for name, candidate, reference in CONTRASTS:
                key = f"seed{seed}__epoch{epoch}__{name}"
                records[key] = {
                    "seed": seed,
                    "epoch": epoch,
                    "contrast": name,
                    "candidate": candidate,
                    "reference": reference,
                    "terms": (
                        (f"seed{seed}__{candidate}__epoch{epoch}", 1),
                        (f"seed{seed}__{reference}__epoch{epoch}", -1),
                    ),
                }
        key = f"seed{seed}__duration_interaction"
        records[key] = {
            "seed": seed,
            "epoch": None,
            "contrast": "(recurrent_3 - static_3) - (recurrent_1 - static_1)",
            "candidate": "recurrent_minus_static_epoch3",
            "reference": "recurrent_minus_static_epoch1",
            "terms": (
                (f"seed{seed}__recurrent_mix__epoch3", 1),
                (f"seed{seed}__static_mix__epoch3", -1),
                (f"seed{seed}__recurrent_mix__epoch1", -1),
                (f"seed{seed}__static_mix__epoch1", 1),
            ),
        }
    return records


def _same_record(actual: dict[str, Any], expected: dict[str, Any], label: str) -> float:
    """Verify registered contrast identity and all bootstrap summary fields."""
    for key in ("seed", "epoch", "contrast", "candidate", "reference"):
        require(actual.get(key) == expected[key], f"{label} {key} identity differs")
    require(actual.get("components") == COMPONENTS, f"{label} component count differs")
    require(actual.get("bootstrap_count") == BOOTSTRAP_COUNT, f"{label} draw count differs")
    require(actual.get("bootstrap_seed") == BOOTSTRAP_SEED, f"{label} bootstrap seed differs")
    require(actual.get("metric_id") == "global_wp_clip2_v1", f"{label} metric ID differs")
    return 0.0


def verify_payload(
    result: dict[str, Any], archive: Any, expected_rows: int = ROWS
) -> dict[str, Any]:
    """Recompute all point scores and 2,000-draw paired contrast summaries."""
    require(result.get("kind") == "independent E488 duration evaluation", "wrong E488 score kind")
    require(result.get("metric_id") == "global_wp_clip2_v1", "wrong E488 score metric")
    require(
        result.get("trajectory_count") == 12 and result.get("snapshot_count") == 24,
        "E488 score inventory differs",
    )
    require(result.get("no_seed_pooling") is True, "E488 result claims pooled seeds")
    require(
        result.get("registered_global_contrast_count") == 27,
        "registered global contrast count differs",
    )
    require(
        result.get("registered_target_contrast_count") == 27,
        "registered target contrast count differs",
    )
    bootstrap = result.get("bootstrap", {})
    require(bootstrap.get("draws") == BOOTSTRAP_COUNT, "registered bootstrap draw count differs")
    require(bootstrap.get("seed") == BOOTSTRAP_SEED, "registered bootstrap seed differs")
    require(
        bootstrap.get("multiplicities_shared_across_all_24_snapshots_and_contrasts") is True,
        "bootstrap multiplicities are not registered as shared",
    )
    require(
        bootstrap.get("unit") == "numeric-sorted fixed component ID",
        "bootstrap resampling unit differs",
    )
    require(
        result.get("test_labels_used_for_fitting_or_selection") is False,
        "test labels used for fitting",
    )
    required_archive = {
        "component_ids",
        "sorted_component_ids",
        "row_groups",
        "raw_targets",
        "selected_mask",
        "bootstrap_component_indices",
        "bootstrap_multiplicities",
    }
    arm_keys = expected_arm_keys()
    expected_arrays = required_archive | {f"prediction__{key}" for key in arm_keys}
    require(set(archive.files) == expected_arrays, "E488 raw prediction archive inventory differs")

    component_ids = np.asarray(archive["component_ids"])
    sorted_ids = np.asarray(archive["sorted_component_ids"])
    row_groups = np.asarray(archive["row_groups"])
    raw_targets = np.asarray(archive["raw_targets"])
    selected_mask = np.asarray(archive["selected_mask"])
    require(
        component_ids.shape == sorted_ids.shape == row_groups.shape == (COMPONENTS,),
        "E488 component metadata shape differs",
    )
    require(len(np.unique(component_ids)) == COMPONENTS, "E488 test component IDs are not unique")
    require(len(np.unique(row_groups)) == COMPONENTS, "E488 test row groups are not unique")
    require(
        np.array_equal(sorted_ids, np.sort(component_ids, kind="stable")), "E488 sorted IDs differ"
    )
    require(
        np.issubdtype(component_ids.dtype, np.integer)
        and np.issubdtype(row_groups.dtype, np.integer),
        "E488 component metadata is noninteger",
    )
    require(
        raw_targets.shape == (COMPONENTS, expected_rows, 2), "E488 raw target array shape differs"
    )
    require(
        selected_mask.shape == (COMPONENTS, expected_rows) and selected_mask.dtype == np.bool_,
        "E488 selection mask differs",
    )
    require(np.isfinite(raw_targets).all(), "E488 raw exported targets are nonfinite")
    counts = selected_mask.sum(axis=1).astype(np.int64)
    require((counts > 0).all(), "E488 test component has no scored rows")
    order = np.argsort(component_ids, kind="stable")
    expected_indices, multiplicities = component_draws()
    require(
        np.array_equal(archive["bootstrap_component_indices"], expected_indices),
        "E488 component draw indices differ",
    )
    require(
        np.array_equal(archive["bootstrap_multiplicities"], multiplicities),
        "E488 draw multiplicities differ",
    )
    require(
        result.get("component_ids") == component_ids.tolist(), "E488 result component order differs"
    )
    require(
        result.get("sorted_component_ids") == sorted_ids.tolist(),
        "E488 result sorted component order differs",
    )
    require(result.get("row_groups") == row_groups.tolist(), "E488 result test row groups differ")
    require(
        result.get("component_row_counts") == counts.tolist(), "E488 result selected counts differ"
    )
    require(
        result.get("selected_rows_total") == int(counts.sum()),
        "E488 result selected row total differs",
    )

    result_arms = result.get("arm_scores", {})
    require(set(result_arms) == arm_keys, "E488 score arm keys differ")
    global_points: dict[str, float] = {}
    target_points: dict[str, np.ndarray] = {}
    target_draws: dict[str, np.ndarray] = {}
    expected_counts = counts.tolist()
    max_error = 0.0
    draw_hashes: dict[str, str] = {}
    for arm_key in sorted(arm_keys):
        prediction = np.asarray(archive[f"prediction__{arm_key}"])
        require(prediction.shape == raw_targets.shape, f"E488 prediction shape differs: {arm_key}")
        moments = component_statistics(raw_targets, prediction, selected_mask, counts)[order]
        point_pair = scores(moments, np.ones((1, COMPONENTS), dtype=np.int64))[0]
        sampled_targets = scores(moments, multiplicities)
        sampled_global = sampled_targets.mean(axis=1)
        record = result_arms[arm_key]
        seed, arm, epoch_label = arm_key.split("__")
        epoch = int(epoch_label.removeprefix("epoch"))
        expected_seed = int(seed.removeprefix("seed"))
        require(
            record.get("seed") == expected_seed
            and record.get("arm") == arm
            and record.get("epoch") == epoch,
            f"E488 score identity differs: {arm_key}",
        )
        require(
            record.get("selected_rows") == int(counts.sum()),
            f"E488 selected rows differ: {arm_key}",
        )
        require(
            record.get("component_row_counts") == expected_counts,
            f"E488 component counts differ: {arm_key}",
        )
        require(record.get("updates") == epoch * 474, f"E488 update count differs: {arm_key}")
        expected_parameters = 61_122 if arm in ("static_mix", "recurrent_mix") else 61_058
        require(
            record.get("parameters") == expected_parameters,
            f"E488 parameter count differs: {arm_key}",
        )
        max_error = max(
            max_error,
            assert_near(
                point_pair, [record.get("t0"), record.get("t1")], f"{arm_key} target scores"
            ),
        )
        max_error = max(
            max_error,
            assert_near(float(point_pair.mean()), record.get("wp"), f"{arm_key} global score"),
        )
        require(
            record.get("metric_id") == "global_wp_clip2_v1", f"E488 arm metric differs: {arm_key}"
        )
        global_points[arm_key] = float(point_pair.mean())
        target_points[arm_key] = point_pair
        target_draws[arm_key] = sampled_targets
        draw_hashes[arm_key] = {
            "wp": hashlib.sha256(sampled_global.astype("<f8", copy=False).tobytes()).hexdigest(),
            "t0": hashlib.sha256(
                sampled_targets[:, 0].astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "t1": hashlib.sha256(
                sampled_targets[:, 1].astype("<f8", copy=False).tobytes()
            ).hexdigest(),
        }

    expected_contrasts = expected_contrast_records()
    global_records = result.get("registered_global_contrasts", {})
    target_records = result.get("registered_target_contrasts", {})
    require(
        set(global_records) == set(expected_contrasts), "E488 global contrast inventory differs"
    )
    require(
        set(target_records) == set(expected_contrasts), "E488 target contrast inventory differs"
    )
    contrast_draw_hashes: dict[str, str] = {}
    for key, definition in expected_contrasts.items():
        global_record = global_records[key]
        target_record = target_records[key]
        _same_record(global_record, definition, f"global {key}")
        for field in ("seed", "epoch", "contrast"):
            require(
                target_record.get(field) == definition[field],
                f"target {key} {field} identity differs",
            )
        require(
            set(target_record.get("targets", {})) == {"t0", "t1"},
            f"target inventory differs: {key}",
        )
        terms = definition["terms"]
        point_targets = sum(sign * target_points[arm_key] for arm_key, sign in terms)
        draw_targets = sum(sign * target_draws[arm_key] for arm_key, sign in terms)
        point_global = float(point_targets.mean())
        draw_global = draw_targets.mean(axis=1)
        max_error = max(
            max_error,
            assert_near(point_global, global_record.get("point_diff"), f"{key} global point"),
        )
        global_summary = summary(point_global, draw_global, definition["seed"], None)
        for field in (
            "bootstrap_sd",
            "ci95",
            "p_le_0",
            "components",
            "bootstrap_count",
            "bootstrap_seed",
            "metric_id",
        ):
            if field in ("components", "bootstrap_count", "bootstrap_seed", "metric_id"):
                require(
                    global_record.get(field) == global_summary[field],
                    f"{key} global {field} differs",
                )
            else:
                max_error = max(
                    max_error,
                    assert_near(
                        global_record.get(field), global_summary[field], f"{key} global {field}"
                    ),
                )
        max_error = max(
            max_error,
            assert_near(
                global_record.get("point_diff"),
                global_summary["point_diff"],
                f"{key} point formula",
            ),
        )
        for target_index, target_name in enumerate(("t0", "t1")):
            target_record_item = target_record["targets"][target_name]
            target_summary = summary(
                float(point_targets[target_index]),
                draw_targets[:, target_index],
                definition["seed"],
                target_name,
            )
            for field in (
                "seed",
                "target",
                "components",
                "bootstrap_count",
                "bootstrap_seed",
                "metric_id",
            ):
                require(
                    target_record_item.get(field) == target_summary[field],
                    f"{key}/{target_name} {field} differs",
                )
            for field in ("point_diff", "bootstrap_sd", "ci95", "p_le_0"):
                max_error = max(
                    max_error,
                    assert_near(
                        target_record_item.get(field),
                        target_summary[field],
                        f"{key}/{target_name} {field}",
                    ),
                )
        packed = np.column_stack((draw_global, draw_targets)).astype("<f8", copy=False)
        contrast_draw_hashes[key] = hashlib.sha256(packed.tobytes()).hexdigest()

    return {
        "study": "E488",
        "status": "all_raw_export_scores_and_draw_summaries_match",
        "arm_snapshots": len(arm_keys),
        "global_contrasts": len(global_records),
        "target_contrasts": len(target_records) * 2,
        "bootstrap_draws_per_summary": BOOTSTRAP_COUNT,
        "bootstrap_draw_vectors_recomputed": {
            "arm_global_scores": len(arm_keys),
            "arm_target_scores": len(arm_keys) * 2,
            "global_contrast_arrays": len(global_records),
            "target_contrast_scores": len(target_records) * 2,
        },
        "max_abs_error": max_error,
        "tolerance": TOLERANCE,
        "paired_draw_seed": BOOTSTRAP_SEED,
        "no_seed_pooling": True,
        "duration_interaction": "(recurrent_mix_epoch3 - static_mix_epoch3) - (recurrent_mix_epoch1 - static_mix_epoch1), within each seed",
        "arm_score_draw_sha256": draw_hashes,
        "contrast_global_t0_t1_draw_sha256": contrast_draw_hashes,
        "method": "independent shifted float64 sufficient-statistic sums over float32-clipped raw target/prediction exports; all bootstrap draws regenerated from the registered sorted-component seed",
    }


def verify_files(scores_path: Path, export_path: Path) -> dict[str, Any]:
    """Verify score/export file hashes and perform independent NumPy arithmetic."""
    result = json.loads(scores_path.read_text())
    export_digest = sha256_file(export_path)
    require(result.get("selected_export_sha256") == export_digest, "E488 raw export digest differs")
    require(
        result.get("selected_export_bytes") == export_path.stat().st_size,
        "E488 raw export size differs",
    )
    with np.load(export_path, allow_pickle=False) as archive:
        return verify_payload(result, archive)


def main() -> None:
    """Write a new, non-overwriting independent arithmetic receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E488 arithmetic receipt: {args.output}")
    result = verify_files(args.scores, args.export)
    result["score_sha256"] = sha256_file(args.scores)
    result["export_sha256"] = sha256_file(args.export)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
