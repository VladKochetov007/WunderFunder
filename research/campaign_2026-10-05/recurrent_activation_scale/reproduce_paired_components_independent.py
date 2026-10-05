"""Reproduce E478 six official scores and paired component CIs from its NPZ export.

This verifier consumes no checkpoints or hidden/unselected labels. It uses the
frozen official Global WP metric and component-moment pooling implementation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
evaluation = importlib.import_module("wnn.evaluation")
gmetric = importlib.import_module("wnn.gmetric")
_score_component_moments = evaluation._score_component_moments
paired_component_bootstrap = evaluation.paired_component_bootstrap
METRIC_CLIP = gmetric.METRIC_CLIP
METRIC_ID = gmetric.METRIC_ID
_centered_moments = gmetric._centered_moments
global_wp = gmetric.global_wp

RUN = ROOT / "research/campaign_2026-10-05/recurrent_activation_scale/eda/six_v1"
NPZ_PATH = RUN / "RESULT_paired_components.npz"
RESULT_PATH = RUN / "RESULT.json"
EXPECTED_NPZ_SHA256 = "235790580187c69532338a7134d23de4e3c3d6b7d4f0f23152857805b59f3be9"
SEEDS = (271828, 271829, 271830)
ALPHAS = (0.0, 0.1)
N_COMPONENTS = 16
N_ROWS = 20_000
N_BOOT = 2_000


def sha256(path: Path) -> str:
    """Hash the small score export using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    """Fail closed when the exported scoring panel is malformed."""
    if not ok:
        raise ValueError(message)


def direct_bootstrap(
    a: np.ndarray, b: np.ndarray, component_ids: np.ndarray, seed: int
) -> dict[str, object]:
    """Recompute paired pooled-WP differences from official component moments."""
    # The official paired bootstrap canonicalizes groups with np.unique, which
    # sorts component IDs before drawing indices. Preserve that seeded order.
    order = np.argsort(component_ids)
    a, b = a[order], b[order]
    counts = np.ones(N_COMPONENTS, dtype=np.int64)
    point_a = _score_component_moments(a, counts)
    point_b = _score_component_moments(b, counts)
    rng = np.random.default_rng(seed)
    deltas = np.empty(N_BOOT, dtype=np.float64)
    for index in range(N_BOOT):
        draw = rng.integers(0, N_COMPONENTS, N_COMPONENTS)
        multiplicity = np.bincount(draw, minlength=N_COMPONENTS)
        deltas[index] = _score_component_moments(a, multiplicity) - _score_component_moments(
            b, multiplicity
        )
    return {
        "candidate_wp": point_a,
        "reference_wp": point_b,
        "mean_diff": point_a - point_b,
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(deltas <= 0)),
        "n_components": N_COMPONENTS,
        "n_boot": N_BOOT,
        "seed": seed,
    }


def run() -> dict[str, object]:
    """Validate export provenance and reproduce all registered score cells."""
    result = json.loads(RESULT_PATH.read_text())
    actual_hash = sha256(NPZ_PATH)
    require(actual_hash == EXPECTED_NPZ_SHA256, "paired-component NPZ hash mismatch")
    require(
        result.get("status") == "scored_exposed_components_descriptive_only",
        "result status differs",
    )
    require(result.get("metric_id") == METRIC_ID == "global_wp_clip2_v1", "metric identity differs")
    require(
        result.get("export", {}).get("sha256") == actual_hash, "result does not bind this export"
    )
    require(
        result.get("bootstrap")
        == {
            "resampling_unit": "component",
            "n_boot": N_BOOT,
            "seed_per_fit_seed": "fit seed",
            "paired": True,
        },
        "bootstrap registration differs",
    )

    with np.load(NPZ_PATH, allow_pickle=False) as z:
        expected_keys = {
            "component_ids",
            "row_groups",
            "selected_rows_per_component",
            "selected_targets_raw",
            "score_masks_by_component",
            "sequence_index",
            "row_index_within_sequence",
            "alpha_values",
            "seed_values",
            "predictions",
            "registered_arm_keys",
        }
        require(set(z.files) == expected_keys, "export array inventory differs")
        arrays = {key: z[key] for key in expected_keys}

    component_ids = arrays["component_ids"]
    row_groups = arrays["row_groups"]
    counts = arrays["selected_rows_per_component"].astype(np.int64)
    y = arrays["selected_targets_raw"]
    masks = arrays["score_masks_by_component"]
    sequence_index = arrays["sequence_index"]
    within_row = arrays["row_index_within_sequence"]
    predictions = arrays["predictions"]
    expected_keys_order = [f"seed{seed}_alpha{alpha:.1f}" for seed in SEEDS for alpha in ALPHAS]

    require(component_ids.shape == (N_COMPONENTS,), "component ID shape differs")
    require(row_groups.shape == (N_COMPONENTS,), "row-group shape differs")
    require(len(set(component_ids.tolist())) == N_COMPONENTS, "component IDs repeat")
    require(len(set(row_groups.tolist())) == N_COMPONENTS, "row groups repeat")
    require(
        masks.shape == (N_COMPONENTS, N_ROWS) and masks.dtype == np.bool_,
        "mask shape/dtype differs",
    )
    require(np.array_equal(counts, masks.sum(axis=1)), "per-component selected-row counts differ")
    expected_sequence_index = np.repeat(np.arange(N_COMPONENTS, dtype=np.int16), counts)
    expected_within_row = np.concatenate([np.flatnonzero(mask) for mask in masks]).astype(np.int32)
    require(sequence_index.shape == expected_sequence_index.shape, "sequence-index size differs")
    require(
        np.array_equal(sequence_index, expected_sequence_index),
        "selected rows are not component ordered",
    )
    require(
        np.array_equal(within_row, expected_within_row),
        "selected row map differs from full score masks",
    )
    n_selected = int(counts.sum())
    require(y.shape == (n_selected, 2) and y.dtype == np.float32, "raw target shape/dtype differs")
    require(
        predictions.shape == (6, n_selected, 2) and predictions.dtype == np.float32,
        "prediction shape/dtype differs",
    )
    require(np.isfinite(y).all(), "nonfinite selected raw target before clipping")
    require(np.isfinite(predictions).all(), "nonfinite raw prediction before clipping")
    require(
        np.array_equal(arrays["seed_values"], np.asarray(SEEDS, dtype=np.int64)),
        "seed registration differs",
    )
    require(
        np.array_equal(arrays["alpha_values"], np.asarray(ALPHAS, dtype=np.float32)),
        "alpha registration differs",
    )
    require(arrays["registered_arm_keys"].tolist() == expected_keys_order, "arm order differs")

    metadata = result["data"]
    require(component_ids.tolist() == metadata["component_ids"], "component IDs differ from result")
    require(row_groups.tolist() == metadata["row_groups"], "row groups differ from result")
    require(
        counts.tolist() == metadata["selected_rows_per_component"],
        "selected counts differ from result",
    )
    require(n_selected == metadata["total_selected_rows"], "total selected rows differ from result")
    require(
        result["export"]["arrays"]["predictions"]["shape"] == list(predictions.shape),
        "declared prediction shape differs",
    )

    arm_scores: dict[str, dict[str, float | int]] = {}
    all_selected = np.ones(n_selected, dtype=bool)
    moments_by_arm: dict[str, np.ndarray] = {}
    for arm_index, key in enumerate(expected_keys_order):
        pred = predictions[arm_index]
        official = global_wp(y[None], pred[None], all_selected[None])
        arm_moments = np.zeros((N_COMPONENTS, 2, 6), dtype=np.float64)
        offset = 0
        for component, count in enumerate(counts):
            n = int(count)
            raw_y = y[offset : offset + n]
            raw_p = pred[offset : offset + n]
            # Finiteness was checked on raw values above, before official clipping.
            yc = np.clip(raw_y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
            pc = np.clip(raw_p, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
            for target in range(2):
                arm_moments[component, target] = _centered_moments(yc[:, target], pc[:, target])
            offset += n
        require(offset == n_selected, "component boundaries do not cover selected rows")
        pooled = _score_component_moments(arm_moments, np.ones(N_COMPONENTS, dtype=np.int64))
        require(
            abs(float(official["wp"]) - pooled) <= 2e-7, "global WP and component moments disagree"
        )
        stored = result["arms"].get(key)
        require(stored is not None, f"registered result arm missing: {key}")
        require(
            abs(float(stored["global_wp"]) - float(official["wp"])) <= 2e-12,
            f"official global score differs: {key}",
        )
        require(
            abs(float(stored["target_wp"]["t0"]) - float(official["t0"])) <= 2e-12
            and abs(float(stored["target_wp"]["t1"]) - float(official["t1"])) <= 2e-12,
            f"official target scores differ: {key}",
        )
        arm_scores[key] = {
            "global_wp": float(official["wp"]),
            "t0_wp": float(official["t0"]),
            "t1_wp": float(official["t1"]),
            "selected_rows": n_selected,
        }
        moments_by_arm[key] = arm_moments

    contrasts: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        candidate_key = f"seed{seed}_alpha0.1"
        reference_key = f"seed{seed}_alpha0.0"
        actual = paired_component_bootstrap(
            moments_by_arm[candidate_key],
            moments_by_arm[reference_key],
            component_ids,
            n_boot=N_BOOT,
            seed=seed,
            centered=True,
        )
        direct = direct_bootstrap(
            moments_by_arm[candidate_key], moments_by_arm[reference_key], component_ids, seed
        )
        for field in ("candidate_wp", "reference_wp", "mean_diff", "bootstrap_sd", "p_le_0"):
            require(
                abs(float(actual[field]) - float(direct[field])) <= 2e-12,
                f"official/direct bootstrap {field} differs for seed {seed}",
            )
        require(
            np.allclose(actual["ci95"], direct["ci95"], atol=2e-12, rtol=0),
            f"direct CI differs for seed {seed}",
        )
        stored = result["contrasts"].get(f"seed{seed}:alpha_p10_minus_zero")
        require(stored is not None, f"registered contrast missing for seed {seed}")
        for field in ("candidate_wp", "reference_wp", "mean_diff", "bootstrap_sd", "p_le_0"):
            require(
                abs(float(stored[field]) - float(actual[field])) <= 2e-12,
                f"stored contrast {field} differs for seed {seed}",
            )
        require(
            np.allclose(stored["ci95"], actual["ci95"], atol=2e-12, rtol=0),
            f"stored CI differs for seed {seed}",
        )
        contrasts[str(seed)] = actual

    return {
        "status": "independent_npz_reproduction_passed",
        "metric_id": METRIC_ID,
        "npz_sha256": actual_hash,
        "selected_rows": n_selected,
        "components": N_COMPONENTS,
        "arms": arm_scores,
        "paired_contrasts": contrasts,
        "checks": {
            "all_raw_export_values_finite_before_clipping": True,
            "selected_rows_match_16_full_score_masks": True,
            "registered_arm_order_exact": True,
            "official_global_wp_matches_official_component_moments": True,
            "all_six_scores_match_result_json": True,
            "official_bootstrap_matches_independent_2000_draw_recomputation": True,
            "all_three_contrasts_match_result_json": True,
            "checkpoints_or_fit_source_used": False,
            "GPU_used": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
