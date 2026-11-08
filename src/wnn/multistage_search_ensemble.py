from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_ensemble_inputs import load_qualified_candidates
from wnn.multistage_search_ensemble_math import (
    accumulate_quadratic,
    solve_simplex,
)
from wnn.multistage_search_ensemble_optuna import search_wp_weights
from wnn.splits import MULTISTAGE_AUXILIARY_SPLIT_SCHEMA, load_campaign_split


def run_stage4_ensemble(
    run_dir: Path,
    data_root: Path,
    auxiliary_manifest: Path,
    candidate_refs: Sequence[str],
) -> dict[str, object]:
    """Fit one simplex-constrained weighted-MSE ensemble on qualified inner predictions."""
    manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(manifest["admission"]["path"])))
    data = admission.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("frozen admission has no data identity")
    frozen_data_root = Path(str(data["data_root"])).resolve(strict=True)
    if data_root.resolve(strict=True) != frozen_data_root:
        raise ValueError("stage 4 data root differs from the frozen admission")
    aux_ref = manifest.get("auxiliary_manifest")
    if not isinstance(aux_ref, Mapping):
        raise ValueError("run manifest has no auxiliary split identity")
    frozen_aux_path = Path(str(aux_ref["path"])).resolve(strict=True)
    if auxiliary_manifest.resolve(strict=True) != frozen_aux_path:
        raise ValueError("stage 4 auxiliary manifest differs from the frozen admission")
    contract = read_json(Path(str(manifest["contract"]["path"])))
    auxiliary_sha = file_sha256(auxiliary_manifest)
    split = load_campaign_split(auxiliary_manifest, data_root)
    if (
        split.schema != MULTISTAGE_AUXILIARY_SPLIT_SCHEMA
        or split.evaluation_source != "train"
        or split.sha256 != auxiliary_sha
    ):
        raise ValueError("stage 4 requires the frozen train-only auxiliary control split")
    expected_fit = [["train", group] for group in split.train_groups]
    expected_control = [["train", group] for group in split.evaluation_groups]
    candidates = load_qualified_candidates(
        run_dir,
        candidate_refs,
        auxiliary_sha,
        split.sha256,
        split.primary_split_sha256,
        expected_fit,
        expected_control,
    )
    if len(candidates) < 2:
        raise ValueError("a learned ensemble requires at least two qualified candidates")
    if len(candidates) > 32:
        raise ValueError("stage 4 candidate pool exceeds the frozen CPU bound of 32")
    predictions = [item["predictions"] for item in candidates]
    if not all(isinstance(item, Mapping) for item in predictions):
        raise TypeError("qualified prediction manifests are malformed")
    quadratic, linear, constant, total_weight = accumulate_quadratic(data_root, split, predictions)
    warmstart_weights = solve_simplex(quadratic, linear)
    warmstart_mse = float(
        (
            warmstart_weights @ quadratic @ warmstart_weights
            - 2 * warmstart_weights @ linear
            + constant
        )
        / total_weight
    )
    weights, scored, optuna_trial_count = search_wp_weights(
        run_dir,
        data_root,
        split,
        predictions,
        [str(item["name"]) for item in candidates],
        warmstart_weights,
    )
    baseline = contract.get("baseline", {})
    result: dict[str, object] = {
        "schema": "wnn.multistage-stage4-ensemble.v1",
        "status": "complete",
        "scope": "inner_train_proxy",
        "auxiliary_split_sha256": auxiliary_sha,
        "primary_split_sha256": split.primary_split_sha256,
        "evaluation_source": "train",
        "evaluation_members": expected_control,
        "evaluation_mask": "all_after_warmup AND row_index>=99",
        "selected_rows": len(split.evaluation_groups) * (SEQUENCE_LENGTH - WARMUP),
        "optimization": {
            "warm_start": {
                "method": "simplex_constrained_weighted_least_squares",
                "objective": "weighted_mse",
                "row_weight": "abs(clipped_target)",
                "weighted_mse": warmstart_mse,
                "weights": warmstart_weights.tolist(),
            },
            "optuna_search": {
                "sampler": "TPE",
                "objective": "pooled_clipped_wp_after_weighted_blend",
                "trial_count": optuna_trial_count,
                "weights_nonnegative": True,
                "weights_sum": float(weights.sum()),
                "best_wp": scored["wp"],
            },
        },
        "ensemble": {
            "members": [
                {
                    "name": item["name"],
                    "weight": float(weight),
                    "trial_result_path": item["trial_result_path"],
                    "trial_result_sha256": item["trial_result_sha256"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "prediction_manifest_sha256": item["prediction_manifest_sha256"],
                    "producer_identity_sha256": item["producer_identity_sha256"],
                    "objective": item["objective"],
                }
                for item, weight in zip(candidates, weights, strict=True)
            ],
            "control_wp": scored,
        },
        "fixed_primary_tl3_reference": {
            "wp": baseline.get("canonical_fresh_TL3_wp"),
            "scope": "full_official_valid_primary_reference",
            "used_for_weight_search": False,
            "directly_comparable_to_inner_proxy": False,
        },
        "interpretation": "Inner-train proxy selection only; freeze weights before any primary-valid evaluation.",
    }
    destination = run_dir / "stage4" / "ensemble_result.json"
    atomic_write_json(destination, result, replace=False)
    from wnn.multistage_plot_ensemble import export_stage4_ensemble_plots

    plots = export_stage4_ensemble_plots(destination)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "stage4_ensemble_complete",
            "at_utc": utc_now(),
            "stage": 4,
            "result_path": str(destination.resolve()),
            "result_sha256": file_sha256(destination),
            "objective_value": scored["wp"],
            "status": "COMPLETE",
        },
    )
    return result | {
        "result_path": str(destination.resolve()),
        "result_sha256": file_sha256(destination),
        "plots": plots,
    }
