import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # E-513: weak-model portfolio as a residual ensemble

    This notebook follows the frozen protocol in PRE_REGISTRATION.md. It asks whether the fixed equal-weight average of three weak models is a more useful partner for the strong anchor than the preselected single-model control. It uses only previously inspected chain-fold validation predictions, so every result is exploratory.

    The order is deliberate: verify source hashes, row-group alignment, fold ancestry, and anchor / ab53 run-card scores first. Cohort comparisons remain gated until those checks pass. The official metric is pooled clipped weighted Pearson, with the fixed official mask and clipping.
    """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path
    import hashlib
    import json
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = Path.cwd()
    PORTFOLIO = ROOT / "research/campaign_2026-10-06/bandit_continuation/weak_cohort_portfolio"
    MANIFEST_PATH = PORTFOLIO / "SOURCE_MANIFEST_E513.json"
    ADDENDUM_PATH = PORTFOLIO / "PRE_SCORE_ADDENDUM_20261007.json"
    MANIFEST = json.loads(MANIFEST_PATH.read_text())
    ADDENDUM = json.loads(ADDENDUM_PATH.read_text())


    def sha256_file(path: Path) -> str:
        """Hash a frozen source file in bounded blocks."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        return digest.hexdigest()


    MANIFEST_SHA256 = sha256_file(MANIFEST_PATH)
    EXPECTED_MANIFEST_SHA256 = ADDENDUM["source_manifest_sha256"]
    MANIFEST_HASH_OK = MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    SOURCE_HASH_CHECKS = {
        relative: (ROOT / relative).is_file()
        and sha256_file(ROOT / relative) == metadata["sha256"]
        for relative, metadata in MANIFEST["sources"].items()
    }
    SOURCE_HASHES_OK = all(SOURCE_HASH_CHECKS.values())

    target_cache = np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False)
    Y = target_cache["y"]
    IS_SCORED = target_cache["mask"]
    ROW_GROUPS = target_cache["row_group"]
    assert Y.shape == (1873, 20000, 2)
    assert IS_SCORED.shape == (1873, 20000)
    ROW_GROUPS_SHA256 = hashlib.sha256(ROW_GROUPS.astype("<i8").tobytes()).hexdigest()
    ROW_GROUP_ORDER_OK = (
        ROW_GROUPS_SHA256
        == MANIFEST["target_cache_row_groups_sha256_little_endian_int64"]
    )

    chain_cache = np.load(ROOT / "cache/chain_folds.npz", allow_pickle=False)
    split_labels = chain_cache["split"]
    valid_chain_rows = np.flatnonzero(split_labels == "valid")
    valid_row_groups = chain_cache["row_group"][valid_chain_rows]
    valid_positions = {int(group): index for index, group in enumerate(valid_row_groups)}
    ROW_GROUP_SET_OK = set(map(int, ROW_GROUPS)) == set(valid_positions)
    CHAIN_MAP_ORDER = np.array([valid_positions[int(group)] for group in ROW_GROUPS])
    CHAIN_BY_SEQUENCE = chain_cache["chain"][valid_chain_rows][CHAIN_MAP_ORDER]
    FOLD_BY_SEQUENCE = chain_cache["fold"][valid_chain_rows][CHAIN_MAP_ORDER]
    TRAIN_CHAIN_IDS = np.unique(chain_cache["chain"][split_labels == "train"])
    OFFICIAL_MASK = IS_SCORED & (np.arange(Y.shape[1])[None, :] >= 99)
    PURE_INDICES = {
        fold_number: np.flatnonzero(
            (FOLD_BY_SEQUENCE == fold_number)
            & ~np.isin(CHAIN_BY_SEQUENCE, TRAIN_CHAIN_IDS)
        )
        for fold_number in (0, 1)
    }
    FULL_FOLD_INDICES = {
        fold_number: np.flatnonzero(FOLD_BY_SEQUENCE == fold_number)
        for fold_number in (0, 1)
    }
    PURE_CHAIN_COUNTS = {
        str(fold_number): {
            "all_valid_groups": int(len(FULL_FOLD_INDICES[fold_number])),
            "pure_chain_groups": int(len(PURE_INDICES[fold_number])),
            "excluded_train_bridge_groups": int(
                len(FULL_FOLD_INDICES[fold_number]) - len(PURE_INDICES[fold_number])
            ),
        }
        for fold_number in (0, 1)
    }

    PREDICTION_FILES = {
        "anchor_f0": "cache/pred_ab62_base_s3_f0_valid.npy",
        "anchor_f1": "cache/pred_ab62_base_s3_f1_valid.npy",
        "basehead_f0": "cache/pred_ab48_basehead_f0_valid.npy",
        "basehead_f1": "cache/pred_ab48_basehead_f1_valid.npy",
        "td5_f0": "cache/pred_ab62_td5_f0_valid.npy",
        "td5_f1": "cache/pred_ab62_td5_f1_valid.npy",
        "p2_f0": "cache/pred_ab53_p2_f0_valid.npy",
        "p2_f1": "cache/pred_ab53_p2_f1_valid.npy",
    }
    PREDICTIONS = {
        name: np.load(ROOT / relative, mmap_mode="r")
        for name, relative in PREDICTION_FILES.items()
    }
    PREDICTION_LAYOUT_OK = all(
        array.shape == Y.shape and array.dtype == np.float32
        for array in PREDICTIONS.values()
    )
    PREDICTIONS_FINITE = all(
        np.isfinite(array[FULL_FOLD_INDICES[fold_number]]).all()
        for array in PREDICTIONS.values()
        for fold_number in (0, 1)
    )

    RUN_CARD_FOLDS_OK = True
    for run_name in (
        "ab48_basehead_f0",
        "ab48_basehead_f1",
        "ab53_p2_f0",
        "ab53_p2_f1",
        "ab56_ema_e4_f0",
        "ab58_ema_e4_f1",
    ):
        card_text = (ROOT / "research/runs" / f"{run_name}.md").read_text()
        match = re.search(r"^- valid_fold: (\d+)$", card_text, re.MULTILINE)
        expected_fold = int(run_name[-1])
        RUN_CARD_FOLDS_OK &= bool(match and int(match.group(1)) == expected_fold)

    TD5_CONFIGS = {
        fold_number: json.loads(
            (ROOT / f"artifacts/logs/ab62_td5_f{fold_number}.log").read_text().splitlines()[0]
        )
        for fold_number in (0, 1)
    }
    TD5_FOLDS_OK = all(
        config["name"] == f"ab62_td5_f{fold_number}"
        and config["valid_fold"] == fold_number
        and config["train_source"] == "train+valid"
        and not config["init"]
        and not config["distill"]
        for fold_number, config in TD5_CONFIGS.items()
    )

    AUDIT_SUMMARY = {
        "manifest_hash_ok": MANIFEST_HASH_OK,
        "all_40_source_hashes_match": SOURCE_HASHES_OK,
        "matching_source_hash_count": sum(SOURCE_HASH_CHECKS.values()),
        "source_hash_count": len(SOURCE_HASH_CHECKS),
        "target_row_group_hash_matches": ROW_GROUP_ORDER_OK,
        "target_groups_align_to_chain_map": ROW_GROUP_SET_OK,
        "prediction_shapes_and_dtypes_match": PREDICTION_LAYOUT_OK,
        "selected_fold_predictions_are_finite": PREDICTIONS_FINITE,
        "run_card_fold_membership_matches": RUN_CARD_FOLDS_OK,
        "td5_logs_confirm_fold_and_independent_training": TD5_FOLDS_OK,
        "fold_membership": PURE_CHAIN_COUNTS,
        "train_chain_count": int(len(TRAIN_CHAIN_IDS)),
        "metric": "global_wp_clip2_v1",
        "official_warmup": 99,
    }
    PROVENANCE_GATE_OK = all(
        (
            MANIFEST_HASH_OK,
            SOURCE_HASHES_OK,
            ROW_GROUP_ORDER_OK,
            ROW_GROUP_SET_OK,
            PREDICTION_LAYOUT_OK,
            PREDICTIONS_FINITE,
            RUN_CARD_FOLDS_OK,
            TD5_FOLDS_OK,
        )
    )
    mo.md("### Source and alignment audit\n\n" + json.dumps(AUDIT_SUMMARY, indent=2))
    return (
        ADDENDUM,
        AUDIT_SUMMARY,
        CHAIN_BY_SEQUENCE,
        FULL_FOLD_INDICES,
        OFFICIAL_MASK,
        PORTFOLIO,
        PREDICTIONS,
        PROVENANCE_GATE_OK,
        PURE_INDICES,
        ROOT,
        Y,
        json,
        np,
        plt,
        sha256_file,
    )


@app.cell(hide_code=True)
def _(
    ADDENDUM,
    FULL_FOLD_INDICES,
    OFFICIAL_MASK,
    PREDICTIONS,
    PROVENANCE_GATE_OK,
    ROOT,
    Y,
    json,
    mo,
    np,
):
    __import__("sys").path.insert(0, str(ROOT / "src"))
    from wnn.gmetric import global_wp

    RUN_CARD_EXPECTED_SCORES = {
        "ab56_ema_e4_f0": 0.6761,
        "ab58_ema_e4_f1": 0.6623,
        "ab53_p2_f0": 0.6739,
        "ab53_p2_f1": 0.6595,
    }
    RUN_CARD_PREDICTIONS = {
        "ab56_ema_e4_f0": "anchor_f0",
        "ab58_ema_e4_f1": "anchor_f1",
        "ab53_p2_f0": "p2_f0",
        "ab53_p2_f1": "p2_f1",
    }
    RUN_CARD_REPRODUCTIONS = {}
    for card_key, expected_score in RUN_CARD_EXPECTED_SCORES.items():
        fold_index = int(card_key[-1])
        evaluation_rows = FULL_FOLD_INDICES[fold_index]
        prediction_key = RUN_CARD_PREDICTIONS[card_key]
        observed_score = global_wp(
            Y[evaluation_rows],
            np.asarray(PREDICTIONS[prediction_key][evaluation_rows], dtype=np.float32),
            OFFICIAL_MASK[evaluation_rows],
        )
        score_difference = float(observed_score["wp"]) - expected_score
        RUN_CARD_REPRODUCTIONS[card_key] = {
            "expected_rounded_wp": expected_score,
            "observed_wp": float(observed_score["wp"]),
            "absolute_difference": abs(score_difference),
            "t0": float(observed_score["t0"]),
            "t1": float(observed_score["t1"]),
            "selected_rows": int(observed_score["selected_rows"]),
            "within_rounding_tolerance": abs(score_difference)
            <= ADDENDUM["run_card_reproduction_gate"]["maximum_absolute_difference"],
        }

    RUN_CARD_REPRODUCTION_OK = all(
        reproduction["within_rounding_tolerance"]
        for reproduction in RUN_CARD_REPRODUCTIONS.values()
    )
    ALIGNMENT_GATE_OK = PROVENANCE_GATE_OK and RUN_CARD_REPRODUCTION_OK
    PRE_SCORE_GATE = {
        "provenance_gate_ok": PROVENANCE_GATE_OK,
        "all_four_run_cards_reproduce": RUN_CARD_REPRODUCTION_OK,
        "maximum_absolute_difference": ADDENDUM["run_card_reproduction_gate"][
            "maximum_absolute_difference"
        ],
        "reproductions": RUN_CARD_REPRODUCTIONS,
        "cohort_comparison_allowed": ALIGNMENT_GATE_OK,
    }
    mo.md("### Pre-score alignment gate\n\n" + json.dumps(PRE_SCORE_GATE, indent=2))
    return (
        ALIGNMENT_GATE_OK,
        PRE_SCORE_GATE,
        RUN_CARD_REPRODUCTION_OK,
        global_wp,
    )


@app.cell(hide_code=True)
def _(
    ADDENDUM,
    ALIGNMENT_GATE_OK,
    CHAIN_BY_SEQUENCE,
    OFFICIAL_MASK,
    PREDICTIONS,
    PRE_SCORE_GATE,
    PURE_INDICES,
    Y,
    global_wp,
    json,
    mo,
    np,
):
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    LAMBDA_GRID = np.round(np.arange(0.0, 0.4000001, 0.05), 2).tolist()


    def blend_raw_predictions(anchor: np.ndarray, partner: np.ndarray, strength: float) -> np.ndarray:
        """Blend un-clipped model outputs and return float32 predictions."""
        mixed = (1.0 - strength) * anchor.astype(np.float64)
        mixed += strength * partner.astype(np.float64)
        return mixed.astype(np.float32)


    def load_fold_payload(fold_number: int, row_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Load one fold slice of the frozen members for bounded analysis."""
        anchor = np.asarray(PREDICTIONS[f"anchor_f{fold_number}"][row_indices], dtype=np.float32)
        basehead = np.asarray(PREDICTIONS[f"basehead_f{fold_number}"][row_indices], dtype=np.float32)
        td5 = np.asarray(PREDICTIONS[f"td5_f{fold_number}"][row_indices], dtype=np.float32)
        p2 = np.asarray(PREDICTIONS[f"p2_f{fold_number}"][row_indices], dtype=np.float32)
        cohort = (
            anchor.astype(np.float64) * 0.0
            + basehead.astype(np.float64)
            + td5.astype(np.float64)
            + p2.astype(np.float64)
        ) / 3.0
        return {
            "targets": Y[row_indices],
            "mask": OFFICIAL_MASK[row_indices],
            "chains": CHAIN_BY_SEQUENCE[row_indices],
            "anchor": anchor,
            "basehead": basehead,
            "td5": td5,
            "p2": p2,
            "cohort": cohort.astype(np.float32),
        }


    def score_fold_grid(fold_number: int) -> dict[str, object]:
        """Score the frozen blend grid on one pure-chain fold."""
        payload = load_fold_payload(fold_number, PURE_INDICES[fold_number])
        targets = payload["targets"]
        selected = payload["mask"]
        anchor_score = global_wp(targets, payload["anchor"], selected)
        standalone_scores = {
            member_name: global_wp(targets, payload[member_name], selected)
            for member_name in ("basehead", "td5", "p2", "cohort")
        }
        grid_rows = []
        for blend_lambda in LAMBDA_GRID:
            cohort_prediction = blend_raw_predictions(
                payload["anchor"], payload["cohort"], blend_lambda
            )
            single_prediction = blend_raw_predictions(
                payload["anchor"], payload["basehead"], blend_lambda
            )
            grid_rows.append(
                {
                    "lambda": blend_lambda,
                    "cohort_endpoint": global_wp(targets, cohort_prediction, selected),
                    "single_endpoint": global_wp(targets, single_prediction, selected),
                }
            )
        return {
            "pure_chain_groups": int(len(PURE_INDICES[fold_number])),
            "anchor": anchor_score,
            "standalone": standalone_scores,
            "grid": grid_rows,
        }


    def choose_lambda(grid_rows: list[dict[str, object]], endpoint_key: str) -> float:
        """Choose the smallest grid value within 1e-8 of the best calibration score."""
        best_score = max(float(row[endpoint_key]["wp"]) for row in grid_rows)
        return min(
            float(row["lambda"])
            for row in grid_rows
            if float(row[endpoint_key]["wp"]) >= best_score - 1e-8
        )


    E513_FOLD_SCORES = {
        fold_number: score_fold_grid(fold_number)
        for fold_number in (0, 1)
    }
    SELECTED_LAMBDAS = {
        calibration_fold: {
            "cohort": choose_lambda(
                E513_FOLD_SCORES[calibration_fold]["grid"], "cohort_endpoint"
            ),
            "single_basehead": choose_lambda(
                E513_FOLD_SCORES[calibration_fold]["grid"], "single_endpoint"
            ),
        }
        for calibration_fold in (0, 1)
    }


    def evaluate_cross_fold_direction(
        calibration_fold: int, evaluation_fold: int
    ) -> dict[str, object]:
        """Evaluate frozen calibration lambdas and paired chain-bootstrap deltas."""
        cohort_lambda = SELECTED_LAMBDAS[calibration_fold]["cohort"]
        single_lambda = SELECTED_LAMBDAS[calibration_fold]["single_basehead"]
        evaluation_indices = PURE_INDICES[evaluation_fold]
        payload = load_fold_payload(evaluation_fold, evaluation_indices)
        cohort_prediction = blend_raw_predictions(
            payload["anchor"], payload["cohort"], cohort_lambda
        )
        single_prediction = blend_raw_predictions(
            payload["anchor"], payload["basehead"], single_lambda
        )
        cohort_score = global_wp(payload["targets"], cohort_prediction, payload["mask"])
        anchor_score = global_wp(payload["targets"], payload["anchor"], payload["mask"])
        single_score = global_wp(payload["targets"], single_prediction, payload["mask"])

        cohort_moments, anchor_moments = paired_component_moments(
            payload["targets"], cohort_prediction, payload["anchor"], payload["mask"]
        )
        _, single_moments = paired_component_moments(
            payload["targets"], cohort_prediction, single_prediction, payload["mask"]
        )
        bootstrap_seed = int(ADDENDUM["source_manifest_sha256"][:8], 16) + evaluation_fold
        cohort_vs_anchor = paired_component_bootstrap(
            cohort_moments,
            anchor_moments,
            payload["chains"],
            n_boot=ADDENDUM["bootstrap_replicates"],
            seed=bootstrap_seed,
            centered=True,
        )
        cohort_vs_single = paired_component_bootstrap(
            cohort_moments,
            single_moments,
            payload["chains"],
            n_boot=ADDENDUM["bootstrap_replicates"],
            seed=bootstrap_seed,
            centered=True,
        )
        if abs(cohort_vs_anchor["candidate_wp"] - cohort_score["wp"]) > 5e-7:
            raise ValueError("component bootstrap moments disagree with the official point score")
        if abs(cohort_vs_anchor["reference_wp"] - anchor_score["wp"]) > 5e-7:
            raise ValueError("anchor component moments disagree with the official point score")
        if abs(cohort_vs_single["reference_wp"] - single_score["wp"]) > 5e-7:
            raise ValueError("single-control component moments disagree with the official point score")
        return {
            "calibration_fold": calibration_fold,
            "evaluation_fold": evaluation_fold,
            "cohort_lambda": cohort_lambda,
            "single_control_lambda": single_lambda,
            "evaluation_pure_chain_groups": int(len(evaluation_indices)),
            "cohort": cohort_score,
            "anchor": anchor_score,
            "single_control": single_score,
            "cohort_vs_anchor": cohort_vs_anchor,
            "cohort_vs_single_control": cohort_vs_single,
            "bootstrap_seed": bootstrap_seed,
        }


    if not ALIGNMENT_GATE_OK:
        E513_DIRECTIONS = None
        E513_STATUS = "stopped_before_cohort_scoring_alignment_gate_failed"
    else:
        E513_DIRECTIONS = [
            evaluate_cross_fold_direction(0, 1),
            evaluate_cross_fold_direction(1, 0),
        ]
        E513_STATUS = "cross_fold_scoring_and_chain_bootstrap_complete"

    E513_CORE = {
        "status": E513_STATUS,
        "alignment_gate": PRE_SCORE_GATE,
        "fold_scores": E513_FOLD_SCORES if ALIGNMENT_GATE_OK else None,
        "selected_calibration_lambdas": SELECTED_LAMBDAS if ALIGNMENT_GATE_OK else None,
        "directions": E513_DIRECTIONS,
        "exposure": "previously inspected validation folds; exploratory only",
        "public_score_forecast": None,
    }
    mo.md("### Cross-fold test\n\n" + json.dumps(E513_CORE, indent=2))
    return E513_CORE, SELECTED_LAMBDAS, load_fold_payload


@app.cell(hide_code=True)
def _(PURE_INDICES, json, load_fold_payload, mo, np):
    from itertools import combinations


    def weighted_corr_selected(
        left: np.ndarray, right: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute weighted Pearson correlation for selected finite observations."""
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return 0.0
        left_mean = float(np.sum(weights * left) / weight_sum)
        right_mean = float(np.sum(weights * right) / weight_sum)
        centered_left = left - left_mean
        centered_right = right - right_mean
        left_variance = float(np.sum(weights * centered_left**2) / weight_sum)
        right_variance = float(np.sum(weights * centered_right**2) / weight_sum)
        if left_variance <= 1e-16 or right_variance <= 1e-16:
            return 0.0
        covariance = float(
            np.sum(weights * centered_left * centered_right) / weight_sum
        )
        return float(np.clip(covariance / np.sqrt(left_variance * right_variance), -1.0, 1.0))


    def residual_correlation_diagnostics(fold_number: int) -> list[dict[str, object]]:
        """Measure weighted residual and prediction complementarity by target."""
        payload = load_fold_payload(fold_number, PURE_INDICES[fold_number])
        clipped_targets = np.clip(payload["targets"].astype(np.float64), -2.0, 2.0)
        clipped_predictions = {
            member_name: np.clip(payload[member_name].astype(np.float64), -2.0, 2.0)
            for member_name in ("anchor", "basehead", "td5", "p2")
        }
        pair_rows = []
        for target_index in (0, 1):
            target_values = clipped_targets[..., target_index]
            weights_all = np.abs(target_values)
            active = payload["mask"] & (weights_all > 0.0)
            target = target_values[active]
            weights = weights_all[active]
            model_values = {
                member_name: prediction[..., target_index][active]
                for member_name, prediction in clipped_predictions.items()
            }
            model_correlations = {
                member_name: weighted_corr_selected(target, prediction, weights)
                for member_name, prediction in model_values.items()
            }
            for left_name, right_name in combinations(
                ("basehead", "td5", "p2"), 2
            ):
                left_prediction = model_values[left_name]
                right_prediction = model_values[right_name]
                left_correlation = model_correlations[left_name]
                right_correlation = model_correlations[right_name]
                prediction_correlation = weighted_corr_selected(
                    left_prediction, right_prediction, weights
                )
                residual_correlation = weighted_corr_selected(
                    target - left_prediction, target - right_prediction, weights
                )
                denominator = np.sqrt(
                    max(1.0 - left_correlation**2, 0.0)
                    * max(1.0 - prediction_correlation**2, 0.0)
                )
                partial = (
                    (right_correlation - left_correlation * prediction_correlation)
                    / denominator
                    if denominator > 1e-12
                    else 0.0
                )
                pair_rows.append(
                    {
                        "fold": fold_number,
                        "target": target_index,
                        "left": left_name,
                        "right": right_name,
                        "left_target_correlation": left_correlation,
                        "right_target_correlation": right_correlation,
                        "prediction_correlation": prediction_correlation,
                        "residual_correlation": residual_correlation,
                        "partial_target_correlation": float(
                            np.clip(partial, -1.0, 1.0)
                        ),
                    }
                )
        return pair_rows


    RESIDUAL_CORRELATION_DIAGNOSTICS = [
        row
        for fold_key in (0, 1)
        for row in residual_correlation_diagnostics(fold_key)
    ]
    mo.md(
        "### Weak-member complementarity diagnostics\n\n"
        + json.dumps(RESIDUAL_CORRELATION_DIAGNOSTICS, indent=2)
    )
    return (RESIDUAL_CORRELATION_DIAGNOSTICS,)


@app.cell(hide_code=True)
def _(
    AUDIT_SUMMARY,
    E513_CORE,
    PORTFOLIO,
    PRE_SCORE_GATE,
    PURE_INDICES,
    RESIDUAL_CORRELATION_DIAGNOSTICS,
    ROOT,
    RUN_CARD_REPRODUCTION_OK,
    SELECTED_LAMBDAS,
    json,
    mo,
    np,
    plt,
    sha256_file,
):
    from datetime import datetime, timezone

    PLOT_DIRECTORY = PORTFOLIO / "plots"
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_PATH = PORTFOLIO / "E513_RESULT.json"
    REPORT_PATH = PORTFOLIO / "REPORT.md"
    CODE_RECEIPT_PATH = PORTFOLIO / "POST_SCORE_CODE_RECEIPT_E513.json"

    PROMOTION_GATE = all(
        direction["cohort_vs_anchor"]["mean_diff"] > 0.0
        and direction["cohort_vs_anchor"]["ci95"][0] > 0.0
        and direction["cohort_vs_single_control"]["mean_diff"] > 0.0
        and direction["cohort_vs_single_control"]["ci95"][0] > 0.0
        for direction in E513_CORE["directions"]
    )
    if PROMOTION_GATE:
        E513_TREE_INTERPRETATION = "supported_for_new_confirmation"
        E513_NEXT_STEP = (
            "Create a new preregistered confirmation with a frozen cohort and "
            "independent evaluation units before considering a served blend."
        )
    else:
        E513_TREE_INTERPRETATION = "insufficient_evidence_for_this_fixed_cohort_recipe"
        E513_NEXT_STEP = (
            "Stop allocation to this exact blend on these exposed folds. Preserve "
            "weak-model averaging as a mechanism family for a materially different "
            "cohort or as a teacher / representation contribution."
        )

    E513_DECISION = {
        "promotion_gate_passed": PROMOTION_GATE,
        "tree_interpretation": E513_TREE_INTERPRETATION,
        "next_step": E513_NEXT_STEP,
        "both_cross_fold_point_deltas_vs_anchor_positive": all(
            direction["cohort_vs_anchor"]["mean_diff"] > 0.0
            for direction in E513_CORE["directions"]
        ),
        "both_cross_fold_point_deltas_vs_single_positive": all(
            direction["cohort_vs_single_control"]["mean_diff"] > 0.0
            for direction in E513_CORE["directions"]
        ),
        "all_primary_bootstrap_intervals_exclude_zero": all(
            direction["cohort_vs_anchor"]["ci95"][0] > 0.0
            for direction in E513_CORE["directions"]
        ),
        "all_single_control_bootstrap_intervals_exclude_zero": all(
            direction["cohort_vs_single_control"]["ci95"][0] > 0.0
            for direction in E513_CORE["directions"]
        ),
        "selected_cohort_lambdas": SELECTED_LAMBDAS,
        "cohort_selection_near_grid_upper_edge": all(
            SELECTED_LAMBDAS[fold_key]["cohort"] >= 0.35 for fold_key in (0, 1)
        ),
        "exposure": "all candidate folds were previously inspected; not fresh confirmation",
        "public_score_forecast": None,
    }

    STANDALONE_ROWS = []
    for fold_key in (0, 1):
        fold_scores = E513_CORE["fold_scores"][fold_key]
        STANDALONE_ROWS.append(
            {
                "fold": fold_key,
                "anchor": float(fold_scores["anchor"]["wp"]),
                "basehead": float(fold_scores["standalone"]["basehead"]["wp"]),
                "td5": float(fold_scores["standalone"]["td5"]["wp"]),
                "p2": float(fold_scores["standalone"]["p2"]["wp"]),
                "weak_cohort_mean": float(fold_scores["standalone"]["cohort"]["wp"]),
                "mean_member_standalone_wp": float(
                    np.mean(
                        [
                            fold_scores["standalone"][member_key]["wp"]
                            for member_key in ("basehead", "td5", "p2")
                        ]
                    )
                ),
            }
        )

    E513_RESULT = {
        "experiment_id": "E-513",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": E513_CORE["status"],
        "hypothesis_id": "H-201",
        "arm": "arm:weak-cohort-residual-portfolio",
        "metric_id": "global_wp_clip2_v1",
        "pre_score_gate": PRE_SCORE_GATE,
        "source_and_alignment_audit": AUDIT_SUMMARY,
        "standalone_pure_fold_scores": STANDALONE_ROWS,
        "calibration_fold_scores": E513_CORE["fold_scores"],
        "calibration_selected_lambdas": SELECTED_LAMBDAS,
        "cross_fold_directions": E513_CORE["directions"],
        "weak_pair_residual_diagnostics": RESIDUAL_CORRELATION_DIAGNOSTICS,
        "decision": E513_DECISION,
        "pre_registration": "PRE_REGISTRATION.md",
        "pre_score_addendum": "PRE_SCORE_ADDENDUM_20261007.json",
        "prediction_sidecar_limit": (
            "Legacy prediction arrays have no per-array row-order sidecars. Exact "
            "source hashes, target/chain row-group alignment and four rounded run-card "
            "scores were reproduced; this reconstructs, but does not independently "
            "prove, legacy array row order."
        ),
        "bootstrap_scope": (
            "Conditional fixed-prediction chain bootstrap; does not measure training "
            "variation, adaptive-selection uncertainty, or public transfer."
        ),
        "public_score_forecast": None,
    }

    SOURCE_HASHES_POST_SCORE = {
        relative: sha256_file(ROOT / relative)
        for relative in (
            "src/wnn/gmetric.py",
            "src/wnn/evaluation.py",
            "scripts/evaluate_bandit_continuation.py",
            "scripts/predict_run.py",
            "research/campaign_2026-10-06/bandit_continuation/weak_cohort_portfolio/PRE_REGISTRATION.md",
            "research/campaign_2026-10-06/bandit_continuation/weak_cohort_portfolio/PRE_SCORE_ADDENDUM_20261007.json",
        )
    }
    NOTEBOOK_PATH = PORTFOLIO / "weak_cohort_eda.py"
    NOTEBOOK_HASH_POST_SCORE = sha256_file(NOTEBOOK_PATH)
    E513_RESULT["post_score_source_hashes"] = SOURCE_HASHES_POST_SCORE
    E513_RESULT["notebook_sha256_at_result_write"] = NOTEBOOK_HASH_POST_SCORE

    result_text = json.dumps(E513_RESULT, indent=2, sort_keys=True) + "\n"
    temporary_result = RESULT_PATH.with_suffix(".json.tmp")
    temporary_result.write_text(result_text)
    temporary_result.replace(RESULT_PATH)

    receipt = {
        "experiment_id": "E-513",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_sha256": NOTEBOOK_HASH_POST_SCORE,
        "supporting_source_sha256": SOURCE_HASHES_POST_SCORE,
        "provenance_note": (
            "The source manifest was frozen before cohort scoring and all its 40 "
            "entries matched. Review found that src/wnn/evaluation.py, used only for "
            "centered chain-bootstrap uncertainty, was not among those 40 sources. "
            "Its hash and the notebook hash are captured here after scoring; the "
            "point scorer gmetric.py was included in the pre-score manifest. Treat "
            "the bootstrap implementation as post-score source-bound, not as "
            "pre-score hash-bound."
        ),
        "replay_required": True,
    }
    temporary_receipt = CODE_RECEIPT_PATH.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary_receipt.replace(CODE_RECEIPT_PATH)

    strength_figure, strength_axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    strength_members = ("anchor", "basehead", "td5", "p2", "cohort")
    strength_labels = ("Anchor", "Baseline head", "TD target", "Prop-v2", "Weak mean")
    for plot_index, fold_key in enumerate((0, 1)):
        fold_scores = E513_CORE["fold_scores"][fold_key]
        member_scores = [fold_scores["anchor"]["wp"]] + [
            fold_scores["standalone"][member_key]["wp"]
            for member_key in ("basehead", "td5", "p2", "cohort")
        ]
        strength_axes[plot_index].bar(strength_labels, member_scores, color="#426b8a")
        strength_axes[plot_index].set_title(f"Pure-chain fold {fold_key}")
        strength_axes[plot_index].set_ylabel("Official global WP")
        strength_axes[plot_index].tick_params(axis="x", rotation=25)
        strength_axes[plot_index].set_ylim(0.65, 0.685)
    strength_figure.suptitle("Weak members and their fixed equal-weight mean")
    strength_figure.tight_layout()
    strength_figure.savefig(PLOT_DIRECTORY / "standalone_strength.png", dpi=180)
    plt.close(strength_figure)

    lambda_figure, lambda_axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for plot_index, fold_key in enumerate((0, 1)):
        fold_scores = E513_CORE["fold_scores"][fold_key]
        grid_rows = fold_scores["grid"]
        anchor_wp = fold_scores["anchor"]["wp"]
        lambda_axes[plot_index].plot(
            [row["lambda"] for row in grid_rows],
            [row["cohort_endpoint"]["wp"] - anchor_wp for row in grid_rows],
            marker="o",
            label="Three-member cohort",
        )
        lambda_axes[plot_index].plot(
            [row["lambda"] for row in grid_rows],
            [row["single_endpoint"]["wp"] - anchor_wp for row in grid_rows],
            marker="s",
            label="Basehead control",
        )
        calibration_fold = 1 - fold_key
        lambda_axes[plot_index].axvline(
            SELECTED_LAMBDAS[calibration_fold]["cohort"],
            linestyle="--",
            color="#222222",
            label=f"Cohort lambda from fold {calibration_fold}",
        )
        lambda_axes[plot_index].axhline(0.0, color="#777777", linewidth=0.8)
        lambda_axes[plot_index].set_title(f"Evaluation fold {fold_key}")
        lambda_axes[plot_index].set_xlabel("Portfolio strength lambda")
        lambda_axes[plot_index].set_ylabel("WP delta versus anchor")
        lambda_axes[plot_index].legend(fontsize=8)
    lambda_figure.suptitle("Opposite-fold score curves; vertical line is frozen test lambda")
    lambda_figure.tight_layout()
    lambda_figure.savefig(PLOT_DIRECTORY / "crossfold_lambda.png", dpi=180)
    plt.close(lambda_figure)

    weak_pair_names = ("basehead", "td5", "p2")
    corr_figure, corr_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for plot_index, fold_key in enumerate((0, 1)):
        corr_matrix = np.eye(len(weak_pair_names))
        for left_index, left_name in enumerate(weak_pair_names):
            for right_index in range(left_index + 1, len(weak_pair_names)):
                right_name = weak_pair_names[right_index]
                pair_values = [
                    row["residual_correlation"]
                    for row in RESIDUAL_CORRELATION_DIAGNOSTICS
                    if row["fold"] == fold_key
                    and row["left"] == left_name
                    and row["right"] == right_name
                ]
                pair_mean = float(np.mean(pair_values))
                corr_matrix[left_index, right_index] = pair_mean
                corr_matrix[right_index, left_index] = pair_mean
        image = corr_axes[plot_index].imshow(
            corr_matrix, vmin=0.94, vmax=1.0, cmap="viridis"
        )
        corr_axes[plot_index].set_xticks(range(3), weak_pair_names)
        corr_axes[plot_index].set_yticks(range(3), weak_pair_names)
        corr_axes[plot_index].set_title(f"Pure fold {fold_key}")
        for row_index in range(3):
            for column_index in range(3):
                corr_axes[plot_index].text(
                    column_index,
                    row_index,
                    f"{corr_matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if corr_matrix[row_index, column_index] < 0.975 else "black",
                )
    corr_figure.colorbar(image, ax=corr_axes, label="Weighted residual correlation")
    corr_figure.suptitle("Weak-member residual correlation by target, averaged for display")
    corr_figure.savefig(PLOT_DIRECTORY / "weak_residual_correlation.png", dpi=180)
    plt.close(corr_figure)

    report_lines = [
        "# E-513 result: weak-model portfolio as a residual ensemble",
        "",
        f"Status: {E513_CORE['status']}.",
        "",
        "This is a retrospective, exposed-fold prediction analysis. It does not establish a fresh holdout result, a public-score gain, or training-seed robustness.",
        "",
        "## Integrity gate",
        "",
        f"- Frozen source hashes matched: {AUDIT_SUMMARY['matching_source_hash_count']}/{AUDIT_SUMMARY['source_hash_count']}.",
        f"- Full-fold run-card score checks passed: {RUN_CARD_REPRODUCTION_OK}.",
        f"- Pure-chain groups: fold 0 = {len(PURE_INDICES[0])}; fold 1 = {len(PURE_INDICES[1])}.",
        "- The legacy arrays have no row-order sidecars; alignment is reconstructed from the shared target-cache row order and score reproduction.",
        "",
        "## Cross-fold comparisons",
        "",
        "| Calibration → evaluation | Cohort lambda | Cohort WP | Δ vs anchor | Paired-chain 95% interval | Δ vs single control | Paired-chain 95% interval |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction_row in E513_CORE["directions"]:
        anchor_delta = direction_row["cohort_vs_anchor"]["mean_diff"]
        anchor_interval = direction_row["cohort_vs_anchor"]["ci95"]
        control_delta = direction_row["cohort_vs_single_control"]["mean_diff"]
        control_interval = direction_row["cohort_vs_single_control"]["ci95"]
        report_lines.append(
            f"| {direction_row['calibration_fold']} → {direction_row['evaluation_fold']} "
            f"| {direction_row['cohort_lambda']:.2f} "
            f"| {direction_row['cohort']['wp']:.6f} "
            f"| {anchor_delta:+.6f} "
            f"| [{anchor_interval[0]:+.6f}, {anchor_interval[1]:+.6f}] "
            f"| {control_delta:+.6f} "
            f"| [{control_interval[0]:+.6f}, {control_interval[1]:+.6f}] |"
        )
    report_lines.extend(
        [
            "",
            "Both cohort point deltas versus the anchor are positive, but both intervals include zero. The cohort does not beat the single-member control consistently: its point delta is slightly negative in one direction and positive in the other, with both intervals including zero.",
            "",
            "The selected cohort strengths, 0.35 and 0.40, sit at the top of the frozen grid. This is a useful interaction signal, not permission to extend the grid on these already-used folds.",
            "",
            "## Standalone strength and averaging",
            "",
            "| Pure fold | Anchor | Basehead | TD target | Prop-v2 | Equal weak mean | Mean member standalone WP |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for standalone_row in STANDALONE_ROWS:
        report_lines.append(
            f"| {standalone_row['fold']} "
            f"| {standalone_row['anchor']:.6f} "
            f"| {standalone_row['basehead']:.6f} "
            f"| {standalone_row['td5']:.6f} "
            f"| {standalone_row['p2']:.6f} "
            f"| {standalone_row['weak_cohort_mean']:.6f} "
            f"| {standalone_row['mean_member_standalone_wp']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "The fixed weak mean is stronger than the arithmetic mean of member standalone scores on both folds. That supports studying portfolio averaging as a mechanism, while the anchor comparison remains uncertain.",
            "",
            "## Bandit-tree decision",
            "",
            "Keep H-201 at insufficient evidence. Stop allocation to this exact fixed cohort and blend on these exposed folds. Retain the broader weak-model portfolio family for a materially different candidate cohort, teacher target, or representation contribution. The next confirmation must be separately preregistered and use evaluation units not selected from these same results.",
            "",
            "E-513 does not change the current best public score and is not a submission candidate. Its paired bootstrap is conditional on fixed predictions and does not estimate training-seed variation or public transfer.",
            "",
            "The bootstrap helper source was not included in the original 40-file pre-score manifest. The official point scorer was hash-bound and passed run-card reproduction; the exact bootstrap and notebook source hashes are captured in POST_SCORE_CODE_RECEIPT_E513.json and should be replay-checked before promotion.",
            "",
            "## Figures",
            "",
            "- [Standalone strength](plots/standalone_strength.png)",
            "- [Cross-fold blend curves](plots/crossfold_lambda.png)",
            "- [Weak residual correlation](plots/weak_residual_correlation.png)",
            "",
            "Notebook: [weak_cohort_eda.py](weak_cohort_eda.py).",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    mo.md("\n".join(report_lines))
    return


if __name__ == "__main__":
    app.run()
