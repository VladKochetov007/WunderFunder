"""Grade frozen train cross-fit predictions on every required post-warmup row.

The train file has no ``is_scored`` mask. This is an explicitly nonofficial
internal proxy using all target-bearing prediction rows, not a validation score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from grade_cv import ARMS, TIME_BINS, _component_moments, _paired_component_bootstrap
from source_utils import (
    ROOT,
    load_members_and_folds,
    require_physical_scratch,
    sha256_file,
    verify_recipe_sources,
    verify_train_footer,
    write_json,
)

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.gmetric import GlobalAccumulator

GRADE_RECIPE = (
    "research/campaign_2026-10-18/online_features_r1/group_fast_slow_r1/grade_train_proxy_r1.json"
)


def _score_summary(accumulator: GlobalAccumulator) -> dict[str, Any]:
    """Return scalar fields without carrying internal moment arrays forward."""
    result = accumulator.result()
    return {key: result[key] for key in ("wp", "t0", "t1", "selected_rows", "metric_id")}


def run(
    root: Path,
    fit_recipe_path: Path,
    fit_receipt_path: Path,
    grade_recipe_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score held-out component predictions with an explicit all-train-row proxy."""
    started = time.perf_counter()
    output = require_physical_scratch(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing grade artifacts: {output}")
    output.mkdir(parents=True)

    fit_recipe_bytes = fit_recipe_path.read_bytes()
    fit_recipe_sha256 = hashlib.sha256(fit_recipe_bytes).hexdigest()
    fit_recipe = json.loads(fit_recipe_bytes)
    fit_source_hashes = verify_recipe_sources(root, fit_recipe)
    grade_recipe = json.loads(grade_recipe_path.read_text())
    if grade_recipe["fit_recipe_sha256"] != fit_recipe_sha256:
        raise ValueError("grade addendum does not match the frozen fit recipe")
    grade_code = root / grade_recipe["grade_code_path"]
    observed_grade_code = sha256_file(grade_code)
    if observed_grade_code != grade_recipe["grade_code_sha256"]:
        raise ValueError("train-proxy grader differs from its grade addendum")
    data_contract = root / grade_recipe["data_contract_path"]
    if sha256_file(data_contract) != grade_recipe["data_contract_sha256"]:
        raise ValueError("train-mask contract differs from its grade addendum")

    members, fold_by_component, input_hashes = load_members_and_folds(root, fit_recipe)
    fit_receipt_path = require_physical_scratch(fit_receipt_path.resolve(strict=True))
    fit_receipt_sha256 = sha256_file(fit_receipt_path)
    if fit_receipt_sha256 != grade_recipe["fit_predict_receipt_sha256"]:
        raise ValueError("fit/prediction receipt differs from the pinned receipt")
    fit_receipt = json.loads(fit_receipt_path.read_text())
    if fit_receipt["recipe_sha256"] != fit_recipe_sha256:
        raise ValueError("fit/prediction receipt belongs to another frozen recipe")
    if fit_receipt["source_hashes"] != fit_source_hashes:
        raise ValueError("fit/prediction receipt source hashes differ from the frozen recipe")
    expected_members = [
        {**member, "fold": fold_by_component[member["component"]]} for member in members
    ]
    if fit_receipt["members"] != expected_members:
        raise ValueError("fit/prediction receipt has different sample or fold membership")
    members_path = require_physical_scratch(Path(fit_receipt["members_path"]).resolve(strict=True))
    if sha256_file(members_path) != fit_receipt["members_sha256"]:
        raise ValueError("fit/prediction member manifest hash differs")
    if json.loads(members_path.read_text())["members"] != expected_members:
        raise ValueError("fit/prediction member manifest differs from the frozen sample")
    model_path = require_physical_scratch(Path(fit_receipt["model_archive"]).resolve(strict=True))
    if sha256_file(model_path) != fit_receipt["model_archive_sha256"]:
        raise ValueError("cross-fit model archive hash differs")
    prediction_arrays = {}
    for arm in ARMS:
        prediction_path = require_physical_scratch(
            Path(fit_receipt["prediction_paths"][arm]).resolve(strict=True)
        )
        if sha256_file(prediction_path) != fit_receipt["prediction_sha256"][arm]:
            raise ValueError(f"prediction hash differs for arm {arm}")
        prediction_arrays[arm] = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(members), SEQUENCE_LENGTH - WARMUP, 2)
    if any(array.shape != expected_shape for array in prediction_arrays.values()):
        raise ValueError(f"OOF prediction shape differs from frozen design: {expected_shape}")

    selection = fit_recipe["selection"]
    train_path = root / selection["data_path"]
    data_identity = verify_train_footer(train_path, selection["data_identity"])
    source = SequenceFile(train_path)
    if len(source) != int(selection["row_groups"]):
        raise ValueError("train Parquet row-group count differs from the frozen recipe")
    if source.has_mask:
        raise ValueError("expected train.parquet without is_scored; refusing proxy mask ambiguity")

    accumulators = {arm: GlobalAccumulator() for arm in ARMS}
    fold_accumulators = {
        arm: {fold: GlobalAccumulator() for fold in range(int(fit_recipe["cv"]["fold_count"]))}
        for arm in ARMS
    }
    bin_accumulators = {
        arm: {index: GlobalAccumulator() for index in range(len(TIME_BINS))} for arm in ARMS
    }
    component_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    moments_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    all_rows = np.ones(SEQUENCE_LENGTH - WARMUP, dtype=np.bool_)

    for position, member in enumerate(members):
        sequence = source.read(member["row_group"], targets=True, mask=True)
        if (
            sequence.seq_ix != member["seq_ix"]
            or sequence.targets is None
            or sequence.scored is not None
        ):
            raise ValueError(f"unexpected train identity/target/mask for {member['row_group']}")
        targets = sequence.targets[WARMUP:]
        if targets.shape != (SEQUENCE_LENGTH - WARMUP, 2):
            raise ValueError("train targets do not align with the frozen post-warmup predictions")
        fold = fold_by_component[member["component"]]
        component_moments = {}
        for arm in ARMS:
            predictions = np.asarray(prediction_arrays[arm][position])
            accumulators[arm].add(targets, predictions, all_rows, group=member["component"])
            fold_accumulators[arm][fold].add(targets, predictions, all_rows)
            component_accumulator = GlobalAccumulator()
            component_accumulator.add(targets, predictions, all_rows)
            component_scores[arm].append(float(component_accumulator.result()["wp"]))
            component_moments[arm] = _component_moments(targets, predictions, all_rows)
            for bin_index, (row_start, row_end) in enumerate(TIME_BINS):
                begin, end = row_start - WARMUP, row_end - WARMUP
                bin_accumulators[arm][bin_index].add(
                    targets[begin:end], predictions[begin:end], all_rows[begin:end]
                )
        for arm in ARMS:
            moments_by_arm[arm].append(component_moments[arm])
        del sequence

    pooled = {arm: _score_summary(accumulators[arm]) for arm in ARMS}
    folds = {
        arm: {
            str(fold): _score_summary(fold_accumulators[arm][fold])
            for fold in fold_accumulators[arm]
        }
        for arm in ARMS
    }
    position_bins = {
        arm: {
            f"{TIME_BINS[index][0]}_{TIME_BINS[index][1]}": _score_summary(
                bin_accumulators[arm][index]
            )
            for index in bin_accumulators[arm]
        }
        for arm in ARMS
    }
    deltas = {arm: pooled[arm]["wp"] - pooled["raw"]["wp"] for arm in ARMS[1:]}
    bootstrap = _paired_component_bootstrap(
        {arm: np.stack(values) for arm, values in moments_by_arm.items()}, len(members)
    )
    for arm in ARMS[1:]:
        bootstrap[arm]["point_delta"] = float(deltas[arm])

    result = {
        "experiment_id": fit_recipe["experiment_id"],
        "metric_id": "global_wp_clip2_v1_formula_train_all_required_proxy",
        "metric_formula": "global_wp_clip2_v1",
        "status": "completed_internal_train_component_crossfit_proxy",
        "mask_policy": "train.parquet has no is_scored; score every target row at steps 99..19999 for this internal proxy",
        "official_validation_score": False,
        "sample": {
            "components": len(members),
            "rows_per_component": SEQUENCE_LENGTH,
            "fit_membership": "32 previously exposed fold0.fit.train components, grouped 4-fold cross-fit",
            "out_of_fold_scoring": True,
            "all_post_warmup_train_rows_selected": True,
        },
        "pooled_oof_scores": pooled,
        "delta_vs_raw": deltas,
        "paired_component_bootstrap": bootstrap,
        "fold_scores": folds,
        "position_bin_scores": position_bins,
        "component_diagnostics": {
            arm: {
                "median_component_wp": float(np.median(values)),
                "component_win_rate_vs_raw": (
                    None
                    if arm == "raw"
                    else float(np.mean(np.asarray(values) > np.asarray(component_scores["raw"])))
                ),
            }
            for arm, values in component_scores.items()
        },
        "uncertainty_scope": "Paired component bootstrap conditions on four fixed CV models and 32 previously exposed fit-side components; it excludes seed, selection, official-mask, and public-transfer uncertainty.",
        "claim_limit": "This is a development-only all-train-rows proxy using the official metric formula, not an official validation score. It does not establish recurrent endpoint gain, competition transfer, online feedback, or long-sequence generalization.",
        "fit_predict_receipt_sha256": fit_receipt_sha256,
        "fit_recipe_sha256": fit_recipe_sha256,
        "fit_source_hashes": fit_source_hashes,
        "grade_code_sha256": observed_grade_code,
        "sample_source_hashes": input_hashes,
        "data_identity": data_identity,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
    }
    result_path = output / "grade_result.json"
    write_json(result_path, result)
    receipt = {
        "experiment_id": fit_recipe["experiment_id"],
        "phase": "internal_train_all_required_rows_proxy_grade_complete",
        "status": "completed",
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "fit_recipe_sha256": fit_recipe_sha256,
        "grade_recipe_sha256": sha256_file(grade_recipe_path),
        "fit_predict_receipt_sha256": fit_receipt_sha256,
        "mask_policy": result["mask_policy"],
        "pooled_scores": {arm: pooled[arm]["wp"] for arm in ARMS},
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
    }
    receipt_path = output / "grade_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "result": str(result_path)}), flush=True)
    return receipt


def parse_args() -> argparse.Namespace:
    """Parse the frozen fit and grade recipes, receipt, and scratch destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit-recipe", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--grade-recipe", type=Path, default=ROOT / GRADE_RECIPE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the separate proxy grader after all prediction hashes are frozen."""
    args = parse_args()
    run(
        args.root.resolve(),
        args.fit_recipe.resolve(),
        args.fit_receipt,
        args.grade_recipe.resolve(),
        args.output,
    )


if __name__ == "__main__":
    main()
