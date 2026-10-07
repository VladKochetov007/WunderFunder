"""Grade a frozen two-seed E521 equal pair on exposed fold-0 groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
E521_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_bandit_continuation as evaluator  # noqa: E402

from wnn.evaluation import (  # noqa: E402
    paired_component_bootstrap,
    paired_component_moments,
)
from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

RUNS = {3: "bd26_w384raw_s3_f0", 4: "bd26_w384raw_s4_f0"}
EPOCH = 3
SEQUENCE_ROWS = 20_000
EXPECTED_EVALUATION_GROUPS = 468
EXPECTED_EVALUATION_COMPONENTS = 141
EXPECTED_TRAIN_GROUPS = 10_535
EXPECTED_VALID_FIT_GROUPS = 1_405
EXPECTED_SELECTED_ROWS = 808_863
CORRUPT_TRAIN_GROUP = 528
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 271_828
SEQUENCES_PER_BATCH = 4
SPEC_PATH = E521_DIR / "E521_EQUAL_PAIR_EXCLUDED_FOLD0_GRADING_SPEC_V4.json"


def _mapping(value: object, label: str) -> dict[str, object]:
    """Require decoded JSON to be an object at an artifact boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object and reject other top-level values."""
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _sha256(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """Stop before scoring when a frozen identity or scientific gate fails."""
    if not condition:
        raise ValueError(message)


def _verify_frozen_files(spec: dict[str, object]) -> dict[str, str]:
    """Rehash every preregistered source, split, fit, and prediction artifact."""
    raw_files = spec.get("artifacts")
    if not isinstance(raw_files, dict):
        raise ValueError("grading spec lacks its frozen artifact map")
    observed: dict[str, str] = {}
    for relative_path, expected_value in cast(dict[str, object], raw_files).items():
        if not isinstance(expected_value, str) or len(expected_value) != 64:
            raise ValueError(f"invalid frozen SHA-256 for {relative_path}")
        path = (ROOT / relative_path).resolve(strict=True)
        if not path.is_file() or ROOT not in path.parents:
            raise ValueError(f"frozen artifact is not a repository file: {relative_path}")
        actual = _sha256(path)
        if actual != expected_value:
            raise ValueError(f"frozen artifact hash changed: {relative_path}")
        observed[relative_path] = actual
    script_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if script_relative not in observed:
        raise ValueError("grader source is not bound by the pre-score specification")
    return observed


def _verify_attempts_and_recipe(
    seed: int, observed: Mapping[str, str]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Bind terminal fit/export attempts to the per-seed frozen recipe."""
    run = RUNS[seed]
    attempt_root = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts"
    fit = _read_json(attempt_root / f"e521-w384-s{seed}-fit-{('v2' if seed == 3 else 'v1')}.json")
    export_version = "v1" if seed == 3 else "v2"
    export = _read_json(attempt_root / f"e521-w384-s{seed}-export-{export_version}.json")
    _require(
        fit.get("status") == "finished" and fit.get("exit_status") == 0,
        f"seed {seed}: fit is not terminal-success",
    )
    _require(
        export.get("status") == "finished" and export.get("exit_status") == 0,
        f"seed {seed}: validation export is not terminal-success",
    )
    fit_command = fit.get("command")
    export_command = export.get("command")
    _require(isinstance(fit_command, list), f"seed {seed}: fit receipt lacks argv")
    _require(isinstance(export_command, list), f"seed {seed}: export receipt lacks argv")
    fit_command = list(map(str, cast(list[object], fit_command)))
    export_command = list(map(str, cast(list[object], export_command)))

    if seed == 3:
        recipe = _read_json(E521_DIR / "RECIPE.json")
        declared_fit = recipe.get("conditional_fit_command")
        _require(isinstance(declared_fit, list), "seed 3 recipe lacks its fit argv")
        declared_fit = list(map(str, cast(list[object], declared_fit)))
        try:
            trainer_index = fit_command.index("scripts/train_rnn.py")
        except ValueError as error:
            raise ValueError("seed 3 fit did not invoke the frozen trainer") from error
        _require(
            fit_command[trainer_index:] == declared_fit[1:],
            "seed 3 actual fit argv differs from the frozen recipe",
        )
        evaluator._verify_command_arguments(
            _read_json(ROOT / "artifacts/runs" / run / "provenance.json")["args"],
            declared_fit,
            run,
        )
        export_contract = _read_json(E521_DIR / "EXPORT_CONTRACT_V1.json")
    else:
        fit_contract = _read_json(E521_DIR / "SEED4_CONTRACT_V1.json")
        declared_fit = fit_contract.get("command")
        _require(isinstance(declared_fit, list), "seed 4 fit contract lacks its argv")
        _require(
            fit_command == list(map(str, cast(list[object], declared_fit))),
            "seed 4 actual fit argv differs from its sealed contract",
        )
        declared_tail = list(map(str, cast(list[object], declared_fit)))
        try:
            trainer_index = declared_tail.index("scripts/train_rnn.py")
        except ValueError as error:
            raise ValueError("seed 4 contract did not invoke the frozen trainer") from error
        normalized = ["python", *declared_tail[trainer_index:]]
        evaluator._verify_command_arguments(
            _read_json(ROOT / "artifacts/runs" / run / "provenance.json")["args"],
            normalized,
            run,
        )
        export_contract = _read_json(E521_DIR / "SEED4_EXPORT_CONTRACT_V2.json")
        declared_export = export_contract.get("command")
        _require(
            isinstance(declared_export, list)
            and export_command == list(map(str, cast(list[object], declared_export))),
            "seed 4 actual export argv differs from its sealed contract",
        )

    try:
        export_index = export_command.index("scripts/predict_run.py")
    except ValueError as error:
        raise ValueError(f"seed {seed}: export did not invoke predict_run.py") from error
    _require(
        export_command[export_index:]
        == ["scripts/predict_run.py", run, "valid", "--epoch", str(EPOCH)],
        f"seed {seed}: actual export differs from fixed valid epoch 3",
    )
    trainer_hashes = fit.get("source_sha256")
    _require(isinstance(trainer_hashes, dict), f"seed {seed}: fit lacks source hashes")
    _require(
        cast(dict[str, object], trainer_hashes).get("scripts/train_rnn.py")
        == observed["scripts/train_rnn.py"],
        f"seed {seed}: fit receipt names another trainer",
    )
    export_sources = export.get("source_sha256")
    _require(isinstance(export_sources, dict), f"seed {seed}: export lacks source hashes")
    _require(
        cast(dict[str, object], export_sources).get("scripts/predict_run.py")
        == observed["scripts/predict_run.py"],
        f"seed {seed}: export receipt names another predictor",
    )
    declared_output = export_contract.get("output")
    if declared_output is not None:
        _require(
            declared_output == f"cache/pred_{run}_valid.npy",
            f"seed {seed}: export contract names another output",
        )
    else:
        _require(
            seed == 4
            and export_contract.get("checkpoint_path")
            == f"artifacts/runs/{run}/model_ep{EPOCH}.pt",
            f"seed {seed}: output path is not bound by its export contract",
        )
    return fit, export, export_contract


def _verify_member(
    seed: int,
    split: object,
    row_groups: list[int],
    observed: Mapping[str, str],
) -> tuple[np.memmap, dict[str, object], dict[str, object]]:
    """Validate one independently trained fixed-epoch member and its ancestry."""
    run = RUNS[seed]
    provenance_path = ROOT / "artifacts/runs" / run / "provenance.json"
    metadata_path = ROOT / "cache" / f"pred_{run}_valid.json"
    prediction, binding = evaluator._member(
        ROOT,
        run,
        seed,
        EPOCH,
        split,
        row_groups,
        expected_init=None,
    )
    provenance = _read_json(provenance_path)
    metadata = _read_json(metadata_path)
    args = _mapping(provenance.get("args"), f"seed {seed} fit args")
    _require(
        args.get("name") == run and args.get("seed") == seed, f"seed {seed}: run identity mismatch"
    )
    _require(
        args.get("arch") == "gru"
        and args.get("cell") == "gru"
        and args.get("hidden") == 384
        and args.get("layers") == 2,
        f"seed {seed}: model architecture differs from E521 GRU-384",
    )
    _require(
        args.get("checkpoint_policy") == "fixed"
        and args.get("checkpoint_epoch") == EPOCH
        and args.get("no_eval_during_fit") is True,
        f"seed {seed}: fit used selection or another epoch",
    )
    _require(
        args.get("init") == ""
        and args.get("raw_only") is True
        and args.get("distill") == ""
        and args.get("distill_cache_root") == ""
        and args.get("distill_family_weight") == 0
        and args.get("propensity") == ""
        and args.get("aux") == "",
        f"seed {seed}: model has undeclared initialization or auxiliary ancestry",
    )
    _require(not binding.get("lineage"), f"seed {seed}: scratch model has ancestor lineage")
    fit_groups = _mapping(provenance.get("fit_row_groups"), f"seed {seed} fit groups")
    train_groups = set(map(int, cast(list[object], fit_groups.get("train", []))))
    valid_groups = set(map(int, cast(list[object], fit_groups.get("valid", []))))
    evaluation_groups = set(
        map(int, cast(list[object], provenance.get("evaluation_row_groups", [])))
    )
    _require(
        len(train_groups) == EXPECTED_TRAIN_GROUPS, f"seed {seed}: train membership size changed"
    )
    _require(
        len(valid_groups) == EXPECTED_VALID_FIT_GROUPS,
        f"seed {seed}: fit-valid membership size changed",
    )
    _require(
        train_groups == set(split.train_groups) and valid_groups == set(split.valid_groups),
        f"seed {seed}: direct fit groups differ from the complete frozen fitting split",
    )
    _require(
        CORRUPT_TRAIN_GROUP not in train_groups,
        f"seed {seed}: corrupt row group {CORRUPT_TRAIN_GROUP} entered fit",
    )
    _require(
        evaluation_groups == set(split.evaluation_groups),
        f"seed {seed}: recorded evaluation membership differs from fold 0",
    )
    split.assert_fit_groups(train_groups, valid_groups)
    _require(
        metadata.get("run") == run
        and metadata.get("split") == "valid"
        and metadata.get("epoch") == EPOCH,
        f"seed {seed}: prediction sidecar names another run or epoch",
    )
    _require(
        metadata.get("checkpoint_sha256") == observed[f"artifacts/runs/{run}/model_ep3.pt"]
        and metadata.get("prediction_sha256") == observed[f"cache/pred_{run}_valid.npy"],
        f"seed {seed}: prediction sidecar is not bound to frozen checkpoint and bytes",
    )
    _require(
        metadata.get("prediction_sha256") == binding.get("prediction_sha256"),
        f"seed {seed}: evaluator binding differs from frozen prediction",
    )
    _require(
        metadata.get("split_manifest_sha256")
        == observed["research/campaign_2026-09-29/artifacts/splits/fold0.json"],
        f"seed {seed}: sidecar names another split manifest",
    )
    return prediction, binding, provenance


def _score(spec_path: Path) -> dict[str, object]:
    """Run the fixed members, pair, and conditional component contrasts once."""
    if spec_path.resolve(strict=True) != SPEC_PATH.resolve():
        raise ValueError("this grader accepts only the frozen E521 equal-pair spec")
    spec = _read_json(spec_path)
    _require(
        spec.get("status") == "frozen_before_candidate_prediction_scoring"
        and spec.get("metric_id") == METRIC_ID,
        "grading spec is not frozen to the official metric",
    )
    outputs = _mapping(spec.get("outputs"), "grade outputs")
    result_path = ROOT / str(outputs["result"])
    report_path = ROOT / str(outputs["report"])
    if result_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a previous E521 equal-pair grade")
    observed = _verify_frozen_files(spec)
    for seed in RUNS:
        _verify_attempts_and_recipe(seed, observed)

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = load_campaign_split(split_path, ROOT)
    _require(
        split.fold == 0 and split.sha256 == observed[split_path.relative_to(ROOT).as_posix()],
        "fold-0 split identity changed",
    )
    _require(
        len(split.evaluation_groups) == EXPECTED_EVALUATION_GROUPS,
        "fold-0 evaluation group count changed",
    )
    members: dict[int, np.memmap] = {}
    bindings: dict[int, dict[str, object]] = {}
    provenances: dict[int, dict[str, object]] = {}

    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        targets = cache["y"]
        scored = cache["mask"].astype(bool, copy=False)
        market_class = cache["market_class"]
        expected_groups = evaluator._expected_target_groups(
            split.valid_groups, split.evaluation_groups
        )
        evaluator._validate_target_index(row_groups, targets, scored, market_class, expected_groups)
        positions = {group: index for index, group in enumerate(row_groups)}
        groups = list(map(int, split.evaluation_groups))
        if any(group not in positions for group in groups):
            raise ValueError("fold-0 evaluation group is absent from validation targets")
        for seed in RUNS:
            members[seed], bindings[seed], provenances[seed] = _verify_member(
                seed, split, row_groups, observed
            )
        for seed in RUNS:
            _require(
                provenances[seed].get("fit_row_groups") == provenances[3].get("fit_row_groups"),
                f"seed {seed}: independently trained member uses different split membership",
            )

        component_by_group = {
            int(group): int(component)
            for source, group, component in split.components
            if source == "valid"
        }
        _require(
            all(group in component_by_group for group in groups),
            "fold-0 component membership is incomplete",
        )
        components = np.asarray([component_by_group[group] for group in groups], dtype=np.int64)
        component_membership = [
            {"row_group": group, "component": int(component)}
            for group, component in zip(groups, components, strict=True)
        ]
        component_count = len(set(map(int, components)))
        _require(
            component_count == EXPECTED_EVALUATION_COMPONENTS,
            "fold-0 connected-component count changed",
        )
        indices = np.asarray([positions[group] for group in groups], dtype=np.int64)
        mask = evaluator._official_mask(np.asarray(scored[indices], dtype=bool))
        selected_rows = int(mask.sum())
        _require(selected_rows == EXPECTED_SELECTED_ROWS, "official selected-row count changed")
        accumulators = {key: GlobalAccumulator() for key in (3, 4, "equal_pair")}
        pair_moments_3: list[np.ndarray] = []
        reference_moments_3: list[np.ndarray] = []
        pair_moments_4: list[np.ndarray] = []
        reference_moments_4: list[np.ndarray] = []
        for start in range(0, len(indices), SEQUENCES_PER_BATCH):
            stop = min(start + SEQUENCES_PER_BATCH, len(indices))
            take = indices[start:stop]
            y = np.asarray(targets[take], dtype=np.float32)
            prediction_3 = np.asarray(members[3][take], dtype=np.float32)
            prediction_4 = np.asarray(members[4][take], dtype=np.float32)
            pair = evaluator._blend_equal_pair(prediction_3, prediction_4)
            batch_mask = mask[start:stop]
            accumulators[3].add(y, prediction_3, batch_mask)
            accumulators[4].add(y, prediction_4, batch_mask)
            accumulators["equal_pair"].add(y, pair, batch_mask)
            candidate_3, reference_3 = paired_component_moments(y, pair, prediction_3, batch_mask)
            candidate_4, reference_4 = paired_component_moments(y, pair, prediction_4, batch_mask)
            pair_moments_3.append(candidate_3)
            reference_moments_3.append(reference_3)
            pair_moments_4.append(candidate_4)
            reference_moments_4.append(reference_4)

    score_3 = accumulators[3].result()
    score_4 = accumulators[4].result()
    score_pair = accumulators["equal_pair"].result()
    pair3 = np.concatenate(pair_moments_3, axis=0)
    ref3 = np.concatenate(reference_moments_3, axis=0)
    pair4 = np.concatenate(pair_moments_4, axis=0)
    ref4 = np.concatenate(reference_moments_4, axis=0)
    _require(np.array_equal(pair3, pair4), "pair moments differ across the two fixed contrasts")
    delta_3 = paired_component_bootstrap(
        pair3,
        ref3,
        components,
        n_boot=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    delta_4 = paired_component_bootstrap(
        pair4,
        ref4,
        components,
        n_boot=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    for observed_score in (score_3, score_4, score_pair):
        _require(
            observed_score.get("metric_id") == METRIC_ID
            and observed_score.get("selected_rows") == EXPECTED_SELECTED_ROWS,
            "official accumulator returned an unexpected metric or row count",
        )

    old_s3 = _read_json(E521_DIR / "E521_S3_EXCLUDED_FOLD0_RESULT_V2.json")
    old_score = _mapping(
        _mapping(old_s3.get("metric"), "historical metric").get("score"), "historical score"
    )
    _require(
        abs(float(score_3["wp"]) - float(old_score["wp"])) <= 1e-8
        and abs(float(score_3["t0"]) - float(old_score["t0"])) <= 1e-8
        and abs(float(score_3["t1"]) - float(old_score["t1"])) <= 1e-8,
        "seed-3 score does not reproduce its prior frozen independent grade",
    )

    return {
        "schema_version": 1,
        "grade_id": "E521_EQUAL_PAIR_EXCLUDED_FOLD0_V4",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_descriptive_grade",
        "claim": "descriptive scores for two independently scratch-trained E521 GRU-384 fixed-epoch members and their preregistered equal raw-prediction average on exposed fold-0 groups",
        "comparison": {
            "primary": "equal_pair_minus_each_member",
            "blend": "0.5 * raw seed-3 prediction + 0.5 * raw seed-4 prediction; official metric clips after blending",
            "training_seed_variability_estimated": False,
            "matched_width_control": None,
            "width_effect_claim": None,
            "public_forecast": None,
        },
        "metric": {
            "id": METRIC_ID,
            "implementation": "src/wnn/gmetric.py:GlobalAccumulator",
            "members": {"seed3": score_3, "seed4": score_4, "equal_pair": score_pair},
            "selection": "need_prediction AND is_scored",
            "warmup_rows_excluded": WARMUP,
            "prediction_and_target_clipping": [-2.0, 2.0],
            "weights": "absolute clipped label magnitude",
            "evaluation_aggregation": "pool selected rows per target, then mean the two weighted Pearson correlations",
        },
        "conditional_component_contrasts": {
            "pair_minus_seed3": _bootstrap_summary(delta_3),
            "pair_minus_seed4": _bootstrap_summary(delta_4),
            "interpretation": "2,000 paired connected-component resamples of fixed predictions; intervals condition on these two fitted checkpoints and do not estimate training-seed variability, unseen-fold performance, or public transfer",
        },
        "evaluation": {
            "split_path": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "split_sha256": observed["research/campaign_2026-09-29/artifacts/splits/fold0.json"],
            "exposure": "exposed development data previously inspected and used in research decisions; not fresh confirmation",
            "evaluation_group_count": len(groups),
            "connected_component_count": component_count,
            "selected_scored_rows": selected_rows,
            "row_groups": groups,
            "component_membership": component_membership,
            "target_cache_sha256": observed["cache/valid_targets.npz"],
            "target_row_groups_sha256": evaluator._list_sha256(row_groups),
        },
        "models": {
            f"seed{seed}": {
                "run": RUNS[seed],
                "seed": seed,
                "checkpoint_epoch_zero_based": EPOCH,
                "checkpoint_sha256": bindings[seed]["checkpoint_sha256"],
                "prediction_sha256": bindings[seed]["prediction_sha256"],
                "prediction_metadata_sha256": observed[f"cache/pred_{RUNS[seed]}_valid.json"],
                "fit_provenance_sha256": observed[f"artifacts/runs/{RUNS[seed]}/provenance.json"],
                "prediction_shape": list(members[seed].shape),
                "prediction_dtype": str(members[seed].dtype),
                "inference_precision": bindings[seed].get("inference_precision"),
                "initialization": "independent scratch training; no teammate or teacher weights",
            }
            for seed in RUNS
        },
        "uncertainty_and_selection": {
            "training_seeds_scored": [3, 4],
            "evaluation_interval": "conditional connected-component bootstrap only",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "checkpoint_or_blend_search": False,
            "further_selection_from_this_grade": False,
            "no_public_forecast": True,
        },
        "frozen_spec_path": SPEC_PATH.relative_to(ROOT).as_posix(),
        "frozen_spec_sha256": _sha256(SPEC_PATH),
        "artifact_hashes": observed,
        "integrity_checks": {
            "fit_and_export_terminal_success_for_both_seeds": True,
            "exact_recipe_and_fixed_epoch3_binding": True,
            "scratch_raw_only_independent_members": True,
            "checkpoint_prediction_sidecar_binding_for_both_seeds": True,
            "full_fold_membership_and_component_validation": True,
            "corrupt_train_group_528_excluded_for_both_seeds": True,
            "only_frozen_evaluation_groups_scored": True,
            "seed3_reproduces_prior_independent_grade": True,
        },
    }


def _bootstrap_summary(value: Mapping[str, object]) -> dict[str, object]:
    """Keep the paired interval but omit its test-like one-sided tail fraction."""
    keys = (
        "candidate_wp",
        "reference_wp",
        "mean_diff",
        "bootstrap_sd",
        "ci95",
        "n_sequences",
        "n_components",
        "n_boot",
        "seed",
        "metric_id",
        "centered_moments",
    )
    return {key: value[key] for key in keys}


def _markdown_report(result: Mapping[str, object]) -> str:
    """Render model scores and comparisons with their exposure limits."""
    metric = _mapping(result["metric"], "result metric")
    members = _mapping(metric["members"], "member scores")
    contrasts = _mapping(result["conditional_component_contrasts"], "contrasts")
    eval_data = _mapping(result["evaluation"], "evaluation")
    lines = [
        "# E521 equal-pair descriptive grade on exposed fold 0 (V4)",
        "",
        f"Official metric: `{metric['id']}`; both checkpoints are fixed at zero-based epoch {EPOCH}.",
        "",
        "| Predictor | WP | T0 | T1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("seed3", "seed4", "equal_pair"):
        score = _mapping(members[name], f"{name} score")
        lines.append(
            f"| {name} | {float(score['wp']):.8f} | {float(score['t0']):.8f} | {float(score['t1']):.8f} |"
        )
    lines.extend(["", "Paired component-bootstrap contrasts for the equal pair:", ""])
    for name in ("pair_minus_seed3", "pair_minus_seed4"):
        value = _mapping(contrasts[name], name)
        lo, hi = cast(list[float], value["ci95"])
        lines.append(
            f"- {name}: {float(value['mean_diff']):+.8f}; conditional 95% component interval [{lo:+.8f}, {hi:+.8f}]."
        )
    lines.extend(
        [
            "",
            f"Evaluation used {eval_data['evaluation_group_count']} fold-0 groups across {eval_data['connected_component_count']} connected components and {eval_data['selected_scored_rows']:,} official selected rows.",
            "",
            "These are exposed development groups, not fresh confirmation. The resampling intervals condition on the two fitted predictors; they do not measure training-seed variability. There is no matched GRU-128 control, so this does not estimate a width effect. No public forecast is made.",
            "",
            f"Frozen grading spec SHA-256: `{result['frozen_spec_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def grade(spec_path: Path) -> dict[str, object]:
    """Score the exact preregistered two-member comparison once."""
    result = _score(spec_path)
    outputs = _mapping(_read_json(spec_path).get("outputs"), "grade outputs")
    result_path = ROOT / str(outputs["result"])
    report_path = ROOT / str(outputs["report"])
    if result_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a previous E521 equal-pair grade")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    return result


def main() -> None:
    """Run the single frozen grade from its preregistered specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args()
    result = grade(args.spec)
    metric = _mapping(result["metric"], "result metric")
    print(json.dumps(metric["members"], sort_keys=True))


if __name__ == "__main__":
    main()
