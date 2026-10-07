"""Grade one frozen E521 seed-3 prediction on excluded fold-0 groups only."""

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

from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

RUN = "bd26_w384raw_s3_f0"
EPOCH = 3
SEED = 3
SEQUENCE_ROWS = 20_000
EXPECTED_EVALUATION_GROUPS = 468
EXPECTED_EVALUATION_COMPONENTS = 141
EXPECTED_TRAIN_GROUPS = 10_535
EXPECTED_VALID_FIT_GROUPS = 1_405
CORRUPT_TRAIN_GROUP = 528
SPEC_PATH = E521_DIR / "E521_S3_EXCLUDED_FOLD0_GRADING_SPEC_V1.json"


def _mapping(value: object, label: str) -> dict[str, object]:
    """Require a decoded JSON object at a trust boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object and reject other top-level values."""
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """Fail the grade before scoring whenever a frozen identity is wrong."""
    if not condition:
        raise ValueError(message)


def _verify_frozen_files(spec: dict[str, object]) -> dict[str, str]:
    """Rehash every preregistered code, data, fit, and prediction artifact."""
    raw_files = spec.get("artifacts")
    if not isinstance(raw_files, dict):
        raise ValueError("grading spec lacks its frozen artifact map")
    artifacts = cast(dict[str, object], raw_files)
    observed: dict[str, str] = {}
    for relative_path, expected_value in artifacts.items():
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
        raise ValueError("grading script is not bound by the pre-score specification")
    return observed


def _verify_fit_contract(
    spec: dict[str, object], observed_hashes: Mapping[str, str]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Bind fit, fixed checkpoint, export, and sidecar to the pre-score recipe."""
    base = E521_DIR
    fit = _read_json(
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/e521-w384-s3-fit-v2.json"
    )
    export = _read_json(
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/e521-w384-s3-export-v1.json"
    )
    provenance = _read_json(ROOT / "artifacts/runs" / RUN / "provenance.json")
    metadata = _read_json(ROOT / "cache" / f"pred_{RUN}_valid.json")
    result = _read_json(base / "FIT_EXPORT_RESULT_V1.json")
    recipe = _read_json(base / "RECIPE.json")
    export_contract = _read_json(base / "EXPORT_CONTRACT_V1.json")

    _require(
        fit.get("status") == "finished" and fit.get("exit_status") == 0,
        "fit attempt is not successful and terminal",
    )
    _require(
        export.get("status") == "finished" and export.get("exit_status") == 0,
        "prediction export is not successful and terminal",
    )
    _require(
        provenance.get("seed") == SEED and provenance.get("selected_epoch") == EPOCH,
        "run provenance does not bind seed 3 fixed epoch 3",
    )
    _require(
        metadata.get("run") == RUN
        and metadata.get("split") == "valid"
        and metadata.get("epoch") == EPOCH,
        "prediction sidecar names another run, split, or epoch",
    )
    _require(
        result.get("fixed_epoch") == EPOCH and result.get("scientific_score") is None,
        "pre-score export result is not fixed epoch 3 with no prior score",
    )
    _require(
        export_contract.get("output") == "cache/pred_bd26_w384raw_s3_f0_valid.npy",
        "export contract names an unexpected prediction path",
    )

    checkpoint_relative = f"artifacts/runs/{RUN}/model_ep{EPOCH}.pt"
    prediction_relative = f"cache/pred_{RUN}_valid.npy"
    _require(
        metadata.get("checkpoint_sha256") == observed_hashes[checkpoint_relative],
        "sidecar checkpoint hash differs from checkpoint bytes",
    )
    _require(
        metadata.get("prediction_sha256") == observed_hashes[prediction_relative],
        "sidecar prediction hash differs from prediction bytes",
    )
    checkpoint_hashes = _mapping(provenance.get("checkpoint_sha256"), "checkpoint hashes")
    _require(
        checkpoint_hashes.get("model_ep3.pt") == observed_hashes[checkpoint_relative],
        "trainer provenance checkpoint hash differs from checkpoint bytes",
    )
    _require(
        provenance.get("split_manifest_sha256")
        == observed_hashes["research/campaign_2026-09-29/artifacts/splits/fold0.json"],
        "fit provenance split hash differs from frozen fold-0 manifest",
    )
    _require(
        metadata.get("split_manifest_sha256") == provenance.get("split_manifest_sha256"),
        "prediction sidecar and fit provenance name different splits",
    )
    _require(
        export.get("source_sha256", {}).get("scripts/train_rnn.py")
        == observed_hashes["scripts/train_rnn.py"],
        "export attempt trainer hash differs from frozen trainer",
    )
    _require(
        fit.get("source_sha256", {}).get("scripts/train_rnn.py")
        == observed_hashes["scripts/train_rnn.py"],
        "fit attempt trainer hash differs from frozen trainer",
    )

    args = _mapping(provenance.get("args"), "trainer provenance arguments")
    _require(
        args.get("name") == RUN and args.get("seed") == SEED,
        "trainer provenance run name or seed differs",
    )
    _require(
        args.get("checkpoint_policy") == "fixed" and args.get("checkpoint_epoch") == EPOCH,
        "checkpoint was not fixed to the preregistered epoch",
    )
    _require(args.get("no_eval_during_fit") is True, "fit ran in-sample evaluation during training")
    _require(
        args.get("init") == "" and args.get("raw_only") is True,
        "candidate is not a scratch raw-only model",
    )
    _require(
        args.get("distill") == ""
        and args.get("distill_cache_root") == ""
        and args.get("distill_family_weight") == 0,
        "candidate has an undeclared teacher or distillation ancestry",
    )
    _require(
        args.get("propensity") == "" and args.get("aux") == "",
        "candidate has an undeclared propensity or auxiliary target",
    )
    _require(
        args.get("valid_fold") == 0 and args.get("include_heldout_valid") is False,
        "fit provenance does not exclude fold-0 evaluation groups",
    )

    command = recipe.get("conditional_fit_command")
    if not isinstance(command, list):
        raise ValueError("E521 recipe has no conditional fit command")
    fit_command = fit.get("command")
    if not isinstance(fit_command, list):
        raise ValueError("terminal fit attempt has no argv")
    try:
        trainer_index = list(map(str, fit_command)).index("scripts/train_rnn.py")
    except ValueError as error:
        raise ValueError("terminal fit attempt did not invoke the frozen trainer") from error
    _require(
        list(map(str, fit_command[trainer_index:])) == list(map(str, command[1:])),
        "actual fit argv differs from the frozen E521 recipe",
    )
    evaluator._verify_command_arguments(args, list(map(str, command)), RUN)

    _require(
        result.get("prediction_sha256") == observed_hashes[prediction_relative],
        "export result prediction digest differs",
    )
    _require(
        result.get("metadata_sha256") == observed_hashes[f"cache/pred_{RUN}_valid.json"],
        "export result metadata digest differs",
    )
    _require(
        result.get("checkpoint_sha256") == observed_hashes[checkpoint_relative],
        "export result checkpoint digest differs",
    )
    return provenance, metadata, export_contract


def grade(spec_path: Path) -> dict[str, object]:
    """Score the preregistered fixed predictor on the fold-0 evaluation panel."""
    spec_path = spec_path.resolve(strict=True)
    if spec_path != SPEC_PATH.resolve():
        raise ValueError("this grader accepts only the frozen E521 seed-3 spec")
    spec = _read_json(spec_path)
    _require(
        spec.get("status") == "frozen_before_candidate_prediction_scoring",
        "grade specification is not frozen",
    )
    _require(
        spec.get("metric_id") == METRIC_ID,
        "frozen metric differs from official metric implementation",
    )
    observed_hashes = _verify_frozen_files(spec)
    provenance, metadata, _ = _verify_fit_contract(spec, observed_hashes)

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = load_campaign_split(split_path, ROOT)
    _require(
        split.fold == 0
        and split.sha256 == observed_hashes[split_path.relative_to(ROOT).as_posix()],
        "loaded fold-0 split identity differs",
    )
    _require(
        len(split.evaluation_groups) == EXPECTED_EVALUATION_GROUPS,
        "fold-0 evaluation membership count changed",
    )

    raw_fit_groups = provenance.get("fit_row_groups")
    if not isinstance(raw_fit_groups, dict):
        raise ValueError("trainer provenance lacks direct fit row-group membership")
    fit_groups = cast(dict[str, object], raw_fit_groups)
    train_groups = set(map(int, cast(list[object], fit_groups.get("train", []))))
    valid_groups = set(map(int, cast(list[object], fit_groups.get("valid", []))))
    evaluation_groups = set(
        map(int, cast(list[object], provenance.get("evaluation_row_groups", [])))
    )
    _require(
        len(train_groups) == EXPECTED_TRAIN_GROUPS and train_groups == set(split.train_groups),
        "training row-group membership differs from the frozen manifest",
    )
    _require(
        len(valid_groups) == EXPECTED_VALID_FIT_GROUPS and valid_groups == set(split.valid_groups),
        "fit-valid row-group membership differs from the frozen manifest",
    )
    _require(
        len(evaluation_groups) == EXPECTED_EVALUATION_GROUPS
        and evaluation_groups == set(split.evaluation_groups),
        "provenance evaluation groups differ from the frozen manifest",
    )
    _require(
        CORRUPT_TRAIN_GROUP not in train_groups,
        "corrupt train row group 528 entered the fitted set",
    )
    split.assert_fit_groups(train_groups, valid_groups)

    targets_path = ROOT / "cache/valid_targets.npz"
    with np.load(targets_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        targets = cache["y"]
        is_scored = cache["mask"].astype(bool, copy=False)
        market_class = cache["market_class"]
        expected_groups = evaluator._expected_target_groups(
            split.valid_groups, split.evaluation_groups
        )
        evaluator._validate_target_index(
            row_groups, targets, is_scored, market_class, expected_groups
        )
        sidecar_groups = metadata.get("row_groups")
        _require(
            sidecar_groups == row_groups, "prediction sidecar row order differs from target cache"
        )
        position = {group: index for index, group in enumerate(row_groups)}
        groups = list(map(int, split.evaluation_groups))
        if any(group not in position for group in groups):
            raise ValueError("frozen evaluation group is missing from prediction target cache")
        component_by_group = {
            int(group): int(component)
            for source, group, component in split.components
            if source == "valid"
        }
        component_membership = [
            {"row_group": group, "component": component_by_group[group]} for group in groups
        ]
        component_ids = sorted({item["component"] for item in component_membership})
        _require(
            len(component_ids) == EXPECTED_EVALUATION_COMPONENTS,
            "fold-0 evaluation component count changed",
        )
        indices = np.asarray([position[group] for group in groups], dtype=np.int64)
        mask = evaluator._official_mask(np.asarray(is_scored[indices], dtype=bool))
        prediction, binding = evaluator._member(
            ROOT,
            RUN,
            SEED,
            EPOCH,
            split,
            row_groups,
            expected_init=None,
        )
        accumulator = GlobalAccumulator()
        for start in range(0, len(indices), evaluator.SEQUENCES_PER_BATCH):
            stop = min(start + evaluator.SEQUENCES_PER_BATCH, len(indices))
            take = indices[start:stop]
            accumulator.add(
                np.asarray(targets[take], dtype=np.float32),
                np.asarray(prediction[take], dtype=np.float32),
                mask[start:stop],
            )

    score = accumulator.result()
    result: dict[str, object] = {
        "schema_version": 1,
        "grade_id": "E521_S3_STANDALONE_EXCLUDED_FOLD0_V1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_descriptive_grade",
        "claim": "one fixed E521 seed-3 scratch GRU-384 checkpoint scored standalone on the fold-0 groups excluded from its fit",
        "comparison": None,
        "public_forecast": None,
        "metric": {
            "id": METRIC_ID,
            "implementation": "src/wnn/gmetric.py:GlobalAccumulator",
            "score": score,
            "selection": "need_prediction AND is_scored",
            "warmup_rows_excluded": WARMUP,
            "prediction_and_target_clipping": [-2.0, 2.0],
            "weights": "absolute clipped label magnitude",
            "evaluation_aggregation": "pool all selected rows per target, then mean the two weighted Pearson correlations",
        },
        "evaluation": {
            "split_path": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "split_sha256": split.sha256,
            "exposure": "exposed development data; excluded from this fit but previously inspected and used for model/research decisions",
            "evaluation_group_count": len(groups),
            "connected_component_count": len(component_ids),
            "selected_scored_rows": int(mask.sum()),
            "row_groups": groups,
            "component_membership": component_membership,
            "target_cache_sha256": observed_hashes["cache/valid_targets.npz"],
            "target_row_groups_sha256": evaluator._list_sha256(row_groups),
        },
        "model": {
            "run": RUN,
            "seed": SEED,
            "checkpoint_epoch_zero_based": EPOCH,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "prediction_sha256": binding["prediction_sha256"],
            "prediction_metadata_sha256": observed_hashes[f"cache/pred_{RUN}_valid.json"],
            "fit_provenance_sha256": observed_hashes[f"artifacts/runs/{RUN}/provenance.json"],
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
            "inference_precision": binding.get("inference_precision"),
            "initialization": "scratch; no teacher or teammate weight ancestry",
        },
        "uncertainty_and_selection": {
            "training_seeds_scored": [SEED],
            "evaluation_interval": None,
            "reason": "single-model descriptive grade only; no matched comparator, training-seed estimate, or confirmation claim",
            "slice_or_blend_search": False,
            "further_selection_from_this_score": False,
        },
        "frozen_spec_path": spec_path.relative_to(ROOT).as_posix(),
        "frozen_spec_sha256": _sha256(spec_path),
        "artifact_hashes": observed_hashes,
        "integrity_checks": {
            "fit_and_export_terminal_success": True,
            "checkpoint_prediction_sidecar_binding": True,
            "fixed_epoch3_no_fit_evaluation": True,
            "full_fold_membership_and_component_disjointness": True,
            "corrupt_train_group_528_excluded": True,
            "only_frozen_evaluation_groups_scored": True,
        },
    }
    outputs = spec.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("grading spec lacks output paths")
    result_path = ROOT / str(outputs["result"])
    report_path = ROOT / str(outputs["report"])
    if result_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a previous E521 independent grade")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    return result


def _markdown_report(result: Mapping[str, object]) -> str:
    """Render the frozen standalone result with its claim limits."""
    metric = cast(dict[str, object], result["metric"])
    score = cast(dict[str, object], metric["score"])
    evaluation = cast(dict[str, object], result["evaluation"])
    model = cast(dict[str, object], result["model"])
    return "\n".join(
        [
            "# E521 seed-3 standalone fold-0 grade",
            "",
            f"Official metric: `{metric['id']}`; fixed epoch: {model['checkpoint_epoch_zero_based']}; seed: {model['seed']}.",
            f"Standalone score: **{float(score['wp']):.8f}** (T0 {float(score['t0']):.8f}; T1 {float(score['t1']):.8f}).",
            f"Evaluation: {evaluation['evaluation_group_count']} fold-0 sequences across {evaluation['connected_component_count']} connected components; {evaluation['selected_scored_rows']:,} selected rows.",
            "",
            "This is one fixed model scored on groups excluded from its fit. Fold 0 is exposed development data, not fresh confirmation. The result has no matched GRU-128 control and does not estimate a width effect, training-seed variability, ensemble value, or public transfer. No public forecast is made.",
            "",
            f"Checkpoint SHA-256: `{model['checkpoint_sha256']}`.",
            f"Prediction SHA-256: `{model['prediction_sha256']}`.",
            f"Frozen grading spec SHA-256: `{result['frozen_spec_sha256']}`.",
            "",
        ]
    )


def main() -> None:
    """Run the single frozen grade from the preregistered specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args()
    result = grade(args.spec)
    metric = cast(dict[str, object], result["metric"])
    print(json.dumps(metric["score"], sort_keys=True))


if __name__ == "__main__":
    main()
