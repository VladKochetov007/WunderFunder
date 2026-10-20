"""Validate exported checkpoint and prediction identities for the frozen screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prediction_artifact_validation import GROUPS, NBYTES, SHAPE, _json, _prediction, _sha

_BOOLEAN_OPTIONS = {
    "--no-eval-during-fit",
    "--inherit-init-preprocessing",
    "--grad-stats",
    "--fast",
    "--save-training-state",
}


def _fit_args_by_name(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse each frozen fit argv into the exact provenance arguments expected."""
    result = {}
    for arm in recipe["design"]["arms"]:
        argv = arm["fit_argv"]
        if not isinstance(argv, list) or len(argv) < 2:
            raise ValueError("recipe arm has no executable fit argv")
        expected: dict[str, Any] = {}
        cursor = 1
        while cursor < len(argv):
            option = argv[cursor]
            if not isinstance(option, str) or not option.startswith("--"):
                raise ValueError(f"invalid fit option in recipe: {option!r}")
            key = option[2:].replace("-", "_")
            if key in expected:
                raise ValueError(f"duplicate fit option in recipe: {option}")
            if option in _BOOLEAN_OPTIONS:
                expected[key] = True
                cursor += 1
                continue
            if cursor + 1 >= len(argv) or str(argv[cursor + 1]).startswith("--"):
                raise ValueError(f"fit option has no value in recipe: {option}")
            expected[key] = argv[cursor + 1]
            cursor += 2
        name = expected.get("name")
        if not isinstance(name, str) or name in result:
            raise ValueError("recipe fit names must be unique strings")
        if expected.get("mean_variance_objective") != arm["objective"]:
            raise ValueError(f"{name} objective differs from its recipe arm")
        result[name] = expected
    return result


def _args_match(actual: Any, expected: Any) -> bool:
    """Compare string argv values with typed provenance values."""
    if isinstance(actual, bool):
        return expected is True and actual is True
    if isinstance(actual, int):
        try:
            return actual == int(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, float):
        try:
            return actual == float(expected)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _export(
    root: Path,
    item: dict[str, Any],
    split: dict[str, Any],
    components: dict[str, dict[int, int]],
    split_sha: str,
    *,
    expected_args: dict[str, Any] | None = None,
    expected_ancestry: dict[str, Any] | None = None,
    check_predictions: bool = True,
) -> dict[str, Any]:
    """Bind a completed checkpoint and provenance, then optionally its export."""
    run, epoch = item["run"], int(item["epoch"])
    run_root = root / "artifacts" / "runs" / run
    checkpoint, prov_path = run_root / f"model_ep{epoch}.pt", run_root / "provenance.json"
    checkpoint_sha, provenance = _sha(checkpoint), _json(prov_path)
    pinned_sha = item.get("checkpoint_sha256")
    args = provenance.get("args", {})
    if (pinned_sha and checkpoint_sha != pinned_sha) or not isinstance(args, dict):
        raise ValueError(f"checkpoint identity mismatch: {run}")
    if expected_args is not None:
        mismatches = [
            key
            for key, expected in expected_args.items()
            if key not in args or not _args_match(args[key], expected)
        ]
        if mismatches:
            raise ValueError(f"{run} provenance differs from frozen fit argv: {mismatches}")
    if expected_ancestry is not None:
        cache = provenance.get("distillation_cache", {})
        if not isinstance(cache, dict) or expected_args is None:
            raise ValueError(f"{run} provenance is missing frozen cache ancestry")
        cache_expected = expected_ancestry["distillation_cache"]
        if (
            provenance.get("initial_checkpoint_sha256")
            != expected_ancestry["initial_checkpoint_sha256"]
            or provenance.get("propensity_checkpoint_sha256")
            != expected_ancestry["propensity_checkpoint_sha256"]
            or cache.get("root") != expected_args.get("distill_cache_root")
            or cache.get("variant") != expected_args.get("distill_cache_variant")
            or cache.get("main_target_scale") != float(expected_args["distill_cache_scale"])
            or cache.get("main_target_scale_applied_before_training_clip") is not True
            or cache.get("metadata_sha256") != cache_expected["metadata_sha256"]
            or cache.get("verified_files") != cache_expected["verified_files"]
        ):
            raise ValueError(f"{run} provenance ancestry differs from pinned parent/cache inputs")
    if (
        args.get("name") != run
        or args.get("valid_fold") != 0
        or args.get("split_manifest_sha256") != split_sha
        or args.get("checkpoint_epoch") != epoch
        or args.get("checkpoint_policy") != "fixed"
        or args.get("no_eval_during_fit") is not True
        or provenance.get("split_manifest_sha256") != split_sha
        or provenance.get("checkpoint_sha256", {}).get(checkpoint.name) != checkpoint_sha
    ):
        raise ValueError(f"checkpoint provenance mismatch: {run}")
    members = provenance.get("fit_row_groups", {})
    for source in ("train", "valid"):
        actual, expected = (
            list(map(int, members.get(source, []))),
            list(map(int, split["fit"][source])),
        )
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(f"{run} fitting membership differs from fold0")
    eval_rows = set(map(int, split["evaluation"]["valid"]))
    if set(map(int, provenance.get("evaluation_row_groups", []))) != eval_rows:
        raise ValueError(f"{run} evaluation membership differs from fold0")
    eval_components = {components["valid"][g] for g in eval_rows}
    fit_components = {components[s][int(g)] for s in ("train", "valid") for g in members[s]}
    if fit_components.intersection(eval_components):
        raise ValueError(f"{run} fitting components overlap evaluation")
    objective = {"wmse_cp1": "wmse", "nll_cp1": "nll"}.get(item["id"])
    if objective and (
        args.get("precision") != "fp32"
        or args.get("loss") != "wmse"
        or args.get("mean_variance_head") != 1
        or args.get("mean_variance_objective") != objective
        or args.get("init") != "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5"
    ):
        raise ValueError(f"{run} differs from the frozen {objective} continuation")
    if item["id"] == "oldz4_cp1_fp32" and (
        args.get("precision") != "fp16"
        or args.get("seed") != 3
        or args.get("train_source") != "train+valid"
        or args.get("init") != "oldz4_h230_parent_f0_s3_r3@ep3"
    ):
        raise ValueError("old Z4 checkpoint differs from its reviewed lineage")
    checkpoint_record = {"path": str(checkpoint), "sha256": checkpoint_sha}
    provenance_record = {"path": str(prov_path), "sha256": _sha(prov_path)}
    fit_counts = {s: len(members[s]) for s in ("train", "valid")}
    if not check_predictions:
        return {
            "id": item["id"],
            "run": run,
            "epoch": epoch,
            "checkpoint": checkpoint_record,
            "provenance": provenance_record,
            "fit_args": expected_args,
            "lineage_pins": expected_ancestry,
            "fit_group_counts": fit_counts,
            "fit_components_exclude_evaluation": True,
        }
    pred = Path(item["prediction_path"])
    side_path, side = pred.with_suffix(".json"), _json(pred.with_suffix(".json"))
    prediction_sha = _prediction(pred, scan=True)
    if (
        side.get("run") != run
        or side.get("split") != "valid"
        or side.get("epoch") != epoch
        or side.get("checkpoint_sha256") != checkpoint_sha
        or side.get("prediction_sha256") != prediction_sha
        or side.get("split_manifest_sha256") != split_sha
        or side.get("campaign_split_manifest_sha256") != split_sha
        or side.get("inference_precision") != "fp32"
        or side.get("feature_only") is not True
        or side.get("campaign_export_scope") != "full-valid-feature-only"
        or side.get("row_groups") != GROUPS
    ):
        raise ValueError(f"prediction sidecar mismatch: {side_path}")
    return {
        "id": item["id"],
        "run": run,
        "epoch": epoch,
        "checkpoint": checkpoint_record,
        "provenance": provenance_record,
        "fit_args": expected_args,
        "lineage_pins": expected_ancestry,
        "fit_group_counts": fit_counts,
        "fit_components_exclude_evaluation": True,
        "sidecar": {"path": str(side_path), "sha256": _sha(side_path)},
        "prediction": {
            "path": str(pred),
            "bytes": NBYTES,
            "sha256": prediction_sha,
            "dtype": "float32",
            "shape": list(SHAPE),
            "all_finite_sequence_by_sequence": True,
            "row_groups": [0, 1872, len(GROUPS)],
        },
    }
