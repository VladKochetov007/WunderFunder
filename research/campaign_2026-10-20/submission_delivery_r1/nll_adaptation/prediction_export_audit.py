"""Validate exported checkpoint and prediction identities for the frozen screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prediction_artifact_validation import GROUPS, NBYTES, SHAPE, _json, _prediction, _sha


def _export(
    root: Path,
    item: dict[str, Any],
    split: dict[str, Any],
    components: dict[str, dict[int, int]],
    split_sha: str,
    *,
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
