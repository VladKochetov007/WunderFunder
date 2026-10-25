"""Select one shared R5 checkpoint horizon from paired F0/F2 inner-fold sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epoch_contract import (
    EXPECTED_EPOCHS,
    canonical_sha256,
    file_sha256,
    read_object,
    validate_epoch_index,
    validate_split_pair,
    write_json_new,
)
from epoch_sweep_receipts import load_epoch_record

SWEEP_SCHEMA = "wnn.recovery-r5-epoch-sweep.v1"
SELECTION_SCHEMA = "wnn.recovery-r5-epoch-selection.v1"


def _load_fold(
    output: Path, expected_fold: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load and revalidate one complete sweep, retaining any verified partial curve."""
    output = output.expanduser().resolve()
    missing: list[dict[str, Any]] = []
    if not output.is_dir() or output.is_symlink():
        return None, [{"fold": expected_fold, "artifact": "plan.json"}]
    plan_path = output / "plan.json"
    if not plan_path.is_file():
        return None, [{"fold": expected_fold, "artifact": "plan.json"}]
    plan, _ = read_object(plan_path)
    if plan.get("schema") != SWEEP_SCHEMA or plan.get("fold") != expected_fold:
        raise ValueError("sweep plan belongs to a different protocol or fold")
    split_binding = validate_split_pair(
        Path(plan["outer_split"]), Path(plan["inner_split"]), expected_fold
    )
    if (
        plan.get("outer_split_sha256") != split_binding["outer_sha256"]
        or plan.get("inner_split_sha256") != split_binding["inner_sha256"]
    ):
        raise ValueError("sweep split manifest changed after evaluation")
    fit = validate_epoch_index(
        Path(plan["fit_dir"]),
        Path(plan["epoch_index"]),
        str(plan["epoch_index_sha256"]),
        Path(plan["config_path"]),
        split_binding,
    )
    for key, value in (
        ("config_file_sha256", fit["config_file_sha256"]),
        ("config_recipe_sha256", fit["config_recipe_sha256"]),
        ("config_identity_sha256", fit["identity"]["config_sha256"]),
        ("epoch_index_file_sha256", fit["index_file_sha256"]),
        ("final_model_sha256", fit["final_model_sha256"]),
    ):
        if plan.get(key) != value:
            raise ValueError(f"sweep plan {key} binding changed")
    if plan.get("identity") != fit["identity"] or plan.get("expected_epochs") != EXPECTED_EPOCHS:
        raise ValueError("sweep plan fit identity or expected horizon changed")

    observed: list[dict[str, Any]] = []
    for completed in range(1, EXPECTED_EPOCHS + 1):
        directory = output / "epochs" / f"epoch-{completed:06d}"
        receipt_path = directory / "epoch.json"
        if not receipt_path.is_file():
            missing.append(
                {"fold": expected_fold, "completed_epochs": completed, "artifact": "epoch.json"}
            )
            continue
        prediction_path = directory / "predictions" / "manifest.json"
        score_path = directory / "score.json"
        if not prediction_path.is_file() or not score_path.is_file():
            missing.append(
                {
                    "fold": expected_fold,
                    "completed_epochs": completed,
                    "artifact": "prediction-or-score",
                }
            )
            continue
        observed.append(
            load_epoch_record(
                output, fit["entries"][completed - 1], expected_fold, fit, split_binding
            )
        )

    sweep_path = output / "sweep.json"
    if not sweep_path.is_file():
        missing.append({"fold": expected_fold, "artifact": "sweep.json"})
    else:
        sweep, _ = read_object(sweep_path)
        unsigned = {key: value for key, value in sweep.items() if key != "sweep_sha256"}
        if (
            sweep.get("schema") != SWEEP_SCHEMA
            or sweep.get("status") != "complete"
            or sweep.get("fold") != expected_fold
            or sweep.get("expected_epochs") != EXPECTED_EPOCHS
            or sweep.get("sweep_sha256") != canonical_sha256(unsigned)
            or sweep.get("plan_sha256") != file_sha256(plan_path)
            or sweep.get("epoch_index_sha256") != fit["index_sha256"]
            or sweep.get("config_identity_sha256") != fit["identity"]["config_sha256"]
            or sweep.get("outer_split_sha256") != split_binding["outer_sha256"]
            or sweep.get("inner_split_sha256") != split_binding["inner_sha256"]
            or len(sweep.get("epochs", [])) != EXPECTED_EPOCHS
        ):
            raise ValueError(f"completed F{expected_fold} sweep manifest is malformed")
        listed = sweep["epochs"]
        if [item.get("completed_epochs") for item in listed] != list(range(1, EXPECTED_EPOCHS + 1)):
            raise ValueError("sweep does not cover every one-based completed epoch in order")
        if len(observed) == EXPECTED_EPOCHS and any(
            listed[index].get("receipt_sha256") != observed[index]["receipt_sha256"]
            or listed[index].get("score_sha256") != observed[index]["score_sha256"]
            for index in range(EXPECTED_EPOCHS)
        ):
            raise ValueError("sweep manifest detached from its per-epoch receipts")
    return {
        "output": str(output),
        "plan": plan,
        "plan_sha256": file_sha256(plan_path),
        "sweep_sha256": file_sha256(sweep_path) if sweep_path.is_file() else None,
        "sweep": sweep if sweep_path.is_file() else None,
        "observed": observed,
        "missing": missing,
        "split_binding": split_binding,
        "fit": fit,
    }, missing


def _shared_recipe(config: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(config)
    recipe.pop("seed", None)
    recipe.pop("expected_fold", None)
    return recipe


def _fold_pin(fold: dict[str, Any], epoch_position: int | None = None) -> dict[str, Any]:
    plan, fit = fold["plan"], fold["fit"]
    result = {
        "fold": plan["fold"],
        "config_file_sha256": plan["config_file_sha256"],
        "config_recipe_sha256": plan["config_recipe_sha256"],
        "config_identity_sha256": plan["config_identity_sha256"],
        "outer_split_sha256": plan["outer_split_sha256"],
        "inner_split_sha256": plan["inner_split_sha256"],
        "epoch_index_sha256": plan["epoch_index_sha256"],
        "epoch_index_file_sha256": plan["epoch_index_file_sha256"],
        "sweep_manifest_sha256": fold["sweep_sha256"],
        "final_model_sha256": plan["final_model_sha256"],
        "all_epoch_score_hashes": [item["score_sha256"] for item in fold["observed"]],
    }
    if epoch_position is not None:
        selected = fold["observed"][epoch_position]
        result["selected_model"] = {
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "score_sha256": selected["score_sha256"],
            "prediction_manifest_sha256": selected["prediction_manifest_sha256"],
            "score": selected["score"],
        }
        result["config_identity"] = fit["identity"]["config_sha256"]
    return result


def select_shared_epoch(f0_output: Path, f2_output: Path, output: Path) -> dict[str, Any]:
    """Select the earliest epoch maximizing equal-fold mean inner WP, or publish HOLD."""
    f0, missing0 = _load_fold(f0_output, 0)
    f2, missing2 = _load_fold(f2_output, 2)
    missing = missing0 + missing2
    if f0 is not None and f2 is not None:
        for fold in (f0, f2):
            if fold["missing"]:
                missing.extend(fold["missing"])
        if f0["plan"]["config_recipe_sha256"] != f2["plan"]["config_recipe_sha256"]:
            c0 = _shared_recipe(f0["fit"]["config"])
            c2 = _shared_recipe(f2["fit"]["config"])
            if c0 != c2:
                raise ValueError("F0/F2 recovery recipes differ beyond seed and expected fold")
        if f0["fit"]["identity"].get("source_sha256") != f2["fit"]["identity"].get("source_sha256"):
            raise ValueError("F0/F2 source code identities differ")
        runtime0 = f0["fit"]["identity"].get("runtime")
        runtime2 = f2["fit"]["identity"].get("runtime")
        if not isinstance(runtime0, dict) or not isinstance(runtime2, dict):
            raise ValueError("F0/F2 runtime precision provenance is missing")
        if (
            runtime0.get("precision") != runtime2.get("precision")
            or runtime0.get("compiler") != runtime2.get("compiler")
            or runtime0.get("attention_backend") != runtime2.get("attention_backend")
        ):
            raise ValueError("F0/F2 precision/compiler execution contract differs")
    if (
        missing
        or f0 is None
        or f2 is None
        or len(f0["observed"]) != EXPECTED_EPOCHS
        or len(f2["observed"]) != EXPECTED_EPOCHS
    ):
        summary = {
            "schema": SELECTION_SCHEMA,
            "status": "incomplete",
            "selection_rule": "max equal-fold mean inner WP; exact ties choose the earlier completed epoch",
            "outer_feedback": "none; outer splits are ancestry-only and never scored",
            "expected_completed_epochs": list(range(1, EXPECTED_EPOCHS + 1)),
            "missing": missing,
            "available_scores": {
                "F0": [] if f0 is None else f0["observed"],
                "F2": [] if f2 is None else f2["observed"],
            },
            "decision": None,
        }
        summary["selection_sha256"] = canonical_sha256(summary)
        write_json_new(output, summary)
        return summary

    config0 = f0["fit"]["config"]
    config2 = f2["fit"]["config"]
    if _shared_recipe(config0) != _shared_recipe(config2):
        raise ValueError("F0/F2 recovery recipes differ beyond seed and expected fold")
    if config0.get("seed") == config2.get("seed"):
        raise ValueError("F0/F2 must use distinct training seeds")
    means = [
        (float(f0["observed"][index]["score"]["wp"]) + float(f2["observed"][index]["score"]["wp"]))
        / 2.0
        for index in range(EXPECTED_EPOCHS)
    ]
    selected_position = max(range(EXPECTED_EPOCHS), key=lambda index: (means[index], -index))
    selected_completed = selected_position + 1
    unsigned = {
        "schema": SELECTION_SCHEMA,
        "status": "selected",
        "selection_rule": "max equal-fold mean inner WP; exact ties choose the earlier completed epoch",
        "outer_feedback": "none; outer splits are ancestry-only and never scored",
        "metric": "official pooled weighted Pearson score.wp from score_sequence.py",
        "selected_epoch": selected_completed - 1,
        "selected_completed_epochs": selected_completed,
        "selected_mean_inner_wp": means[selected_position],
        "mean_inner_wp_by_completed_epoch": means,
        "shared_recipe_sha256": canonical_sha256(_shared_recipe(config0)),
        "folds": {
            "F0": _fold_pin(f0, selected_position),
            "F2": _fold_pin(f2, selected_position),
        },
        "curve_score_hashes": {
            "F0": [item["score_sha256"] for item in f0["observed"]],
            "F2": [item["score_sha256"] for item in f2["observed"]],
        },
    }
    decision = unsigned | {"selection_sha256": canonical_sha256(unsigned)}
    write_json_new(output, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-output", type=Path, required=True)
    parser.add_argument("--f2-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_shared_epoch(args.f0_output, args.f2_output, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
