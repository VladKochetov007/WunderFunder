"""Build a hash-bound post-selection checkpoint-retention plan for one R5 LR pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import (
    EXPECTED_EPOCHS,
    canonical_sha256,
    file_sha256,
    read_object,
    safe_file,
    write_json_new,
)
from epoch_selection import _load_fold, selection_from_folds
from retention_inventory import _tree_artifacts, _validate_prediction_inventory

PLAN_SCHEMA = "wnn.recovery-r5-retention-plan.v1"
SELECTION_SCHEMA = "wnn.recovery-r5-epoch-selection.v1"


def _implementation_hashes() -> dict[str, str]:
    """Bind the validators and split contract used to authorize compaction."""
    packet = Path(__file__).resolve().parent
    paths = {
        "epoch_contract.py": packet / "epoch_contract.py",
        "epoch_selection.py": packet / "epoch_selection.py",
        "epoch_sweep_receipts.py": packet / "epoch_sweep_receipts.py",
        "retention_plan.py": packet / "retention_plan.py",
        "retention_apply.py": packet / "retention_apply.py",
        "retention_contract.py": packet / "retention_contract.py",
        "retention_verify.py": packet / "retention_verify.py",
        "retention_receipts.py": packet / "retention_receipts.py",
        "retention_inventory.py": packet / "retention_inventory.py",
        "split_inputs.py": packet.parent / "high_compute_r4" / "split_inputs.py",
    }
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("retention implementation sources are missing or unsafe")
    return {name: file_sha256(path) for name, path in paths.items()}


def _checkpoint_inventory(fit_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Require exactly the complete, indexed snapshot set before selection cleanup."""
    directory = fit_dir / "epoch_checkpoints"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("epoch checkpoint directory is missing or unsafe")
    expected = {entry["checkpoint_path"] for entry in entries}
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
            raise ValueError("epoch checkpoint directory contains an unsafe or partial file")
        observed.add(path.relative_to(fit_dir).as_posix())
    if observed != expected:
        raise ValueError("epoch checkpoint directory differs from the complete index")


def _verified_selection(
    f0_output: Path, f2_output: Path, selection_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Recompute the paired selector from complete sweeps and require exact equality."""
    selection_path = selection_path.expanduser().resolve(strict=True)
    selection, raw = read_object(selection_path)
    selection_sha = hashlib.sha256(raw).hexdigest()
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("status") != "selected"
        or selection.get("selection_sha256") != canonical_sha256(unsigned)
        or type(selection.get("selected_completed_epochs")) is not int
        or selection.get("selected_completed_epochs") not in range(1, EXPECTED_EPOCHS + 1)
    ):
        raise ValueError("retention requires a valid completed shared-epoch decision")
    f0, missing0 = _load_fold(f0_output, 0)
    f2, missing2 = _load_fold(f2_output, 2)
    if f0 is None or f2 is None:
        raise ValueError("retention requires both complete fit/sweep pairs")
    recomputed = selection_from_folds(f0, f2, missing0, missing2)
    if recomputed != selection:
        raise ValueError("selection differs from the exact recomputed full-curve decision")
    return selection, selection_sha, f0, f2


def _fold_plan(fold_data: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Bind one complete fit, all curve receipts, and its exact deletion set."""
    fold = int(fold_data["plan"]["fold"])
    key = f"F{fold}"
    fit_dir = Path(fold_data["fit"]["fit_dir"]).resolve(strict=True)
    sweep_dir = Path(fold_data["output"]).resolve(strict=True)
    selected = selection["folds"][key]["selected_model"]
    entries = fold_data["fit"]["entries"]
    _checkpoint_inventory(fit_dir, entries)
    matches = [
        entry
        for entry in entries
        if entry["checkpoint_path"] == selected["checkpoint_path"]
        and entry["checkpoint_sha256"] == selected["checkpoint_sha256"]
        and entry["completed_epochs"] == selection["selected_completed_epochs"]
    ]
    if len(matches) != 1:
        raise ValueError(f"selected {key} checkpoint is not a unique indexed epoch")
    selected_path = safe_file(fit_dir, selected["checkpoint_path"])
    selected_checkpoint = {
        "path": selected["checkpoint_path"],
        "sha256": file_sha256(selected_path),
        "size": selected_path.stat().st_size,
    }
    removed = []
    for entry in entries:
        if entry["checkpoint_path"] == selected_checkpoint["path"]:
            continue
        path = safe_file(fit_dir, entry["checkpoint_path"])
        removed.append(
            {
                "path": entry["checkpoint_path"],
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
        )

    fit_artifacts = _tree_artifacts(fit_dir, fit=True)
    fit_paths = {item["path"] for item in fit_artifacts}
    if (
        not {"epoch_index.json", "model.pt", "training_state.pt", "terminal_status.json"}
        <= fit_paths
    ):
        raise ValueError(f"{key} fit is missing required retained state or metadata")
    sweep_artifacts = _tree_artifacts(sweep_dir, fit=False)
    curve = [
        {
            "completed_epochs": item["completed_epochs"],
            "checkpoint_path": item["checkpoint_path"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "score_sha256": item["score_sha256"],
            "prediction_manifest_sha256": item["prediction_manifest_sha256"],
            "receipt_sha256": item["receipt_sha256"],
            "score": item["score"],
        }
        for item in fold_data["observed"]
    ]
    if [item["completed_epochs"] for item in curve] != list(range(1, EXPECTED_EPOCHS + 1)):
        raise ValueError(f"{key} sweep does not contain all 30 epochs")
    for item in curve:
        prediction_path = safe_file(
            sweep_dir,
            f"epochs/epoch-{item['completed_epochs']:06d}/predictions/manifest.json",
        )
        manifest, _ = read_object(prediction_path)
        _validate_prediction_inventory(sweep_dir, item["completed_epochs"], manifest)
    terminal, _ = read_object(safe_file(fit_dir, "terminal_status.json"))
    if terminal.get("status") != "completed" or terminal.get("completed_epochs") != EXPECTED_EPOCHS:
        raise ValueError(f"{key} fit is not a completed 30-epoch endpoint")
    return {
        "fold": fold,
        "fit_dir": str(fit_dir),
        "sweep_dir": str(sweep_dir),
        "identity": fold_data["fit"]["identity"],
        "config_path": fold_data["plan"]["config_path"],
        "config_file_sha256": fold_data["fit"]["config_file_sha256"],
        "config_recipe_sha256": fold_data["fit"]["config_recipe_sha256"],
        "outer_split": fold_data["plan"]["outer_split"],
        "outer_split_sha256": fold_data["split_binding"]["outer_sha256"],
        "inner_split": fold_data["plan"]["inner_split"],
        "inner_split_sha256": fold_data["split_binding"]["inner_sha256"],
        "sweep_plan_sha256": file_sha256(sweep_dir / "plan.json"),
        "sweep_sha256": file_sha256(sweep_dir / "sweep.json"),
        "index_sha256": fold_data["fit"]["index_sha256"],
        "index_file_sha256": fold_data["fit"]["index_file_sha256"],
        "index_entries": fold_data["fit"]["entries"],
        "final_model_sha256": fold_data["fit"]["final_model_sha256"],
        "selected_completed_epochs": selection["selected_completed_epochs"],
        "selected_checkpoint": selected_checkpoint,
        "remove": sorted(removed, key=lambda item: item["path"]),
        "curve": curve,
        "fit_artifacts": fit_artifacts,
        "sweep_artifacts": sweep_artifacts,
    }


def create_retention_plan(
    f0_output: Path, f2_output: Path, selection_path: Path, output: Path
) -> tuple[dict[str, Any], str]:
    """Publish a no-overwrite plan after exact full-curve and pair-selection validation."""
    f0_output = f0_output.expanduser().resolve(strict=True)
    f2_output = f2_output.expanduser().resolve(strict=True)
    selection_path = selection_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    selection, selection_file_sha256, f0, f2 = _verified_selection(
        f0_output, f2_output, selection_path
    )
    folds = []
    for fold in (f0, f2):
        if fold["missing"] or fold["sweep"] is None:
            raise ValueError(
                f"F{fold['plan']['fold']} is not a complete fully verified epoch sweep"
            )
        folds.append(_fold_plan(fold, selection))
    fit_dirs = [Path(item["fit_dir"]) for item in folds]
    sweep_dirs = [Path(item["sweep_dir"]) for item in folds]
    if fit_dirs[0] == fit_dirs[1] or sweep_dirs[0] == sweep_dirs[1]:
        raise ValueError("retention requires distinct F0 and F2 fit/sweep directories")
    for protected in (*fit_dirs, *sweep_dirs, selection_path):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("retention plan output must be outside fits, sweeps, and selection")
    unsigned = {
        "schema": PLAN_SCHEMA,
        "status": "ready",
        "selection_path": str(selection_path),
        "selection_file_sha256": selection_file_sha256,
        "selection": selection,
        "selected_completed_epochs": selection["selected_completed_epochs"],
        "selection_rule": selection["selection_rule"],
        "implementation_sha256": _implementation_hashes(),
        "folds": folds,
    }
    plan = unsigned | {"plan_sha256": canonical_sha256(unsigned)}
    raw_sha256 = write_json_new(output, plan)
    return plan, raw_sha256


def main() -> None:
    """Create a reviewed retention plan without deleting any checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-output", type=Path, required=True)
    parser.add_argument("--f2-output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, raw_sha256 = create_retention_plan(
        args.f0_output, args.f2_output, args.selection, args.output
    )
    print(
        json.dumps(
            {
                "plan": str(args.output.resolve()),
                "plan_sha256": raw_sha256,
                "folds": [item["fold"] for item in plan["folds"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
