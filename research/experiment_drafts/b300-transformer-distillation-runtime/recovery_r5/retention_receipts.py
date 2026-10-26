"""Validate paired R5 retention intent and completion receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import canonical_sha256, file_sha256, read_object
from retention_contract import (
    COMPLETE_SCHEMA,
    INTENT_SCHEMA,
    _load_plan,
    _plan_path,
)
from retention_verify import _verify_plan_state, _verify_removed_absent


def _intent(
    plan: dict[str, Any], raw_sha256: str, fold: dict[str, Any], plan_path: Path
) -> dict[str, Any]:
    """Build the stable authorization receipt that must precede deletion."""
    return {
        "schema": INTENT_SCHEMA,
        "status": "delete_authorized",
        "plan_path": str(plan_path),
        "plan_file_sha256": raw_sha256,
        "plan_sha256": plan["plan_sha256"],
        "fold": fold["fold"],
        "fit_dir": fold["fit_dir"],
        "selection_path": plan["selection_path"],
        "selection_file_sha256": plan["selection_file_sha256"],
        "selected_completed_epochs": plan["selected_completed_epochs"],
        "selected_checkpoint": fold["selected_checkpoint"],
        "remove": fold["remove"],
    }


def _complete(
    plan: dict[str, Any], raw_sha256: str, fold: dict[str, Any], plan_path: Path
) -> dict[str, Any]:
    """Build the post-delete receipt binding retained assets and deleted hashes."""
    intent_path = Path(fold["fit_dir"]) / "retention_intent.json"
    return {
        "schema": COMPLETE_SCHEMA,
        "status": "completed",
        "plan_path": str(plan_path),
        "plan_file_sha256": raw_sha256,
        "plan_sha256": plan["plan_sha256"],
        "intent_file_sha256": file_sha256(intent_path),
        "fold": fold["fold"],
        "fit_dir": fold["fit_dir"],
        "selected_completed_epochs": plan["selected_completed_epochs"],
        "selected_checkpoint": fold["selected_checkpoint"],
        "removed": fold["remove"],
        "retained_fit_artifacts_sha256": canonical_sha256(fold["fit_artifacts"]),
    }


def _verify_receipts(plan: dict[str, Any], raw_sha256: str, plan_path: Path) -> tuple[int, int]:
    """Check every existing intent and completion receipt against this plan."""
    intents, completions = 0, 0
    for fold in plan["folds"]:
        root = Path(fold["fit_dir"])
        intent_path = root / "retention_intent.json"
        complete_path = root / "retention_complete.json"
        if intent_path.is_symlink() or complete_path.is_symlink():
            raise ValueError("retention receipts must not be symlinks")
        if intent_path.exists():
            if not intent_path.is_file():
                raise ValueError("retention intent must be a regular file")
            intent, _ = read_object(intent_path)
            if intent != _intent(plan, raw_sha256, fold, plan_path):
                raise ValueError("retention intent is detached from the reviewed plan")
            intents += 1
        if complete_path.exists():
            if not intent_path.is_file() or not complete_path.is_file():
                raise ValueError("retention completion has no regular durable intent")
            complete, _ = read_object(complete_path)
            if complete != _complete(plan, raw_sha256, fold, plan_path):
                raise ValueError("retention completion receipt differs from verified plan state")
            completions += 1
    return intents, completions


def verify_retention(fit_dir: Path) -> dict[str, Any]:
    """Verify pair cleanup state for a fit, including retained hashes and plan ancestry."""
    if fit_dir.is_symlink():
        raise ValueError("fit directory must not be a symlink")
    fit_dir = fit_dir.expanduser().resolve(strict=True)
    intent_path = fit_dir / "retention_intent.json"
    complete_path = fit_dir / "retention_complete.json"
    if intent_path.is_symlink() or complete_path.is_symlink():
        raise ValueError("retention receipts must not be symlinks")
    if not intent_path.exists() and not complete_path.exists():
        return {"status": "not_pruned", "fit_dir": str(fit_dir)}
    if not intent_path.is_file():
        raise ValueError("retention completion exists without a regular intent")
    intent, _ = read_object(intent_path)
    plan_path = _plan_path(intent.get("plan_path", ""))
    plan, raw_sha256 = _load_plan(plan_path, str(intent.get("plan_file_sha256", "")))
    intents, completions = _verify_receipts(plan, raw_sha256, plan_path)
    allow_missing = intents == len(plan["folds"])
    _verify_plan_state(plan, allow_missing=allow_missing)
    if completions == len(plan["folds"]):
        _verify_removed_absent(plan)
    if fit_dir not in {Path(item["fit_dir"]).resolve(strict=True) for item in plan["folds"]}:
        raise ValueError("fit directory is not a member of the retention plan")
    status = "completed" if completions == len(plan["folds"]) else "incomplete"
    return {"status": status, "fit_dir": str(fit_dir), "plan_sha256": raw_sha256}


def verify_pair_retention(plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    """Require complete matching receipts and validated bytes for both folds."""
    if plan_path.is_symlink():
        raise ValueError("retention plan must not be a symlink")
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan, raw_sha256 = _load_plan(plan_path, plan_sha256)
    intents, completions = _verify_receipts(plan, raw_sha256, plan_path)
    if intents != 2 or completions != 2:
        raise ValueError("both folds must have completed matching retention receipts")
    _verify_plan_state(plan, allow_missing=True)
    _verify_removed_absent(plan)
    statuses = [
        {"status": "completed", "fit_dir": fold["fit_dir"], "plan_sha256": raw_sha256}
        for fold in plan["folds"]
    ]
    return {"status": "verified", "plan_sha256": raw_sha256, "folds": statuses}
