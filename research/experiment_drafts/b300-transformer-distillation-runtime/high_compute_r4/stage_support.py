"""Load frozen parent selections and bind downstream stage dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import load_campaign, read_object, require, sha256_file
from cell_materialize import _relative


def _stage_manifest(campaign_root: Path, stage: str) -> dict[str, Any]:
    """Read a stage manifest after validating its raw-byte hash sidecar."""
    path = campaign_root / "stages" / stage / "stage.json"
    value = read_object(path)
    require(
        path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == sha256_file(path),
        f"stage manifest hash mismatch: {stage}",
    )
    return value


def _read_selection(campaign_root: Path, stage: str) -> tuple[dict[str, Any], str, str]:
    """Return a parent selection plus both its file and parent-stage hashes."""
    selection_path = campaign_root / "stages" / stage / "selection.json"
    selection = read_object(selection_path)
    return selection, _relative(campaign_root, selection_path), sha256_file(selection_path)


def _selected_cells(
    campaign_root: Path, stage: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate the parent selection and resolve its explicit two-fold pair."""
    from results import validate_selection_receipt

    campaign = load_campaign(campaign_root)
    selection, _, _ = _read_selection(campaign_root, stage)
    validate_selection_receipt(campaign_root, campaign, selection)
    parent_stage = _stage_manifest(campaign_root, stage)
    selected: dict[int, dict[str, Any]] = {}
    for item in selection["selected_pair"]:
        fold = int(item["fold"])
        require(
            fold in (0, 2) and fold not in selected,
            "parent selection must name one cell per outer fold",
        )
        record = next(
            cell
            for cell in parent_stage["cells"]
            if cell["id"] == item["id"] and cell["variant"] == selection["variant"]
        )
        selected[fold] = record
    require(set(selected) == {0, 2}, "parent selection is missing an F0/F2 endpoint")
    return selection, selected


def _dependencies(campaign_root: Path, parent_stage: str) -> dict[str, Any]:
    """Bind a child stage to its selected parent stage and endpoint evidence."""
    parent_manifest = campaign_root / "stages" / parent_stage / "stage.json"
    selection, selection_path, selection_hash = _read_selection(campaign_root, parent_stage)
    return {
        "parent_stage": parent_stage,
        "stage_manifest": {
            "path": _relative(campaign_root, parent_manifest),
            "sha256": sha256_file(parent_manifest),
        },
        "selection": {"path": selection_path, "sha256": selection_hash},
        "selected_variant": selection["variant"],
    }


def _load_selected_parent(
    campaign_root: Path, stage: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    """Load one selected recipe and its dependency hashes for a downstream stage."""
    previous = {"C": "B", "D": "C", "E": "D", "F": "E", "G": "F", "H": "F"}[stage]
    selection, cells = _selected_cells(campaign_root, previous)
    return selection, cells, _dependencies(campaign_root, previous)
