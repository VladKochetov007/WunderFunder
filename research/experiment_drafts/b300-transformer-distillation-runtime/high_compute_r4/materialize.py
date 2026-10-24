"""Load and materialize source-bound R4 stages without dispatching any fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import (
    campaign_digest,
    load_campaign,
    read_object,
    require,
    sha256_file,
    write_bytes_new,
    write_json_new,
)
from cell_materialize import _cell_path, _relative, _validate_record
from stage_inner import prepare_inner_stage
from stage_outer import _prepare_g, _prepare_h
from stage_support import _load_selected_parent

STAGES = tuple("BCDEFGH")
SELECTABLE_STAGES = tuple("BCDEF")
INNER_STAGES = frozenset("BCDEF")


def _read_stage_manifest(campaign_root: Path, stage: str) -> dict[str, Any]:
    """Read an immutable stage declaration after validating its sidecar hash."""
    path = campaign_root / "stages" / stage / "stage.json"
    value = read_object(path)
    raw_hash = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    require(
        sidecar.read_text(encoding="utf-8").strip() == raw_hash,
        f"stage manifest hash mismatch: {stage}",
    )
    return value


def load_stage(campaign_root: Path, stage: str) -> dict[str, Any]:
    """Load a prepared stage after rechecking source and parent endpoint evidence."""
    require(stage in STAGES, f"unsupported R4 stage: {stage}")
    root = campaign_root.expanduser().resolve(strict=True)
    campaign = load_campaign(root)
    manifest = _read_stage_manifest(root, stage)
    require(
        manifest.get("schema") == "wnn.high-compute-r4-stage.v1" and manifest.get("stage") == stage,
        "invalid stage manifest",
    )
    require(
        manifest.get("campaign_sha256") == campaign_digest(root),
        "stage belongs to a different immutable campaign",
    )
    require(
        manifest.get("runtime_source_sha256") == campaign["runtime_source"]["sha256"],
        "stage runtime source changed",
    )
    dependencies = manifest.get("dependencies", {})
    for dependency in dependencies.values():
        if isinstance(dependency, dict) and {"path", "sha256"} <= set(dependency):
            path = _cell_path(root, dependency["path"])
            require(sha256_file(path) == dependency["sha256"], f"stage dependency changed: {path}")
    if dependencies:
        from results import validate_stage_dependencies

        validate_stage_dependencies(root, campaign, manifest)
    require(
        isinstance(manifest.get("cells"), list) and manifest["cells"], "stage has no declared cells"
    )
    for record in manifest["cells"]:
        _validate_record(root, campaign, record)
    return manifest


def get_cell(campaign_root: Path, stage: str, cell_id: str) -> dict[str, Any]:
    """Return one cell by ID after validating the stage and all parent evidence."""
    root = campaign_root.expanduser().resolve(strict=True)
    manifest = load_stage(root, stage)
    matches = [cell for cell in manifest["cells"] if cell["id"] == cell_id]
    require(bool(matches), f"unknown cell ID in {stage}: {cell_id}")
    canonical = [cell for cell in matches if not cell["reused"]]
    record = canonical[0] if canonical else matches[0]
    require(
        not (root / record["run_dir"] / "failure.json").is_file(),
        f"cell has a terminal failure receipt and cannot run: {cell_id}",
    )
    return record


def prepare_stage(campaign_root: Path, stage: str) -> dict[str, Any]:
    """Materialize one conditional stage; this function never launches a fit."""
    require(stage in STAGES, f"unsupported R4 stage: {stage}")
    root = campaign_root.expanduser().resolve(strict=True)
    campaign = load_campaign(root)
    stage_dir = root / "stages" / stage
    require(not stage_dir.exists(), f"stage is immutable and already exists: {stage_dir}")
    dependencies: dict[str, Any] = {}
    if stage in "BCDEF":
        records, accounting, dependencies = prepare_inner_stage(root, campaign, stage)
    elif stage == "G":
        _selection, selected, dependencies = _load_selected_parent(root, stage)
        records, accounting = _prepare_g(root, campaign, selected)
    else:
        _selection, _selected, dependencies = _load_selected_parent(root, "H")
        g_comparison = root / "stages/G/comparison.json"
        require(g_comparison.is_file(), "H requires a completed G comparison")
        from results import validate_comparison_file

        validate_comparison_file(root, campaign, "G", g_comparison)
        dependencies["G_comparison"] = {
            "path": _relative(root, g_comparison),
            "sha256": sha256_file(g_comparison),
        }
        g_stage = load_stage(root, "G")
        records, accounting = _prepare_h(root, campaign, g_stage)
    require(
        len({record["id"] for record in records if not record["reused"]})
        == accounting["new_teacher_fits"] + accounting["new_student_fits"],
        "stage fit accounting differs from unique materialized cells",
    )
    for record in records:
        record["stage"] = stage
    declared_variants = sorted({str(record["variant"]) for record in records})
    manifest: dict[str, Any] = {
        "schema": "wnn.high-compute-r4-stage.v1",
        "stage": stage,
        "campaign_sha256": campaign_digest(root),
        "runtime_source_sha256": campaign["runtime_source"]["sha256"],
        "packet_source_sha256": campaign["packet_source"]["sha256"],
        "exposure": "inner_component_validation_within_outer_fit"
        if stage in INNER_STAGES
        else "exposed_outer_fold",
        "selection_policy": "manual_variant_with_reason; no automatic score ranking",
        "dependencies": dependencies,
        "variants": declared_variants,
        "accounting": accounting,
        "cells": records,
        "automatic_fit": False,
        "limits": "Materialized configurations and paths are planned only; this command does not queue or launch training.",
    }
    stage_dir.mkdir(parents=True, exist_ok=False)
    raw_hash = write_json_new(stage_dir / "stage.json", manifest)
    write_bytes_new(stage_dir / "stage.sha256", (raw_hash + "\n").encode("ascii"))
    return manifest
