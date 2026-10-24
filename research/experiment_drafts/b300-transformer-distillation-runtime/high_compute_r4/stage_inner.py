"""Materialize conditional B-F inner component-validation recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import PACKET_ROOT, read_object, require
from catalog import teacher_base
from cell_materialize import _cell_path, _new_cell, _reused_cell
from stage_support import _load_selected_parent


def _prepare_b(
    campaign_root: Path, campaign: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the six inner-fold learning-rate discovery teachers."""
    records = []
    for lr_name, lr in (("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3)):
        variant = f"lr_{lr_name}"
        for fold, seed in ((0, 201001), (2, 201002)):
            config = teacher_base(fold, seed)
            config["lr"] = lr
            cell_id = f"B_{variant}_f{fold}"
            require(
                config == read_object(PACKET_ROOT / "stage_b/configs" / f"{cell_id}.json"),
                f"Stage B config differs from the committed handoff: {cell_id}",
            )
            records.append(
                _new_cell(
                    campaign_root,
                    campaign,
                    cell_id,
                    variant,
                    fold,
                    seed,
                    "teacher",
                    config,
                    inner_split=True,
                )
            )
    return records, {"new_teacher_fits": 6, "reused_teacher_fits": 0, "new_student_fits": 0}


def _prepare_c(
    campaign_root: Path, campaign: dict[str, Any], selected: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize two new warmup arms and reuse the exact selected-LR 5% pair."""
    records = []
    parent_variant = selected[0]["variant"]
    for warmup_name, warmup in (("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.1)):
        variant = f"warmup_{warmup_name}"
        for fold in (0, 2):
            source = selected[fold]
            if warmup == 0.05:
                records.append(_reused_cell(source, variant, "B"))
                continue
            config = read_object(_cell_path(campaign_root, source["config"]))
            config["warmup_fraction"] = warmup
            cell_id = f"C_{variant}_f{fold}"
            records.append(
                _new_cell(
                    campaign_root,
                    campaign,
                    cell_id,
                    variant,
                    fold,
                    int(config["seed"]),
                    "teacher",
                    config,
                    inner_split=True,
                )
            )
    return records, {
        "new_teacher_fits": 4,
        "reused_teacher_fits": 2,
        "new_student_fits": 0,
        "selected_parent_variant": parent_variant,
    }


def _prepare_d(
    campaign_root: Path, campaign: dict[str, Any], selected: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize longer attention contexts while preserving the selected recipe."""
    records = []
    for context in (1024, 2048, 4096):
        variant = f"context_{context}"
        for fold in (0, 2):
            source = selected[fold]
            if context == 1024:
                records.append(_reused_cell(source, variant, "C"))
                continue
            config = read_object(_cell_path(campaign_root, source["config"]))
            config["model"]["context_rows"] = context
            cell_id = f"D_{variant}_f{fold}"
            records.append(
                _new_cell(
                    campaign_root,
                    campaign,
                    cell_id,
                    variant,
                    fold,
                    int(config["seed"]),
                    "teacher",
                    config,
                    inner_split=True,
                )
            )
    return records, {"new_teacher_fits": 4, "reused_teacher_fits": 2, "new_student_fits": 0}


def _prepare_e(
    campaign_root: Path, campaign: dict[str, Any], selected: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the long-chunk recipe and reuse the selected 512/8 pair."""
    records = []
    for variant, chunk, accumulation in (("chunk512_acc8", 512, 8), ("chunk2048_acc2", 2048, 2)):
        for fold in (0, 2):
            source = selected[fold]
            if chunk == 512:
                records.append(_reused_cell(source, variant, "D"))
                continue
            config = read_object(_cell_path(campaign_root, source["config"]))
            config["chunk_rows"] = chunk
            config["accumulate_chunks"] = accumulation
            cell_id = f"E_{variant}_f{fold}"
            records.append(
                _new_cell(
                    campaign_root,
                    campaign,
                    cell_id,
                    variant,
                    fold,
                    int(config["seed"]),
                    "teacher",
                    config,
                    inner_split=True,
                )
            )
    return records, {"new_teacher_fits": 2, "reused_teacher_fits": 2, "new_student_fits": 0}


def _prepare_f(
    campaign_root: Path, campaign: dict[str, Any], selected: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the 3.15B targeted capacity arm against the selected 1.54B pair."""
    records = []
    for fold in (0, 2):
        source = selected[fold]
        config = read_object(_cell_path(campaign_root, source["config"]))
        model = config["model"].copy()
        records.append(_reused_cell(source, "teacher_1.54B", "E"))
        config["model"] = model | {"hidden": 2560, "layers": 40, "heads": 40, "ff_multiplier": 4}
        cell_id = f"F_teacher_3.15B_f{fold}"
        records.append(
            _new_cell(
                campaign_root,
                campaign,
                cell_id,
                "teacher_3.15B",
                fold,
                int(config["seed"]),
                "teacher",
                config,
                inner_split=True,
            )
        )
    return records, {"new_teacher_fits": 2, "reused_teacher_fits": 2, "new_student_fits": 0}


def prepare_inner_stage(
    campaign_root: Path, campaign: dict[str, Any], stage: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Prepare one inner-search stage and return its parent dependency bindings."""
    if stage == "B":
        records, accounting = _prepare_b(campaign_root, campaign)
        return records, accounting, {}
    _selection, selected, dependencies = _load_selected_parent(campaign_root, stage)
    if stage == "C":
        records, accounting = _prepare_c(campaign_root, campaign, selected)
    elif stage == "D":
        records, accounting = _prepare_d(campaign_root, campaign, selected)
    elif stage == "E":
        records, accounting = _prepare_e(campaign_root, campaign, selected)
    else:
        records, accounting = _prepare_f(campaign_root, campaign, selected)
    return records, accounting, dependencies
