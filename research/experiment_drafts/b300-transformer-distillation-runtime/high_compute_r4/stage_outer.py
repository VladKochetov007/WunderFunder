"""Materialize G robustness teachers and H student-transfer cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import read_object, require
from catalog import r3_reference_pair_equal, student_base, teacher_base
from cell_materialize import _cell_path, _new_cell, _reused_cell


def _validate_selected_pair_same_recipe(
    selected: dict[int, dict[str, Any]], campaign_root: Path
) -> dict[str, Any]:
    """Require F0/F2 selected configs to differ only in fold and seed identity."""
    configs = {
        fold: read_object(_cell_path(campaign_root, record["config"]))
        for fold, record in selected.items()
    }
    left, right = configs[0].copy(), configs[2].copy()
    left.pop("expected_fold")
    right.pop("expected_fold")
    left.pop("seed")
    right.pop("seed")
    require(left == right, "selected F0/F2 recipe differs beyond fold and seed")
    return configs[0]


def _prepare_g(
    campaign_root: Path, campaign: dict[str, Any], selected: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize three paired outer-fold robustness seeds for both recipe arms."""
    selected_base = _validate_selected_pair_same_recipe(selected, campaign_root)
    alias = all(r3_reference_pair_equal(selected_base, fold, 301001) for fold in (0, 2))
    records = []
    reference_records: dict[tuple[int, int], dict[str, Any]] = {}
    for fold in (0, 2):
        for seed in (301001, 301002, 301003):
            reference = teacher_base(fold, seed)
            reference_id = f"G_reference_f{fold}_s{seed}"
            ref_record = _new_cell(
                campaign_root,
                campaign,
                reference_id,
                "reference",
                fold,
                seed,
                "teacher",
                reference,
                inner_split=False,
            )
            records.append(ref_record)
            reference_records[(fold, seed)] = ref_record
    if alias:
        for fold in (0, 2):
            for seed in (301001, 301002, 301003):
                records.append(
                    _reused_cell(reference_records[(fold, seed)], "candidate", "G-reference-alias")
                )
        new_count = 6
    else:
        for fold in (0, 2):
            for seed in (301001, 301002, 301003):
                candidate = selected_base.copy()
                candidate["model"] = selected_base["model"].copy()
                candidate["expected_fold"] = fold
                candidate["seed"] = seed
                cell_id = f"G_candidate_f{fold}_s{seed}"
                records.append(
                    _new_cell(
                        campaign_root,
                        campaign,
                        cell_id,
                        "candidate",
                        fold,
                        seed,
                        "teacher",
                        candidate,
                        inner_split=False,
                    )
                )
        new_count = 12
    return records, {
        "new_teacher_fits": new_count,
        "reused_teacher_fits": 12 - new_count,
        "new_student_fits": 0,
        "candidate_aliases_reference": alias,
    }


def _cache_producer(record: dict[str, Any], fold: int, seed: int) -> dict[str, Any]:
    """Declare one future G candidate cache input without claiming it exists."""
    return {
        "path": record["teacher_cache"],
        "producer_run_dir": record["run_dir"],
        "checkpoint_path": f"{record['run_dir']}/fit/model.pt",
        "producer_cell_id": record["id"],
        "producer_config_sha256": record["config_sha256"],
        "fold": fold,
        "seed": seed,
        "variant": "candidate",
        "state": "planned_from_G_cell",
    }


def _prepare_h(
    campaign_root: Path, campaign: dict[str, Any], g_stage: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the six fixed student cells and bind one/three G cache inputs."""
    records = []
    candidates = [cell for cell in g_stage["cells"] if cell["variant"] == "candidate"]
    for fold, seed in ((0, 401001), (2, 401002)):
        fold_candidates = sorted(
            (cell for cell in candidates if cell["fold"] == fold), key=lambda item: item["seed"]
        )
        require(
            [cell["seed"] for cell in fold_candidates] == [301001, 301002, 301003],
            "G candidate cache set is incomplete",
        )
        for variant, weight in (("labels", 0.0), ("single_kd", 1.0), ("mean3_kd", 1.0)):
            producers = []
            merged_cache = None
            if variant == "single_kd":
                producers = [_cache_producer(fold_candidates[0], fold, 301001)]
                merged_cache = producers[0]["path"]
            elif variant == "mean3_kd":
                producers = [
                    _cache_producer(cell, fold, int(cell["seed"])) for cell in fold_candidates
                ]
                merged_cache = f"caches/g_candidate_f{fold}_mean"
            inputs = [item["path"] for item in producers]
            config = student_base(fold, seed, weight)
            cell_id = f"H_{variant}_f{fold}"
            records.append(
                _new_cell(
                    campaign_root,
                    campaign,
                    cell_id,
                    variant,
                    fold,
                    seed,
                    "student",
                    config,
                    inner_split=False,
                    cache_inputs=inputs,
                    cache_producers=producers,
                    teacher_cache=merged_cache,
                )
            )
    return records, {"new_teacher_fits": 0, "reused_teacher_fits": 0, "new_student_fits": 6}
