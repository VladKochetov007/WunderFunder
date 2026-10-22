"""Run the frozen online relationship readout panel from immutable caches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.online_relationship import fit_online_relationship_normalizer
from wnn.splits import file_sha256
from wnn.training.online_relationship_cache_validation import load_fold_caches
from wnn.training.online_relationship_exports import (
    canonical_json_sha256,
    write_candidate,
    write_json_create_only,
)
from wnn.training.online_relationship_fit import fit_online_relationship_head
from wnn.training.online_relationship_sources import (
    find_fold,
    read_json,
    resolve,
)

_LOG = logging.getLogger(__name__)


def run_online_relationship_panel(
    recipe_path: Path,
    *,
    root: Path,
    device: str = "cuda",
) -> dict[str, object]:
    """Fit each of the twelve recipe cells in order from two validated parent caches."""
    root = root.resolve()
    recipe_path = resolve(root, recipe_path)
    recipe = read_json(recipe_path)
    recipe_sha = file_sha256(recipe_path)
    if (
        recipe.get("schema_version") != 1
        or recipe.get("resource_contract", {}).get("total_new_fits") != 12
    ):
        raise ValueError("recipe is not the frozen 12-fit online relationship panel")
    if recipe.get("status") != "admitted_for_execution":
        raise ValueError("the root must admit the complete panel in the recipe before fitting")
    _require_weight_contract(recipe)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for fold_index in (0, 1):
        fold = find_fold(recipe, fold_index)
        fit_cache, eval_cache, parent_manifest = load_fold_caches(
            root, recipe, recipe_sha, fold, device
        )
        for lag in (1, 16):
            normalizer = fit_online_relationship_normalizer(
                (sequence.raw for sequence in fit_cache.iter_sequences()), lag
            )
            normalizer_dict = normalizer.to_dict()
            normalizer_sha = canonical_json_sha256(normalizer_dict)
            normalizer_path = Path(fold["parent_cache"]) / f"relationship_normalizer_lag{lag}.json"
            _save_normalizer(normalizer_path, normalizer_dict, normalizer_sha)
            for arm in ("A", "B", "C"):
                cell = _find_cell(recipe, fold_index, lag, arm)
                result = fit_online_relationship_head(
                    fit_cache.iter_sequences,
                    eval_cache.iter_sequences,
                    arm=arm,
                    lag=lag,
                    seed=int(fold["seed"]),
                    parent_mean=np.asarray(fit_cache.metadata["parent_mean"], dtype=np.float32),
                    parent_std=np.asarray(fit_cache.metadata["parent_std"], dtype=np.float32),
                    relationship_normalizer=normalizer,
                    device=device,
                    progress=lambda record, run=cell["run"]: _LOG.info(
                        "online_relationship_progress %s",
                        json.dumps({"run": run, **record}, sort_keys=True),
                    ),
                )
                results.append(
                    write_candidate(
                        root,
                        recipe,
                        recipe_path,
                        recipe_sha,
                        fold,
                        cell,
                        fit_cache,
                        eval_cache,
                        parent_manifest,
                        normalizer_dict,
                        normalizer_sha,
                        result,
                    )
                )
    return {
        "schema": "wnn.online-relationships.fit-panel-result.v1",
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha,
        "completed_cells": len(results),
        "results": results,
    }


def _save_normalizer(path: Path, value: dict[str, object], digest: str) -> None:
    """Create or verify the shared fit-only normalizer for one fold and lag."""
    record = {"normalizer": value, "normalizer_sha256": digest}
    if path.exists():
        current = read_json(path)
        if current != record:
            raise ValueError(
                "saved relationship normalizer differs from current fitting membership"
            )
        return
    write_json_create_only(path, record)


def _find_cell(recipe: dict[str, Any], fold: int, lag: int, arm: str) -> dict[str, Any]:
    matches = [
        cell
        for cell in recipe["cells"]
        if int(cell["fold"]) == fold and int(cell["lag"]) == lag and cell["arm"] == arm
    ]
    if len(matches) != 1:
        raise ValueError(f"recipe must define exactly one fold={fold}, lag={lag}, arm={arm} cell")
    return matches[0]


def _require_weight_contract(recipe: dict[str, Any]) -> None:
    """Require the approved common sequence-level weighting amendment."""
    readout = recipe.get("readout", {})
    args = {
        "mask_weight": False,
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "prop_cache": False,
        "class_weight": None,
        "valid_true_mask": 0.03,
        "sync_free": False,
        "since_weight": 0.0,
        "since_floor": 0.1,
        "episode_weight": 0.0,
        "episode_floor": 0.1,
    }
    semantics = (
        "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
    )
    if (
        readout.get("row_weight_args") != args
        or readout.get("row_weight_batch_semantics") != semantics
    ):
        raise ValueError("recipe lacks the frozen common fit row-weight contract")
