"""Finalize one streamed target cache while hardlinking the frozen auxiliary heads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (
    TARGET_BYTES,
    TRAIN_SEQUENCES,
    combine_target_chunk,
    load_state,
    partial_root,
)
from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
    validate_distillation_cache,
)
from wnn.splits import load_campaign_split


def _empty_array(path: Path, width: int) -> np.ndarray:
    """Write a zero-row primary-valid array required by the canonical cache reader."""
    array = np.empty((0, SEQUENCE_ROWS, width), dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    return array


def _array_record(path: Path, row_groups: list[int], array: np.ndarray) -> dict[str, Any]:
    """Create the canonical hash/shape/order record for one cache array."""
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": "float32",
        "row_groups": row_groups,
        "row_groups_sha256": row_groups_sha256(row_groups),
    }


def finalize_target(output: Path, family: str) -> None:
    """Apply the frozen equation, link immutable auxiliary arrays, and validate the cache."""
    state = load_state(output, family)
    if state["processed_runs"] != state["expected_runs"]:
        raise ValueError("cannot finalize before every frozen teacher has been aggregated")
    old_cache = Path(state["old_cache"])
    project_root = Path(state["project_root"])
    split = load_campaign_split(Path(state["split_manifest"]), project_root)
    if file_sha256(old_cache / "metadata.json") != state["old_cache_metadata_sha256"]:
        raise ValueError("qualified PR60 cache metadata changed after target initialization")
    old_metadata = state["old_metadata"]
    old_main_entry = old_metadata["files"]["main_source_train"]
    old_main_path = old_cache / old_main_entry["path"]
    if file_sha256(old_main_path) != state["old_main_sha256"]:
        raise ValueError("qualified PR60 main target changed after target initialization")
    old_main = np.load(old_main_path, mmap_mode="r", allow_pickle=False)
    total = np.lib.format.open_memmap(state["accumulator"], mode="r+")
    count = len(state["processed_runs"])
    for start in range(0, TRAIN_SEQUENCES, 32):
        stop = min(start + 32, TRAIN_SEQUENCES)
        total[start:stop] = combine_target_chunk(
            np.asarray(old_main[start:stop], dtype=np.float32),
            np.asarray(total[start:stop], dtype=np.float32),
            count,
        )
    total.flush()
    del total, old_main
    partial = partial_root(output)
    main_path = partial / "main_source_train.npy"
    Path(state["accumulator"]).replace(main_path)
    train_groups = list(map(int, state["fit_row_groups"]))
    valid_groups: list[int] = []
    files: dict[str, dict[str, Any]] = {}
    main = np.load(main_path, mmap_mode="r", allow_pickle=False)
    if main.shape != (TRAIN_SEQUENCES, SEQUENCE_ROWS, 2) or main.dtype != np.float32:
        raise ValueError("streamed stage-2 target shape/dtype changed")
    for start in range(0, TRAIN_SEQUENCES, 64):
        if not np.isfinite(main[start : start + 64]).all():
            raise ValueError("streamed stage-2 target contains nonfinite values")
    files["main_source_train"] = _array_record(main_path, train_groups, main)
    del main
    empty_main = _empty_array(partial / "main_source_valid.npy", 2)
    files["main_source_valid"] = _array_record(
        partial / "main_source_valid.npy", valid_groups, empty_main
    )
    for split_name, groups, width in (
        ("train", train_groups, 8),
        ("valid", valid_groups, 8),
    ):
        old_entry = old_metadata["files"][f"families_{split_name}"]
        source = old_cache / old_entry["path"]
        target = partial / f"families_{split_name}.npy"
        os.link(source, target)
        if target.stat().st_size != old_entry["bytes"]:
            raise ValueError(f"retained family array size changed: {split_name}")
        files[f"families_{split_name}"] = dict(old_entry, path=target.name)
        if old_entry["shape"] != [len(groups), SEQUENCE_ROWS, width]:
            raise ValueError(f"retained family array schema changed: {split_name}")
    (partial / "stage2_state.json").replace(partial / "stage2_build_receipt.json")
    metadata = {
        key: old_metadata[key]
        for key in (
            "format",
            "version",
            "fold",
            "split_manifest",
            "split_manifest_sha256",
            "fit_groups",
            "evaluation_valid_groups",
            "family_order",
            "source_coefficients",
            "coefficient_sum",
            "homogeneous_family",
            "output_profile",
            "stored_targets",
            "epoch_convention",
            "sources",
        )
    }
    metadata.update(
        {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "cache_profile": "l4_teacher_capacity_stage2_stream_v1",
            "source_coefficients_scope": "unchanged PR60 base target and retained auxiliary heads",
            "base_cache": {
                "path": str(old_cache),
                "metadata_sha256": state["old_cache_metadata_sha256"],
                "main_source_train_sha256": state["old_main_sha256"],
                "families_train_sha256": old_metadata["files"]["families_train"]["sha256"],
            },
            "target_contract": {
                "formula": "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))",
                "family": family,
                "teacher_runs_equal_weight": list(state["expected_runs"]),
                "teacher_epoch_zero_based": 3,
                "teacher_mean": "ordered float32 clipped sum divided by teacher count",
                "clipping": {
                    "lower": -2.0,
                    "upper": 2.0,
                    "applied_per_teacher_before_mean": True,
                },
                "base_weight": 0.8,
                "new_family_weight": 0.2,
                "new_family_source_scale": 0.8125,
                "fit_groups": state["fit_row_groups"],
                "fit_groups_sha256": state["fit_row_groups_sha256"],
                "teachers": state["teacher_records"],
            },
            "retained_auxiliary_targets": {
                "source_cache": str(old_cache),
                "family_order": old_metadata["family_order"],
                "train_sha256": files["families_train"]["sha256"],
                "train_hardlink_reused_without_writes": True,
                "valid_sha256": files["families_valid"]["sha256"],
            },
            "student_contract": {
                "init": "tl3c_fbc1k_s5@ep3",
                "inherit_init_preprocessing": True,
                "seed_set": [1001, 1002],
                "epochs": 2,
                "checkpoint_policy": "fixed",
                "checkpoint_epoch": 1,
                "precision": "fp32",
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "distill_alpha": 0.7,
                "distill_family_weight": 0.1,
                "hidden": 256,
                "layers": 2,
                "batch": 16,
                "chunk": 2000,
            },
            "files": files,
            "estimated_new_cache_bytes": TARGET_BYTES,
            "prediction_banks_persisted": False,
        }
    )
    metadata_path = partial / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    validate_distillation_cache(partial, project_root, split, "source", require_families=True)
    partial.replace(output)
    print(
        json.dumps(
            {
                "cache": str(output.resolve()),
                "family": family,
                "metadata_sha256": file_sha256(output / "metadata.json"),
                "main_source_train_bytes": files["main_source_train"]["bytes"],
                "new_allocated_cache_bytes": TARGET_BYTES,
                "families_train_sha256": files["families_train"]["sha256"],
            }
        )
    )
