from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from wnn.all_data_distillation_cache import (
    validate_all_data_distillation_cache,
    validate_all_data_training_groups,
)

from wnn.distillation_cache import FAMILY_ORDER, row_groups_sha256
from wnn.splits import file_sha256


def _write_array(path: Path, values: np.ndarray, groups: list[int]) -> dict[str, object]:
    np.save(path, values, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(values.shape),
        "dtype": "float32",
        "row_groups": groups,
        "row_groups_sha256": row_groups_sha256(groups),
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    monkeypatch.setattr("wnn.all_data_distillation_cache.SEQUENCE_ROWS", 2)
    data_root = tmp_path / "data"
    cache_root = data_root / "cache" / "all_data_family"
    cache_root.mkdir(parents=True)
    (data_root / "cache").mkdir(exist_ok=True)
    orders = {"train": [0, 1], "valid": [3, 4]}
    for split, groups in orders.items():
        np.savez(data_root / "cache" / f"{split}_targets.npz", row_group=np.asarray(groups))

    teacher_manifest = {
        "version": 1,
        "fit": {"train": [0], "valid": [3]},
        "evaluation": {"valid": [4]},
    }
    split_path = data_root / "teacher_split.json"
    split_path.write_text(json.dumps(teacher_manifest))
    split_sha = file_sha256(split_path)
    coefficients = [
        ("strong", "teacher_strong_s3", 0.15),
        ("strong", "teacher_strong_s4", 0.15),
        ("full_finetuned", "teacher_ft_s3", 0.10),
        ("full_finetuned", "teacher_ft_s4", 0.10),
        ("raw_wmse", "teacher_raw_s3", 0.125),
        ("raw_wmse", "teacher_raw_s4", 0.125),
        ("samefold_baseline", "teacher_base_s3", 0.03125),
        ("samefold_baseline", "teacher_base_s4", 0.03125),
    ]
    source_records = []
    source_coefficients = []
    for family, run, coefficient in coefficients:
        checkpoint_sha = hashlib.sha256(run.encode()).hexdigest()
        fit_groups = {"train": [0], "valid": [3]}
        fit_counts = {split: len(groups) for split, groups in fit_groups.items()}
        fit_hashes = {split: row_groups_sha256(groups) for split, groups in fit_groups.items()}
        predictions = {
            split: {
                "row_groups_sha256": row_groups_sha256(groups),
                "checkpoint_sha256": checkpoint_sha,
                "prediction_sha256": hashlib.sha256(f"{run}-{split}".encode()).hexdigest(),
                "sidecar_sha256": hashlib.sha256(f"{run}-{split}-side".encode()).hexdigest(),
            }
            for split, groups in orders.items()
        }
        source_records.append(
            {
                "run": run,
                "family": family,
                "epoch": 3,
                "coefficient": coefficient,
                "checkpoint_sha256": checkpoint_sha,
                "fit_row_groups": fit_groups,
                "fit_group_counts": fit_counts,
                "fit_row_group_hashes": fit_hashes,
                "ancestor_closure": [
                    {
                        "run": run,
                        "checkpoint_sha256": checkpoint_sha,
                        "split_manifest_sha256": split_sha,
                        "fit_row_groups": fit_groups,
                        "fit_group_counts": fit_counts,
                        "fit_row_group_hashes": fit_hashes,
                    }
                ],
                "predictions": predictions,
            }
        )
        source_coefficients.append(
            {"family": family, "run": run, "epoch": 3, "coefficient": coefficient}
        )

    files = {}
    for split, groups in orders.items():
        main = np.zeros((len(groups), 2, 2), dtype=np.float32)
        families = np.zeros((len(groups), 2, 8), dtype=np.float32)
        files[f"main_source_{split}"] = _write_array(
            cache_root / f"main_source_{split}.npy", main, groups
        )
        files[f"families_{split}"] = _write_array(
            cache_root / f"families_{split}.npy", families, groups
        )
    metadata: dict[str, object] = {
        "format": "wnn.heterogeneous-distillation-cache",
        "version": 1,
        "membership_mode": "all_supplied_train_valid",
        "teacher_training_scope": "clean_fold_fit_only",
        "teacher_prediction_coverage": "complete_train_valid_target_index",
        "heldout_claim": False,
        "family_order": list(FAMILY_ORDER),
        "evaluation_valid_groups": [],
        "fit_groups": orders,
        "train_row_groups_sha256": row_groups_sha256(orders["train"]),
        "valid_row_groups_sha256": row_groups_sha256(orders["valid"]),
        "source_coefficients": source_coefficients,
        "coefficient_sum": 0.8125,
        "homogeneous_family": "strong",
        "sources": source_records,
        "files": files,
        "teacher_source_split": {
            "path": "teacher_split.json",
            "sha256": split_sha,
            "evaluation_valid_groups": [4],
            "teachers_fit_evaluation_groups": False,
        },
    }
    (cache_root / "metadata.json").write_text(json.dumps(metadata))
    return cache_root, data_root, metadata


def test_all_data_cache_accepts_complete_ordered_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, data_root, _ = _fixture(tmp_path, monkeypatch)

    result = validate_all_data_distillation_cache(
        cache_root, data_root, "source", require_families=True
    )

    assert set(result["verified_files"]) == {
        "main_source_train",
        "main_source_valid",
        "families_train",
        "families_valid",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_row", "complete target-index membership"),
        ("reordered_row", "complete target-index membership"),
        ("missing_teacher_row", "prediction order is incomplete"),
        ("teacher_fits_eval", "outside clean valid membership"),
        ("heldout_claim", "held-out score claim"),
    ],
)
def test_all_data_cache_rejects_incomplete_or_leaky_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    cache_root, data_root, metadata = _fixture(tmp_path, monkeypatch)
    if mutation == "missing_row":
        metadata["fit_groups"]["valid"] = [3]
    elif mutation == "reordered_row":
        metadata["fit_groups"]["valid"] = [4, 3]
    elif mutation == "missing_teacher_row":
        metadata["sources"][0]["predictions"]["valid"]["row_groups_sha256"] = row_groups_sha256([3])
    elif mutation == "teacher_fits_eval":
        ancestor = metadata["sources"][0]["ancestor_closure"][0]
        ancestor["fit_row_groups"]["valid"] = [3, 4]
        ancestor["fit_group_counts"]["valid"] = 2
        ancestor["fit_row_group_hashes"]["valid"] = row_groups_sha256([3, 4])
    else:
        metadata["heldout_claim"] = True
    (cache_root / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=message):
        validate_all_data_distillation_cache(cache_root, data_root, "source", require_families=True)


def test_training_membership_requires_every_supplied_group_once() -> None:
    validate_all_data_training_groups([4, 2, 3], [8, 9], [2, 3, 4], [8, 9])

    with pytest.raises(ValueError, match="exactly once"):
        validate_all_data_training_groups([2, 3], [8, 9], [2, 3, 4], [8, 9])
    with pytest.raises(ValueError, match="exactly once"):
        validate_all_data_training_groups([2, 3, 3, 4], [8, 9], [2, 3, 4], [8, 9])
