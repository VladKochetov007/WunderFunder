"""Focused checks for the separate H390 all-data cache contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.build_final_distill_cache import PARENT_ARGS, _same_value
from wnn.final_distillation_cache import (
    EXPECTED_PROPENSITY_SHA256,
    EXPECTED_SPLIT_SHA256,
    FINAL_CACHE_FORMAT,
    FINAL_CACHE_VERSION,
    PARENTS,
    TARGET_MULTIPLIER,
    TARGET_WIDTH,
    TEACHER_SPLIT_MANIFEST_RELATIVE,
    TEACHERS,
    file_sha256,
    homogeneous_teacher_target,
    row_groups_sha256,
    source_qualified_groups_sha256,
    validate_final_distillation_cache,
    verify_h390_source_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
H390_SOURCE = (
    REPO / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/h390_all_data_homogeneous_r1/child_source_v1"
)
PARENT_RECEIPTS = H390_SOURCE.parent / "parent_receipts_r1"


def _groups() -> dict[str, list[int]]:
    return {
        "train": list(range(10_606)),
        "valid": list(range(20_000, 21_873)),
    }


def _metadata(groups: dict[str, list[int]]) -> dict:
    split_path = REPO / TEACHER_SPLIT_MANIFEST_RELATIVE
    split_manifest = json.loads(split_path.read_text())
    fit_pools = split_manifest["fit"]
    evaluation_groups = split_manifest["evaluation"]["valid"]
    membership = {
        split: {
            "source": f"datasets/{split}.parquet",
            "count": len(rows),
            "row_groups": rows,
            "row_groups_sha256": row_groups_sha256(rows),
            "source_qualified_sha256": source_qualified_groups_sha256(
                f"datasets/{split}.parquet", rows
            ),
        }
        for split, rows in groups.items()
    }
    membership["evaluation_row_groups"] = []
    membership["source_qualified_groups_sha256"] = {
        split: source_qualified_groups_sha256(f"datasets/{split}.parquet", rows)
        for split, rows in groups.items()
    }
    sources = [
        {
            "run": run,
            "seed": spec["seed"],
            "epoch": 2,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "propensity_checkpoint_sha256": EXPECTED_PROPENSITY_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "fit_membership": {
                split: {
                    "count": len(fit_pools[split]),
                    "row_groups_sha256": row_groups_sha256(sorted(map(int, fit_pools[split]))),
                }
                for split in ("train", "valid")
            },
            "evaluation_annotation": {
                "count": len(evaluation_groups),
                "classification": "teacher clean-fold evaluation pool excluded from teacher fit",
            },
            "predictions": {
                split: {
                    "rows": len(groups[split]),
                    "row_groups_sha256": row_groups_sha256(groups[split]),
                    "prediction_sha256": "c" * 64,
                    "prediction_sidecar_sha256": "d" * 64,
                }
                for split in ("train", "valid")
            },
        }
        for run, spec in TEACHERS.items()
    ]
    parents = [
        {
            "run": spec["run"],
            "seed": seed,
            "checkpoint_epoch": 3,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "provenance_sha256": spec["provenance_sha256"],
            "fit_membership": {
                split: {
                    "count": len(rows),
                    "row_groups_sha256": row_groups_sha256(rows),
                }
                for split, rows in groups.items()
            },
            "evaluation_annotation": {
                "count": 600,
                "row_groups_sha256": spec["reporting_groups_sha256"],
                "classification": "reporting_population_not_heldout",
                "no_eval_during_fit": True,
            },
        }
        for seed, spec in PARENTS.items()
    ]
    return {
        "format": FINAL_CACHE_FORMAT,
        "version": FINAL_CACHE_VERSION,
        "mode": "all_data_final",
        "heldout_claim": False,
        "builder_source_manifest_sha256": "e" * 64,
        "packet_manifest_sha256": "f" * 64,
        "source_split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "target_contract": {
            "family": "strong",
            "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
            "coefficient_multiplier": TARGET_MULTIPLIER,
            "teacher_mean": "unweighted arithmetic mean in float32",
            "teacher_epoch": 2,
            "clipping": "none during cache construction",
        },
        "membership": membership,
        "sources": sources,
        "parents": parents,
        "files": {
            "main_homogeneous_train": {},
            "main_homogeneous_valid": {},
        },
    }


def _data_root(path: Path) -> Path:
    (path / "cache").mkdir(parents=True)
    split_path = path / TEACHER_SPLIT_MANIFEST_RELATIVE
    split_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO / TEACHER_SPLIT_MANIFEST_RELATIVE, split_path)
    groups = _groups()
    for split, rows in groups.items():
        np.savez(
            path / "cache" / f"{split}_targets.npz", row_group=np.asarray(rows, dtype=np.int64)
        )
    return path


def test_h390_source_snapshot_matches_r1_parent_and_child_manifest() -> None:
    result = verify_h390_source_snapshot(H390_SOURCE)
    assert (
        result["parent_archive_sha256"]
        == "35761b8cf6bc5bcf5079b134bfcce81e48b5fd83136630c51b89d48f8759caf1"
    )
    assert (
        result["parent_manifest_sha256"]
        == "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc"
    )
    assert "scripts/train_rnn.py" in result["verified_sources"]
    assert "src/wnn/final_distillation_cache.py" in result["verified_sources"]


@pytest.mark.parametrize(
    ("seed", "digest"),
    [
        (3, "7afe92cbcc9f5f1f615d0b536571bd4fe526365ddbd663e48b8c14619a237a06"),
        (4, "8a33e822638a292beda829163e1f2a750e6f70649b3182aaba7890ae44c4605e"),
    ],
)
def test_parent_receipt_files_match_frozen_provenance_hash(seed: int, digest: str) -> None:
    path = PARENT_RECEIPTS / f"vt_all_fin_mu256p0e4_s{seed}.provenance.json"
    assert file_sha256(path) == digest == PARENTS[seed]["provenance_sha256"]


@pytest.mark.parametrize("seed", [3, 4])
def test_parent_receipts_match_frozen_recipe_and_population(seed: int) -> None:
    run = PARENTS[seed]["run"]
    receipt = json.loads((PARENT_RECEIPTS / f"{run}.provenance.json").read_text())
    args = receipt["args"]
    assert all(_same_value(args.get(key), value) for key, value in PARENT_ARGS.items())
    assert receipt["checkpoint_sha256"]["model_ep3.pt"] == PARENTS[seed]["checkpoint_sha256"]
    assert {split: len(groups) for split, groups in receipt["fit_row_groups"].items()} == {
        "train": 10_606,
        "valid": 1_873,
    }
    assert receipt["args"]["no_eval_during_fit"] is True
    assert len(receipt["evaluation_row_groups"]) == 600


def test_homogeneous_target_uses_float32_mean_then_scale() -> None:
    values = [np.full((2, 3, TARGET_WIDTH), value, dtype=np.float32) for value in (0.1, 0.2, 0.4)]
    expected = (np.add.reduce(values, dtype=np.float32) / np.float32(3)) * np.float32(
        TARGET_MULTIPLIER
    )
    result = homogeneous_teacher_target(values)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_final_cache_rejects_omitting_any_supplied_group(tmp_path: Path) -> None:
    cache = tmp_path / "cache_out"
    cache.mkdir()
    groups = _groups()
    metadata = _metadata(groups)
    metadata["membership"]["train"]["row_groups"].pop()
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train membership differs"):
        validate_final_distillation_cache(cache, _data_root(tmp_path / "data"))


def test_final_cache_rejects_source_qualification_drift(tmp_path: Path) -> None:
    cache = tmp_path / "cache_out"
    cache.mkdir()
    metadata = _metadata(_groups())
    metadata["membership"]["train"]["source"] = "datasets/valid.parquet"
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train membership differs"):
        validate_final_distillation_cache(cache, _data_root(tmp_path / "data"))


def test_final_cache_schema_refuses_family_or_extra_arrays(tmp_path: Path) -> None:
    cache = tmp_path / "cache_out"
    cache.mkdir()
    metadata = _metadata(_groups())
    metadata["files"]["families_train"] = {"path": "families_train.npy"}
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly two homogeneous main arrays"):
        validate_final_distillation_cache(cache, _data_root(tmp_path / "data"))


def test_final_cache_rejects_wrong_population_mode_before_data_access(tmp_path: Path) -> None:
    cache = tmp_path / "cache_out"
    cache.mkdir()
    metadata = _metadata(_groups())
    metadata["mode"] = "clean_fold"
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="explicitly make no heldout claim"):
        validate_final_distillation_cache(cache, tmp_path / "missing_data")
