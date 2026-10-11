"""Focused checks for the separate H390 all-data cache contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from scripts.build_final_distill_cache import (
    PARENT_ARGS,
    _same_value,
    _teacher_provenance_contract,
)
from wnn.final_distillation_cache import (
    DISK_SAFETY_RESERVE_BYTES,
    EXPECTED_PROPENSITY_SHA256,
    EXPECTED_SPLIT_SHA256,
    FINAL_CACHE_FORMAT,
    FINAL_CACHE_VERSION,
    PARENTS,
    TARGET_MULTIPLIER,
    TARGET_WIDTH,
    TEACHER_SPLIT_MANIFEST_RELATIVE,
    TEACHERS,
    cache_disk_requirement,
    file_sha256,
    homogeneous_teacher_target,
    row_groups_sha256,
    source_qualified_groups_sha256,
    validate_final_distillation_cache,
    verify_h390_r1_source_snapshot,
    verify_h390_source_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
H390_SOURCE = (
    REPO / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/h390_all_data_homogeneous_r1/child_source_v1"
)
H390_SOURCE_R2 = (
    REPO / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/h390_all_data_homogeneous_r2/child_source_v1"
)
PARENT_RECEIPTS = H390_SOURCE.parent / "parent_receipts_r1"
TEACHER_RECEIPTS_R2 = H390_SOURCE_R2.parent / "teacher_receipts_r2"


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
            "provenance_sha256": spec["provenance_sha256"],
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
        "output_bytes": cache_disk_requirement(12_479)[0],
        "disk_safety_reserve_bytes": DISK_SAFETY_RESERVE_BYTES,
        "estimated_bytes": cache_disk_requirement(12_479)[2],
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


def test_h390_r1_source_snapshot_remains_immutable() -> None:
    result = verify_h390_r1_source_snapshot(H390_SOURCE)
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


def test_h390_r2_snapshot_binds_teacher_receipts_and_r1_ancestry() -> None:
    result = verify_h390_source_snapshot(H390_SOURCE_R2)
    assert (
        result["parent_archive_sha256"]
        == "35761b8cf6bc5bcf5079b134bfcce81e48b5fd83136630c51b89d48f8759caf1"
    )
    assert result["teacher_provenance_sha256"] == {
        run: spec["provenance_sha256"] for run, spec in TEACHERS.items()
    }
    assert "scripts/train_rnn.py" in result["verified_sources"]


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


@pytest.mark.parametrize("run", list(TEACHERS))
def test_actual_teacher_producer_receipt_is_pinned_before_checkpoint_access(run: str) -> None:
    receipt_path = TEACHER_RECEIPTS_R2 / f"{run}.provenance.json"
    provenance, digest = _teacher_provenance_contract(run, receipt_path)
    assert digest == TEACHERS[run]["provenance_sha256"]
    assert provenance["checkpoint_sha256"]["model_ep2.pt"] == TEACHERS[run]["checkpoint_sha256"]


def test_teacher_receipt_check_rejects_missing_provenance_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = next(iter(TEACHERS))
    monkeypatch.delitem(TEACHERS[run], "provenance_sha256")
    with pytest.raises(ValueError, match="no reviewed producer provenance hash"):
        _teacher_provenance_contract(run, TEACHER_RECEIPTS_R2 / f"{run}.provenance.json")


def test_real_sequences_consumer_reads_homogeneous_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_values = np.full((1, 20_000, 2), 0.25, dtype=np.float32)
    np.save(cache_root / "main_homogeneous_train.npy", cache_values)
    source_root = H390_SOURCE_R2
    consumer_script = textwrap.dedent(
        """
        import importlib.util
        import sys
        from pathlib import Path

        import numpy as np

        source_root = Path(sys.argv[1])
        cache_root = Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location(
            "h390_r2_train_rnn_consumer", source_root / "scripts/train_rnn.py"
        )
        assert spec is not None and spec.loader is not None
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)
        trainer.HEURISTICS = False

        class FakeSequenceFile:
            def __init__(self, _path):
                pass

            def read(self, _row_group):
                return type(
                    "SequenceFixture",
                    (),
                    {
                        "features": np.zeros((20_000, trainer.N_FEATURES), dtype=np.float32),
                        "targets": np.zeros((20_000, 2), dtype=np.float32),
                        "scored": np.zeros(20_000, dtype=np.bool_),
                    },
                )()

        trainer.SequenceFile = FakeSequenceFile
        dataset = trainer.Sequences(
            "train",
            [17],
            distill_cache_root=cache_root,
            distill_cache_variant="homogeneous",
            distill_cache_metadata={
                "metadata": {"files": {"main_homogeneous_train": {"row_groups": [17]}}}
            },
        )
        features, targets, scored = dataset[0]
        assert tuple(targets.shape) == (20_000, 4)
        assert abs(targets[:, 2:].mean().item() - 0.25) < 1e-6
        assert features.shape[0] == 20_000
        assert not scored.any().item()
        """
    )
    env = dict(os.environ)
    child_src = str(source_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (child_src, env.get("PYTHONPATH", "")) if part
    )
    env["WNN_PROJECT_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", consumer_script, str(source_root), str(cache_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_cache_disk_estimate_includes_safety_reserve() -> None:
    output_bytes, reserve_bytes, admitted_bytes = cache_disk_requirement(12_479)
    assert reserve_bytes >= 16 * 1024 * 1024
    assert admitted_bytes == output_bytes + reserve_bytes


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
