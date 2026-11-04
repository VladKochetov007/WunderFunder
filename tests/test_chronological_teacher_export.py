from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.distillation import TeacherSource
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
)
from wnn.training.cache_identity import bind_fold_teachers
from wnn.training.chronological_teacher import (
    select_chronological_full_train_index,
    validate_chronological_teacher_checkpoint,
    validate_chronological_teacher_membership,
    validate_chronological_train_index,
)
from wnn.training.environment import TrainingEnvironment


def _split(path: Path) -> CampaignSplit:
    return CampaignSplit(
        path,
        "chronological-split-sha",
        0,
        1,
        (1, 2),
        (),
        (9,),
        (("train", 1, 0), ("train", 2, 1), ("train", 3, 2), ("valid", 9, 3)),
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )


def _checkpoint_artifacts(root: Path, split: CampaignSplit) -> tuple[dict, dict, Path, str]:
    run = root / "artifacts" / "runs" / "teacher"
    run.mkdir(parents=True)
    checkpoint_path = run / "model_ep2.pt"
    checkpoint_provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [2, 1], "valid": []},
        "evaluation_row_groups": [9],
    }
    args = {
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 2,
        "no_eval_during_fit": False,
        "train_source": "train",
        "split_manifest_sha256": split.sha256,
    }
    checkpoint = {"epoch": 2, "args": args, "provenance": checkpoint_provenance}
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    run_provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [1, 2], "valid": []},
        "evaluation_row_groups": [9],
        "args": args,
        "checkpoint_sha256": {"model_ep2.pt": checkpoint_sha},
    }
    provenance_path = run / "provenance.json"
    provenance_path.write_text(json.dumps(run_provenance), encoding="utf-8")
    return checkpoint, run_provenance, checkpoint_path, checkpoint_sha


def test_chronological_membership_requires_full_train_empty_valid_and_full_eval(
    tmp_path: Path,
) -> None:
    split = _split(tmp_path / "primary.json")
    provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [2, 1], "valid": []},
        "evaluation_row_groups": [9],
    }
    validate_chronological_teacher_membership(provenance, split, "teacher")

    for section, bad in (
        ("train", [1]),
        ("valid", [9]),
    ):
        changed = json.loads(json.dumps(provenance))
        changed["fit_row_groups"][section] = bad
        with pytest.raises(ValueError):
            validate_chronological_teacher_membership(changed, split, "teacher")
    changed = provenance | {"evaluation_row_groups": []}
    with pytest.raises(ValueError, match="complete official validation"):
        validate_chronological_teacher_membership(changed, split, "teacher")


def test_chronological_checkpoint_is_bound_to_hash_fixed_epoch_and_provenance(
    tmp_path: Path,
) -> None:
    split = _split(tmp_path / "primary.json")
    checkpoint, provenance, checkpoint_path, digest = _checkpoint_artifacts(tmp_path, split)
    validate_chronological_teacher_checkpoint(
        checkpoint, provenance, split, checkpoint_path.name, digest, "teacher"
    )

    changed = dict(provenance)
    changed["checkpoint_sha256"] = {"model_ep2.pt": "f" * 64}
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_chronological_teacher_checkpoint(
            checkpoint, changed, split, checkpoint_path.name, digest, "teacher"
        )
    wrong_epoch = dict(checkpoint)
    wrong_epoch["args"] = dict(checkpoint["args"], checkpoint_epoch=1)
    with pytest.raises(ValueError, match="checkpoint_epoch"):
        validate_chronological_teacher_checkpoint(
            wrong_epoch, provenance, split, checkpoint_path.name, digest, "teacher"
        )


def test_builder_accepts_validated_fixed_teacher_and_rejects_no_eval_ancestor(
    tmp_path: Path,
) -> None:
    from scripts.build_distill_cache import _audit_ancestor_closure

    split = _split(tmp_path / "primary.json")
    checkpoint, _, checkpoint_path, _ = _checkpoint_artifacts(tmp_path, split)
    checkpoint_root = tmp_path / "artifacts" / "runs"
    closure = _audit_ancestor_closure("teacher", split, checkpoint_root, checkpoint_path.name)
    assert closure[0]["fit_group_counts"] == {"train": 2, "valid": 0}

    checkpoint["args"] = dict(checkpoint["args"], no_eval_during_fit=True)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="skipped required official validation"):
        _audit_ancestor_closure("teacher", split, checkpoint_root, checkpoint_path.name)


def test_full_train_index_must_cover_source_without_reordering_it(tmp_path: Path) -> None:
    split = _split(tmp_path / "primary.json")
    validate_chronological_train_index([3, 1, 2], split)
    with pytest.raises(ValueError, match="full official training source"):
        validate_chronological_train_index([1, 2], split)
    with pytest.raises(ValueError, match="repeats"):
        validate_chronological_train_index([1, 2, 2], split)


def test_prediction_selector_uses_full_source_index_after_checkpoint_admission(
    tmp_path: Path,
) -> None:
    split = _split(tmp_path / "primary.json")
    checkpoint, provenance, checkpoint_path, digest = _checkpoint_artifacts(tmp_path, split)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(
        cache / "train_index.npz", row_group=np.asarray([3, 1, 2]), seq_ix=np.asarray([30, 10, 20])
    )
    np.savez(cache / "train_targets.npz", row_group=np.asarray([3, 1, 2]))

    selected = select_chronological_full_train_index(
        tmp_path,
        split,
        checkpoint,
        checkpoint_path.name,
        digest,
        provenance,
        "teacher",
    )
    np.testing.assert_array_equal(selected, [3, 1, 2])


def test_legacy_binder_accepts_only_full_index_chronological_exports(tmp_path: Path) -> None:
    split = _split(tmp_path / "primary.json")
    _, _, checkpoint_path, checkpoint_sha = _checkpoint_artifacts(tmp_path, split)
    cache = tmp_path / "cache"
    cache.mkdir()
    groups = np.asarray([3, 1, 2], dtype=np.int64)
    np.savez(cache / "train_index.npz", row_group=groups, seq_ix=np.asarray([30, 10, 20]))
    np.savez(cache / "train_targets.npz", row_group=groups)
    prediction_path = cache / "pred_teacher_train.npy"
    np.save(prediction_path, np.zeros((3, 4, 2), dtype=np.float32))
    provenance_path = checkpoint_path.parent / "provenance.json"
    sidecar = {
        "run": "teacher",
        "split": "train",
        "epoch": 2,
        "split_manifest_sha256": split.sha256,
        "campaign_split_manifest_sha256": split.sha256,
        "campaign_export_scope": "chronological-full-train-index-feature-only",
        "feature_only": True,
        "row_groups": groups.tolist(),
        "evaluation_membership_sha256": hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest(),
        "prediction_sha256": file_sha256(prediction_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "run_provenance_sha256": file_sha256(provenance_path),
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

    bound = bind_fold_teachers(
        [("teacher", 0.4)],
        split,
        ("train",),
        environment=TrainingEnvironment(root=tmp_path, sequence_length=4),
    )
    assert bound[0]["checkpoint_sha256"] == checkpoint_sha
    assert bound[0]["epoch"] == 2
    assert bound[0]["run_provenance_sha256"] == sidecar["run_provenance_sha256"]
    assert (
        bound[0]["row_groups_sha256"] == hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest()
    )

    sidecar["campaign_export_scope"] = "fit"
    prediction_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="full-index train-only"):
        bind_fold_teachers(
            [("teacher", 0.4)],
            split,
            ("train",),
            environment=TrainingEnvironment(root=tmp_path, sequence_length=4),
        )


def test_chronological_cache_builder_emits_train_only_full_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_distill_cache as builder
    from wnn.distillation_cache import validate_distillation_cache

    split = _split(tmp_path / "primary.json")
    data_root = tmp_path / "data"
    (data_root / "cache").mkdir(parents=True)
    np.savez(
        data_root / "cache/train_index.npz",
        row_group=np.asarray([3, 1, 2]),
        seq_ix=np.asarray([30, 10, 20]),
    )
    np.savez(data_root / "cache/train_targets.npz", row_group=np.asarray([3, 1, 2]))
    sources = [
        TeacherSource("strong", "strong_teacher", 0.3, 2),
        TeacherSource("full_finetuned", "full_teacher", 0.2, 2),
        TeacherSource("raw_wmse", "raw_teacher", 0.25, 2),
        TeacherSource("samefold_baseline", "base_teacher", 0.0625, 2),
    ]
    values = {
        source.run: np.full((3, builder.SEQUENCE_ROWS, 2), index / 10, dtype=np.float32)
        for index, source in enumerate(sources, start=1)
    }
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    monkeypatch.setattr(
        builder,
        "_target_order",
        lambda _root, name: [3, 1, 2] if name == "train" else pytest.fail("valid index unused"),
    )
    monkeypatch.setattr(builder, "_audit_ancestor_closure", lambda *_args, **_kwargs: [])

    def inspect_teacher(run: str, *_args: object) -> tuple[dict, dict, dict]:
        return (
            {
                "run": run,
                "checkpoint_sha256": "a" * 64,
                "provenance_sha256": "b" * 64,
                "epoch": 2,
            },
            {"train": values[run]},
            {},
        )

    monkeypatch.setattr(builder, "_inspect_teacher", inspect_teacher)
    args = Namespace(
        data_root=data_root,
        teacher_cache_root=tmp_path / "predictions",
        checkpoint_root=tmp_path / "checkpoints",
        split_manifest=split.path,
        output=tmp_path / "teacher-cache",
        teacher=sources,
        expected_coefficient_sum=0.8125,
        homogeneous_family="strong",
        chunk_sequences=1,
        min_free_gb=0.0,
        output_budget_gb=1.0,
        output_profile="all",
    )
    output = builder.build_cache(args)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["fit_groups"] == {"train": [1, 2], "valid": []}
    assert metadata["files"]["main_source_train"]["row_groups"] == [1, 2]
    assert metadata["files"]["main_source_valid"]["row_groups"] == []
    assert np.load(output / "main_source_valid.npy", allow_pickle=False).shape == (
        0,
        builder.SEQUENCE_ROWS,
        2,
    )
    result = validate_distillation_cache(output, data_root, split, "source", require_families=True)
    assert set(result["verified_files"]) == {
        "main_source_train",
        "families_train",
        "main_source_valid",
        "families_valid",
    }
