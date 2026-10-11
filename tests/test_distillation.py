from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH
from wnn.distillation import (
    TeacherSource,
    combine_teacher_chunk,
    extend_family_output_head,
    four_family_weighted_mse,
)
from wnn.distillation_cache import CACHE_FORMAT, CACHE_VERSION, FAMILY_ORDER, row_groups_sha256
from wnn.splits import CampaignSplit


def test_teacher_mix_preserves_source_scale_and_family_means() -> None:
    sources = (
        TeacherSource("strong", "strong3", 0.15),
        TeacherSource("strong", "strong4", 0.15),
        TeacherSource("full_finetuned", "full3", 0.1),
        TeacherSource("full_finetuned", "full4", 0.1),
        TeacherSource("raw_wmse", "raw3", 0.125),
        TeacherSource("raw_wmse", "raw4", 0.125),
        TeacherSource("samefold_baseline", "baseline3", 0.03125),
        TeacherSource("samefold_baseline", "baseline4", 0.03125),
    )
    values = {
        name: np.full((1, 2, 2), value, dtype=np.float32)
        for name, value in (
            ("strong3", 0.1),
            ("strong4", 0.3),
            ("full3", 0.2),
            ("full4", 0.4),
            ("raw3", 0.6),
            ("raw4", 0.8),
            ("baseline3", 1.0),
            ("baseline4", 1.2),
        )
    }

    main, normalized, homogeneous, families = combine_teacher_chunk(
        values, sources, FAMILY_ORDER, "strong"
    )

    assert main.shape == (1, 2, 2)
    assert families.shape == (1, 2, 8)
    np.testing.assert_allclose(main, 0.36375, rtol=1e-6)
    np.testing.assert_allclose(normalized, main / np.float32(0.8125), rtol=1e-6)
    np.testing.assert_allclose(homogeneous, 0.1625, rtol=1e-6)
    np.testing.assert_allclose(families[0, 0], [0.2, 0.2, 0.3, 0.3, 0.7, 0.7, 1.1, 1.1])


def test_family_loss_uses_the_original_two_target_weights() -> None:
    predictions = torch.zeros((1, 2, 4, 2))
    targets = torch.ones_like(predictions)
    weights = torch.tensor([[[0.25, 0.0], [2.0, 1.0]]])

    loss = four_family_weighted_mse(predictions, targets, weights)

    assert loss.item() == pytest.approx(1.0)
    assert four_family_weighted_mse(predictions, targets, torch.zeros_like(weights)).item() == 0.0


def test_family_head_warm_start_copies_parent_and_preserves_appended_outputs() -> None:
    parent = {
        "trunk": torch.tensor([1.0, 2.0]),
        "head.2.weight": torch.arange(69, dtype=torch.float32).reshape(23, 3),
        "head.2.bias": torch.arange(23, dtype=torch.float32),
    }
    initialized = {
        "trunk": torch.zeros(2),
        "head.2.weight": torch.full((31, 3), 7.0),
        "head.2.bias": torch.full((31,), 8.0),
    }

    merged = extend_family_output_head(parent, initialized)

    torch.testing.assert_close(merged["trunk"], parent["trunk"])
    torch.testing.assert_close(merged["head.2.weight"][:23], parent["head.2.weight"])
    torch.testing.assert_close(merged["head.2.bias"][:23], parent["head.2.bias"])
    torch.testing.assert_close(merged["head.2.weight"][23:], initialized["head.2.weight"][23:])
    torch.testing.assert_close(merged["head.2.bias"][23:], initialized["head.2.bias"][23:])


def _write_cache_array(path: Path, values: np.ndarray, groups: list[int]) -> dict[str, object]:
    np.save(path, values, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "shape": list(values.shape),
        "dtype": "float32",
        "row_groups": groups,
        "row_groups_sha256": row_groups_sha256(groups),
    }


def _tiny_cache(tmp_path: Path) -> tuple[Path, Path, CampaignSplit]:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "distill"
    (data_root / "cache").mkdir(parents=True)
    cache_root.mkdir()
    np.savez(data_root / "cache/train_targets.npz", row_group=np.array([10, 11]))
    np.savez(data_root / "cache/valid_targets.npz", row_group=np.array([20, 21]))
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "fold0-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    train_groups, valid_groups = [10], [20]
    arrays = {
        "main_source_train": (2, train_groups),
        "main_source_valid": (2, valid_groups),
        "families_train": (8, train_groups),
        "families_valid": (8, valid_groups),
    }
    files: dict[str, dict[str, object]] = {}
    for key, (width, groups) in arrays.items():
        values = np.ones((1, SEQUENCE_LENGTH, width), dtype=np.float32)
        files[key] = _write_cache_array(cache_root / f"{key}.npy", values, groups)
    coefficients = [
        {"family": "strong", "run": "s3", "epoch": 3, "coefficient": 0.15},
        {"family": "strong", "run": "s4", "epoch": 3, "coefficient": 0.15},
        {"family": "full_finetuned", "run": "f3", "epoch": 0, "coefficient": 0.1},
        {"family": "full_finetuned", "run": "f4", "epoch": 0, "coefficient": 0.1},
        {"family": "raw_wmse", "run": "r3", "epoch": 3, "coefficient": 0.125},
        {"family": "raw_wmse", "run": "r4", "epoch": 3, "coefficient": 0.125},
        {"family": "samefold_baseline", "run": "o3", "epoch": 3, "coefficient": 0.03125},
        {"family": "samefold_baseline", "run": "o4", "epoch": 3, "coefficient": 0.03125},
    ]
    metadata = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "fold": 0,
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": [10], "valid": [20]},
        "evaluation_valid_groups": [21],
        "family_order": list(FAMILY_ORDER),
        "sources": [
            {
                "run": item["run"],
                "family": item["family"],
                "epoch": item["epoch"],
                "coefficient": item["coefficient"],
            }
            for item in coefficients
        ],
        "source_coefficients": coefficients,
        "coefficient_sum": 0.8125,
        "homogeneous_family": "strong",
        "epoch_convention": {
            "friend_at_ep1": "second additional epoch",
            "student_epochs": 2,
            "student_checkpoint_policy": "fixed",
            "student_checkpoint_epoch": 1,
        },
        "files": files,
    }
    (cache_root / "metadata.json").write_text(json.dumps(metadata))
    return data_root, cache_root, split


def test_preweighted_cache_validates_membership_alignment_and_bytes(tmp_path: Path) -> None:
    data_root, cache_root, split = _tiny_cache(tmp_path)

    result = validate_cache(cache_root, data_root, split)

    assert set(result["verified_files"]) == {
        "main_source_train",
        "main_source_valid",
        "families_train",
        "families_valid",
    }
    main_path = cache_root / "main_source_train.npy"
    np.save(main_path, np.zeros((1, SEQUENCE_LENGTH, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="bytes disagree"):
        validate_cache(cache_root, data_root, split)


def test_preweighted_cache_recomputes_declared_coefficient_sum(tmp_path: Path) -> None:
    data_root, cache_root, split = _tiny_cache(tmp_path)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_coefficients"][0]["coefficient"] += 0.01
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="coefficients must sum"):
        validate_cache(cache_root, data_root, split)


def test_homogeneous_only_cache_validates_only_the_requested_main_target(tmp_path: Path) -> None:
    from wnn.distillation_cache import validate_distillation_cache

    data_root, cache_root, split = _tiny_cache(tmp_path)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    files: dict[str, dict[str, object]] = {}
    for split_name, groups in (("train", [10]), ("valid", [20])):
        key = f"main_homogeneous_{split_name}"
        files[key] = _write_cache_array(
            cache_root / f"{key}.npy",
            np.full((1, SEQUENCE_LENGTH, 2), np.float32(0.8125), dtype=np.float32),
            groups,
        )
    metadata["files"] = files
    metadata["output_profile"] = "homogeneous-only"
    metadata["stored_targets"] = ["main_homogeneous"]
    metadata_path.write_text(json.dumps(metadata))

    result = validate_distillation_cache(
        cache_root, data_root, split, "homogeneous", require_families=False
    )

    assert set(result["verified_files"]) == {
        "main_homogeneous_train",
        "main_homogeneous_valid",
    }
    with pytest.raises(ValueError, match="missing main_source_train"):
        validate_distillation_cache(cache_root, data_root, split, "source", require_families=False)
    with pytest.raises(ValueError, match="missing families_train"):
        validate_distillation_cache(
            cache_root, data_root, split, "homogeneous", require_families=True
        )


def test_homogeneous_only_producer_is_bitwise_equal_to_full_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_distill_cache as builder

    data_root = tmp_path / "data"
    data_root.mkdir()
    split = CampaignSplit(
        tmp_path / "fold1.json",
        "fold1-sha",
        1,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    sources = [
        TeacherSource("strong", "strong_s3", 0.15, 2),
        TeacherSource("strong", "strong_s4", 0.15, 2),
        TeacherSource("full_finetuned", "full_alias", 0.2, 2),
        TeacherSource("raw_wmse", "raw_alias", 0.25, 2),
        TeacherSource("samefold_baseline", "baseline_alias", 0.0625, 2),
    ]
    values = {
        source.run: np.full(
            (1, SEQUENCE_LENGTH, 2),
            np.float32(index / 10 + 0.0000001),
            dtype=np.float32,
        )
        for index, source in enumerate(sources, start=1)
    }
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    monkeypatch.setattr(
        builder,
        "_target_order",
        lambda _root, name: [10] if name == "train" else [20],
    )
    monkeypatch.setattr(builder, "_audit_ancestor_closure", lambda *_args, **_kwargs: [])

    def inspect_teacher(
        run: str, *_args: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {"run": run}, {"train": values[run], "valid": values[run]}, {}

    monkeypatch.setattr(builder, "_inspect_teacher", inspect_teacher)

    def arguments(output: Path, profile: str) -> Namespace:
        return Namespace(
            data_root=data_root,
            teacher_cache_root=tmp_path / "predictions",
            checkpoint_root=tmp_path / "checkpoints",
            split_manifest=split.path,
            output=output,
            teacher=sources,
            expected_coefficient_sum=0.8125,
            homogeneous_family="strong",
            chunk_sequences=1,
            min_free_gb=0.0,
            output_budget_gb=1.0,
            output_profile=profile,
        )

    full_path = builder.build_cache(arguments(tmp_path / "full", "all"))
    compact_path = builder.build_cache(arguments(tmp_path / "compact", "homogeneous-only"))

    assert np.array_equal(
        np.load(full_path / "main_homogeneous_train.npy", allow_pickle=False),
        np.load(compact_path / "main_homogeneous_train.npy", allow_pickle=False),
    )
    assert np.array_equal(
        np.load(full_path / "main_homogeneous_valid.npy", allow_pickle=False),
        np.load(compact_path / "main_homogeneous_valid.npy", allow_pickle=False),
    )
    compact_metadata = json.loads((compact_path / "metadata.json").read_text())
    assert compact_metadata["stored_targets"] == ["main_homogeneous"]
    assert set(compact_metadata["files"]) == {
        "main_homogeneous_train",
        "main_homogeneous_valid",
    }
    assert {path.name for path in compact_path.iterdir()} == {
        "main_homogeneous_train.npy",
        "main_homogeneous_valid.npy",
        "metadata.json",
    }
    full_metadata = json.loads((full_path / "metadata.json").read_text())
    assert set(full_metadata["files"]) == {
        "main_source_train",
        "main_source_valid",
        "main_normalized_train",
        "main_normalized_valid",
        "main_homogeneous_train",
        "main_homogeneous_valid",
        "families_train",
        "families_valid",
    }


def test_compact_cache_storage_preflight_precedes_source_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_distill_cache as builder

    split = CampaignSplit(
        tmp_path / "fold1.json",
        "fold1-sha",
        1,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    sources = [
        TeacherSource("strong", "strong_s3", 0.3, 2),
        TeacherSource("full_finetuned", "full_alias", 0.2, 2),
        TeacherSource("raw_wmse", "raw_alias", 0.25, 2),
        TeacherSource("samefold_baseline", "baseline_alias", 0.0625, 2),
    ]
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    monkeypatch.setattr(
        builder,
        "_target_order",
        lambda _root, name: [10] if name == "train" else [20],
    )
    monkeypatch.setattr(
        builder,
        "shutil",
        SimpleNamespace(disk_usage=lambda _path: SimpleNamespace(free=0)),
    )

    def unexpected_inspection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("source checkpoint and prediction inspection must follow storage preflight")

    monkeypatch.setattr(builder, "_audit_ancestor_closure", unexpected_inspection)
    monkeypatch.setattr(builder, "_inspect_teacher", unexpected_inspection)
    args = Namespace(
        data_root=tmp_path,
        teacher_cache_root=tmp_path / "predictions",
        checkpoint_root=tmp_path / "checkpoints",
        split_manifest=split.path,
        output=tmp_path / "compact",
        teacher=sources,
        expected_coefficient_sum=0.8125,
        homogeneous_family="strong",
        chunk_sequences=1,
        min_free_gb=20.0,
        output_budget_gb=16.0,
        output_profile="homogeneous-only",
    )

    with pytest.raises(OSError, match="distillation cache needs"):
        builder.build_cache(args)


def test_homogeneous_cache_size_matches_float32_payload_widths() -> None:
    from scripts.build_distill_cache import estimate_output_bytes

    assert estimate_output_bytes(11_949, "all") == 13_382_880_000
    assert estimate_output_bytes(11_949, "homogeneous-only") == 1_911_840_000


def test_teacher_epoch_and_clean_membership_allow_valid_only_teacher() -> None:
    from scripts.build_distill_cache import _verify_run_membership, parse_teacher

    teacher = parse_teacher("full_finetuned=run_name@ep0*0.1")
    assert teacher == TeacherSource("full_finetuned", "run_name", 0.1, 0)
    split = CampaignSplit(
        Path("fold0.json"),
        "fold0-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [], "valid": [20]},
    }

    assert _verify_run_membership(provenance, split, "valid_only") == {
        "train": 0,
        "valid": 1,
    }
    provenance["fit_row_groups"]["valid"] = [21]
    with pytest.raises(ValueError, match="outside the clean valid pool"):
        _verify_run_membership(provenance, split, "eval_leak")


def validate_cache(cache_root: Path, data_root: Path, split: CampaignSplit) -> dict[str, object]:
    from wnn.distillation_cache import validate_distillation_cache

    return validate_distillation_cache(
        cache_root, data_root, split, "source", require_families=True
    )


def test_legacy_sequence_batch_width_is_unchanged_when_family_head_is_off() -> None:
    from scripts.train_rnn import Sequences

    sequence = Sequences("train", [0])
    expected_per_row = (N_FEATURES + 2) * np.dtype(np.float32).itemsize + np.dtype(
        np.bool_
    ).itemsize

    assert sequence.batch_tensor_bytes(1) == SEQUENCE_LENGTH * expected_per_row
