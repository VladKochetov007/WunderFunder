"""Prepared receiving-host checks; do not infer execution from this file's presence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.sequence_cache import TeacherCache, validate_export_schema, validate_producer
from wnn.training.sequence_config import load_config
from wnn.training.sequence_data import SequenceRecords, permuted_hint_members
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_provenance import validate_profile_receipt


def test_short_profile_admission_accepts_zero_completed_epochs(tmp_path: Path) -> None:
    identity = {"world_size": 1, "model_parameters": {"total": 4, "trainable": 4}}
    receipt = {
        "receipt_schema": "wnn.sequence-profile.v1",
        "status": "profile_complete",
        "approved": True,
        "identity": identity,
        "updates": 8,
        "forwarded_rows": 8192,
        "real_rows": 8192,
        "valid_rows": 8093,
        "epochs_traversed": 0,
        "elapsed_seconds": 1.0,
        "scientific_checkpoint_written": False,
        "errors": [],
        "error_count": 0,
        "losses": {"label": 1.0, "distillation": 1.0, "hint": 1.0, "total": 1.0},
        "model_parameters": identity["model_parameters"],
        "memory_by_rank": [
            {
                "rank": 0,
                "peak_allocated_bytes": 10,
                "peak_reserved_bytes": 20,
                "device_total_memory_bytes": 100,
                "host_peak_rss_bytes": 10,
            }
        ],
        "peak_memory_bytes": 10,
        "peak_reserved_bytes": 20,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(receipt))
    assert validate_profile_receipt(path, identity) == file_sha256(path)
    receipt["epochs_traversed"] = -1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completed-epoch"):
        validate_profile_receipt(path, identity)


def _split() -> CampaignSplit:
    return CampaignSplit(
        Path("split.json"),
        "a" * 64,
        0,
        4,
        (0, 1),
        (1,),
        (0,),
        (("train", 0, 0), ("train", 1, 1), ("valid", 1, 1), ("valid", 0, 2)),
    )


def _producer() -> dict[str, object]:
    return {
        "split_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "fit_members": [["train", 0], ["train", 1], ["valid", 1]],
        "teacher": None,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"precision": "bf16"},
        {"chunk_rows": 0},
        {"lr": float("nan")},
        {"model": {"hidden": 129, "heads": 4}},
        {"hint_mode": "component_permuted", "hint_weight": 0},
    ],
)
def test_recipe_rejects_unsupported_or_incoherent_options(tmp_path: Path, change: dict) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(change))
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_padding_is_all_zero_and_real_warmup_is_unscored(tmp_path: Path) -> None:
    records = SequenceRecords(tmp_path, [("train", 3)], 4)
    padded = records[-1]
    assert not padded["mask"].any()
    assert all(torch.count_nonzero(value) == 0 for value in padded.values())

    class FakeSource:
        def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
            assert index == 0
            return (
                torch.ones(SEQUENCE_LENGTH, N_FEATURES),
                torch.ones(SEQUENCE_LENGTH, 2),
                torch.ones(SEQUENCE_LENGTH, dtype=torch.bool),
            )

    records.sources["train"] = FakeSource()
    sample = records[0]
    assert not sample["mask"][:WARMUP].any()
    assert sample["mask"][WARMUP:].all()
    assert sample["x"][0, 0] == 1  # Warm-up inputs are forwarded, not zeroed.


def test_hint_control_has_no_same_component_and_is_repeatable() -> None:
    split = _split()
    members = [("train", 0), ("train", 1), ("valid", 1)]
    first = permuted_hint_members(members, split, 1001)
    assert first == permuted_hint_members(members, split, 1001)
    component = {(s, g): c for s, g, c in split.components}
    assert all(
        component[left] != component[right] for left, right in zip(members, first, strict=True)
    )
    assert set(first) <= set(members)


def test_ancestor_fold_mismatch_is_rejected() -> None:
    producer = _producer()
    producer["teacher"] = {"producer": _producer() | {"split_sha256": "c" * 64}}
    with pytest.raises(ValueError, match="crosses"):
        validate_producer(producer, "a" * 64, [("train", 0), ("train", 1), ("valid", 1)])


def test_export_schema_rejects_evaluation_cache() -> None:
    with pytest.raises(ValueError, match="membership"):
        validate_export_schema(
            {
                "input_schema": "raw112-v1",
                "membership": "evaluation",
                "sequence_length": SEQUENCE_LENGTH,
                "hint_dim": 64,
            },
            "fit",
        )


def test_teacher_cache_hash_and_source_identity(tmp_path: Path) -> None:
    split = _split()
    members = [("train", 0), ("train", 1), ("valid", 1)]
    entries = []
    for position, (source, group) in enumerate(members):
        path = tmp_path / f"{source}-{group}.npy"
        np.save(path, np.full((SEQUENCE_LENGTH, 4), position, dtype=np.float32))
        entries.append(
            {
                "split": source,
                "row_group": group,
                "file": path.name,
                "sha256": file_sha256(path),
            }
        )
    metadata = {
        "schema": "sequence-teacher-cache-v1",
        "complete": True,
        "input_schema": "raw112-v1",
        "membership": "fit",
        "sequence_length": SEQUENCE_LENGTH,
        "split_sha256": split.sha256,
        "hint_dim": 2,
        "producer": _producer(),
        "sequences": entries,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    cache = TeacherCache(tmp_path, split, members)
    prediction, hints = cache.read("valid", 1, 2)
    assert torch.all(prediction == 2) and torch.all(hints == 2)
    assert torch.all(cache.read("train", 1, 0)[0] == 1)
    with (tmp_path / entries[0]["file"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        TeacherCache(tmp_path, split, members)


def test_accumulation_normalizes_global_mass_not_microbatch_means() -> None:
    target = torch.tensor([[[0.1, 0.2], [1.8, -2.0], [0.5, 0.3]]])
    mask = torch.tensor([[True, True, False]])
    teacher = target * 0.5
    hint_target = torch.ones(1, 3, 3)
    prediction = torch.zeros_like(target, requires_grad=True)
    hints = torch.zeros_like(hint_target, requires_grad=True)
    denominators = window_denominators(target, mask, hint_target)
    reference = scaled_objective(
        objective_numerators(prediction, hints, target, teacher, hint_target, mask),
        denominators,
        1,
        1.0,
        0.1,
    )
    expected = torch.autograd.grad(reference, (prediction, hints))
    parts = []
    for start, end in ((0, 1), (1, 3)):
        parts.append(
            scaled_objective(
                objective_numerators(
                    prediction[:, start:end],
                    hints[:, start:end],
                    target[:, start:end],
                    teacher[:, start:end],
                    hint_target[:, start:end],
                    mask[:, start:end],
                ),
                denominators,
                1,
                1.0,
                0.1,
            )
        )
    actual = torch.autograd.grad(sum(parts), (prediction, hints))
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right)
    assert torch.count_nonzero(actual[0][:, 2:]) == 0
