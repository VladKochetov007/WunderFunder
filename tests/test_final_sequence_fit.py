from __future__ import annotations

from pathlib import Path

import pytest

from wnn.chronological_splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.final_submit_manifest import FINAL_FIT_SCHEMA, FinalFitMembership
from wnn.training import sequence_final_fit as final_fit


def _config() -> dict[str, object]:
    return {
        "epochs": 8,
        "batch_size": 256,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "seed": 1302,
        "optimizer": "adamw",
        "lr": 0.0015008627097145002,
        "weight_decay": 0.01,
        "precision_mode": "fp32",
        "schedule": "legacy_warmup_cosine",
        "min_lr_ratio": 0.0,
        "training_mask": "all_after_warmup",
        "normalization_scope": "prefix",
        "normalization_sequences": 32,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "hint_mode": "aligned",
        "checkpoint_retention": "all_epochs",
        "expected_fold": None,
        "model": {
            "architecture": "lstm",
            "hidden": 192,
            "layers": 1,
            "hint_dim": 0,
            "input_feature_mode": "raw112",
            "dropout": 0.0,
            "compile_blocks": False,
        },
    }


def _membership(path: Path) -> FinalFitMembership:
    return FinalFitMembership(
        path=path,
        manifest_sha256="a" * 64,
        fit_members_sha256="b" * 64,
        train_groups=tuple(range(10_607)),
        valid_groups=tuple(range(1_873)),
        train_sha256="c" * 64,
        valid_sha256="d" * 64,
        source_validation_sha256="e" * 64,
    )


def test_source4_final_lstm_admission_is_fixed_and_no_eval() -> None:
    final_fit.validate_final_lstm_recipe(
        _config(),
        split_manifest=None,
        teacher_cache=None,
        initializer=None,
        resume=None,
    )
    invalid = _config()
    invalid["monitoring"] = {}
    with pytest.raises(ValueError, match="forbids monitoring"):
        final_fit.validate_final_lstm_recipe(
            invalid,
            split_manifest=None,
            teacher_cache=None,
            initializer=None,
            resume=None,
        )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("model", {"architecture": "gru", "hidden": 192, "layers": 1}),
        ("epochs", 9),
        ("training_mask", "is_scored"),
        ("kd_weight", 0.1),
        ("checkpoint_retention", "final_only"),
    ],
)
def test_source4_final_lstm_rejects_recipe_drift(change: str, value: object) -> None:
    config = _config()
    config[change] = value
    with pytest.raises(ValueError):
        final_fit.validate_final_lstm_recipe(
            config,
            split_manifest=None,
            teacher_cache=None,
            initializer=None,
            resume=None,
        )


def test_final_sequence_data_has_all_12480_members_and_no_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[object] = []

    class FakeRecords:
        def __init__(self, root: Path, members: list[tuple[str, int]], hint_dim: int, **kwargs):
            self.root = root
            self.members = list(members)
            self.hint_dim = hint_dim
            self.training_mask = kwargs["training_mask"]
            self.input_feature_mode = kwargs["input_feature_mode"]
            created.append(self)

        def validate_source_schema(self) -> None:
            return None

    monkeypatch.setattr(final_fit, "SequenceRecords", FakeRecords)
    membership = _membership(tmp_path / "final-fit.json")
    receipt = tmp_path / "SOURCE_VALIDATION_R1.json"
    data = final_fit.build_final_sequence_data(_config(), tmp_path, membership, receipt)

    assert len(data.fit.members) == 12_480
    assert data.fit.members[0] == ("train", 0)
    assert data.fit.members[10_607] == ("valid", 0)
    assert ("train", 528) in data.fit.members
    assert data.fit.training_mask == "all_after_warmup"
    assert data.evaluation.members == []
    assert data.split.schema == FINAL_FIT_SCHEMA
    assert data.split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
    assert data.split.evaluation_groups == ()
    assert data.split.evaluation_source == "none"
    assert data.provenance["evaluation_members"] == []
    assert data.provenance["evaluation_source"] == "none"
    assert data.provenance["evaluation_role"] == "none"
    assert data.provenance["independent_score_claim"] == "forbidden"
    assert data.provenance["normalization"]["scope"] == "prefix"
    assert data.provenance["normalization"]["fit_prefix_sequences"] == 32
    assert data.provenance["final_fit_source_validation_receipt"]["path"] == str(receipt.resolve())
    assert len(created) == 2


def test_final_lstm_admission_rejects_teachers_initializers_resume_and_fold(
    tmp_path: Path,
) -> None:
    for option in (
        {"teacher_cache": tmp_path / "cache"},
        {"initializer": tmp_path / "init.pt"},
        {"resume": tmp_path / "state.pt"},
        {"split_manifest": tmp_path / "split.json"},
    ):
        kwargs = {
            "split_manifest": None,
            "teacher_cache": None,
            "initializer": None,
            "resume": None,
        }
        kwargs.update(option)
        with pytest.raises(ValueError):
            final_fit.validate_final_lstm_recipe(_config(), **kwargs)

    config = _config()
    config["expected_fold"] = 0
    with pytest.raises(ValueError, match="forbids monitoring, folds"):
        final_fit.validate_final_lstm_recipe(
            config,
            split_manifest=None,
            teacher_cache=None,
            initializer=None,
            resume=None,
        )
