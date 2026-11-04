"""Source-bound all-labeled final-fit admission for the native recurrent trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wnn.final_submit_manifest import (
    FINAL_FIT_SCHEMA,
    FinalFitMembership,
    load_final_fit_membership,
)
from wnn.splits import CampaignSplit
from wnn.training.final_submit_bridge import (
    SOURCE_VALIDATION_SHA256,
    final_fit_identity,
    validate_source_validation_receipt,
)
from wnn.training.sequence_data import SequenceData, SequenceRecords
from wnn.training.sequence_features import RAW_INPUT_MODE, sequence_input_schema


def load_final_sequence_membership(
    manifest: Path,
    source_validation_receipt: Path,
    source_validation_sha256: str,
    data_root: Path,
) -> FinalFitMembership:
    """Load the pinned complete-fit contract and its reviewed receiver receipt."""
    if source_validation_sha256 != SOURCE_VALIDATION_SHA256:
        raise ValueError("native final fit requires the reviewed source-validation receipt")
    membership = load_final_fit_membership(manifest, data_root, source_validation_sha256)
    validate_source_validation_receipt(
        source_validation_receipt, source_validation_sha256, membership
    )
    return membership


def validate_final_lstm_recipe(
    config: dict[str, Any],
    *,
    split_manifest: Path | None,
    teacher_cache: Path | None,
    initializer: Path | None,
    resume: Path | None,
) -> None:
    """Admit only the fixed Source4 LSTM192x1 objective under final-fit data."""
    model = config.get("model")
    expected = {
        "architecture": "lstm",
        "hidden": 192,
        "layers": 1,
        "hint_dim": 0,
        "input_feature_mode": RAW_INPUT_MODE,
        "dropout": 0.0,
        "compile_blocks": False,
    }
    if not isinstance(model, dict) or any(
        model.get(key, value) != value for key, value in expected.items()
    ):
        raise ValueError("final native mode requires the Source4 raw112 LSTM192x1 model")
    recipe = {
        "epochs": 8,
        "batch_size": 256,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "seed": 1302,
        "optimizer": "adamw",
        "lr": 0.0015008627097145002,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "adamw_eps": 1e-8,
        "precision_mode": "fp32",
        "schedule": "legacy_warmup_cosine",
        "min_lr_ratio": 0.0,
        "training_mask": "all_after_warmup",
        "normalization_scope": "prefix",
        "normalization_sequences": 32,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
    }
    if (
        any(config.get(key, value) != value for key, value in recipe.items())
        or config.get("input_prefetch_batches", 1) != 1
    ):
        raise ValueError("final native mode differs from the reviewed fixed Source4 LSTM recipe")
    if (
        config.get("checkpoint_retention") != "all_epochs"
        or "monitoring" in config
        or config.get("expected_fold") is not None
        or split_manifest is not None
        or teacher_cache is not None
        or initializer is not None
        or resume is not None
    ):
        raise ValueError(
            "final native mode requires all-epoch checkpoints and forbids monitoring, folds, teachers, initializers, and resume"
        )


def build_final_sequence_data(
    config: dict[str, Any],
    data_root: Path,
    membership: FinalFitMembership,
    source_validation_receipt: Path,
) -> SequenceData:
    """Build 12,480 fitting records, fit-prefix normalization, and no evaluation rows."""
    members = [("train", group) for group in membership.train_groups] + [
        ("valid", group) for group in membership.valid_groups
    ]
    if len(members) != 12_480 or ("train", 528) not in members:
        raise ValueError("final native fitting membership is incomplete or omits row group 528")
    split = CampaignSplit(
        membership.path,
        membership.manifest_sha256,
        0,
        1,
        membership.train_groups,
        membership.valid_groups,
        (),
        (),
        schema=FINAL_FIT_SCHEMA,
        evaluation_source="none",
        evaluation_role="none",
        evaluation_mask="none",
        primary_split_sha256=None,
    )
    fit = SequenceRecords(
        data_root,
        members,
        0,
        training_mask="all_after_warmup",
        input_feature_mode=RAW_INPUT_MODE,
    )
    fit.validate_source_schema()
    evaluation = SequenceRecords(
        data_root,
        [],
        0,
        training_mask="all_after_warmup",
        input_feature_mode=RAW_INPUT_MODE,
    )
    evaluation.validate_source_schema()
    normalization_members = members[:32]
    normalization_sources = {
        "datasets/train.parquet": {"sha256": membership.train_sha256},
        "datasets/valid.parquet": {"sha256": membership.valid_sha256},
    }
    final_identity = final_fit_identity(membership)
    provenance = {
        "split_sha256": membership.manifest_sha256,
        "split_schema": FINAL_FIT_SCHEMA,
        "final_fit": final_identity,
        "final_fit_source_validation_receipt": {
            "path": str(source_validation_receipt.resolve()),
            "sha256": membership.source_validation_sha256,
        },
        "sources": normalization_sources,
        "fit_members": [list(member) for member in members],
        "fit_row_groups": {
            "train": list(membership.train_groups),
            "valid": list(membership.valid_groups),
        },
        "evaluation_members": [],
        "evaluation_row_groups": [],
        "evaluation_role": "none",
        "evaluation_source": "none",
        "evaluation_mask": "none",
        "primary_split_sha256": None,
        "input_schema": sequence_input_schema(RAW_INPUT_MODE),
        "mask": "all_after_warmup AND row_index>=99",
        "training_mask": "all_after_warmup",
        "normalization": {
            "scope": "prefix",
            "sequence_count": len(normalization_members),
            "fit_prefix_sequences": len(normalization_members),
            "source_fingerprints": normalization_sources,
        },
        "teacher": None,
        "hint_mode": config["hint_mode"],
        "hint_members": None,
        "independent_score_claim": "forbidden",
    }
    return SequenceData(fit, evaluation, split, provenance, 32)
