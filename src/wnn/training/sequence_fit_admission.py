"""Common CLI admission and the opt-in native all-labeled final-fit branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wnn.final_submit_manifest import FinalFitMembership
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.galam_adapter import validate_galam_batch_plan
from wnn.training.sequence_config import load_config, validate_expected_fold
from wnn.training.sequence_config_validation import (
    STANDALONE_LABELS_OBJECTIVE_MODE,
)
from wnn.training.sequence_data import SequenceData
from wnn.training.sequence_final_fit import (
    load_final_sequence_membership,
    validate_final_lstm_recipe,
)


@dataclass(frozen=True)
class FitAdmission:
    """Resolved config, paths, and optional final-fit identity for one trainer run."""

    config: dict[str, Any]
    output: Path
    resume_path: Path | None
    receipt_path: Path | None
    initializer_path: Path | None
    final_membership: FinalFitMembership | None
    final_source_validation_receipt: Path | None


def prepare_fit_admission(
    config_path: Path,
    output: Path,
    split_manifest: Path | None,
    teacher_cache: Path | None,
    resume: Path | None,
    profile_updates: int,
    profile_receipt: Path | None,
    initializer: Path | None,
    final_fit_manifest: Path | None,
    final_fit_source_validation_receipt: Path | None,
    final_fit_source_validation_sha256: str | None,
    data_root: Path,
) -> FitAdmission:
    """Validate profile/resume controls and resolve ordinary or final-fit admission."""
    config = load_config(config_path)
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    if config.get("objective_mode") == STANDALONE_LABELS_OBJECTIVE_MODE:
        if teacher_cache is not None or initializer is not None:
            raise ValueError("standalone labels objective forbids teacher caches and initializers")
        if any(
            value is not None
            for value in (
                final_fit_manifest,
                final_fit_source_validation_receipt,
                final_fit_source_validation_sha256,
            )
        ):
            raise ValueError("standalone labels objective requires the official primary split")
    output = output.resolve()
    resume_path = resume.resolve() if resume is not None else None
    receipt_path = profile_receipt.resolve() if profile_receipt is not None else None
    initializer_path = initializer.resolve() if initializer is not None else None
    if initializer_path is not None and resume_path is not None:
        raise ValueError("fresh --initializer and exact-state --resume are mutually exclusive")
    if resume_path is not None and resume_path != output / "training_state.pt":
        raise ValueError("--resume must name <output>/training_state.pt")
    if profile_updates < 0 or (profile_updates and (resume_path or receipt_path)):
        raise ValueError("profile runs cannot resume or consume a profile admission receipt")
    if profile_updates and "monitoring" in config:
        diagnostic_interval = int(config["monitoring"]["diagnostics_every_steps"])
        if profile_updates < diagnostic_interval:
            raise ValueError(
                "profile updates must reach diagnostics_every_steps to exercise sampled hooks"
            )
    if not profile_updates and receipt_path is None:
        raise ValueError("a full fit requires --profile-receipt with explicit approval")
    options = (
        final_fit_manifest,
        final_fit_source_validation_receipt,
        final_fit_source_validation_sha256,
    )
    final_membership = None
    if any(value is not None for value in options):
        if any(value is None for value in options):
            raise ValueError(
                "native final fit requires manifest, source-validation receipt path, and SHA256"
            )
        validate_final_lstm_recipe(
            config,
            split_manifest=split_manifest,
            teacher_cache=teacher_cache,
            initializer=initializer_path,
            resume=resume_path,
        )
        final_membership = load_final_sequence_membership(
            final_fit_manifest,
            final_fit_source_validation_receipt,
            final_fit_source_validation_sha256,
            data_root,
        )
    elif split_manifest is None:
        raise ValueError("ordinary sequence fitting requires --split-manifest")
    return FitAdmission(
        config,
        output,
        resume_path,
        receipt_path,
        initializer_path,
        final_membership,
        final_fit_source_validation_receipt,
    )


def validate_fit_data_contract(config: dict[str, Any], data: SequenceData, world_size: int) -> bool:
    """Keep batch-shape and split-fold validation on the shared trainer path."""
    validate_galam_batch_plan(config, len(data.fit), world_size)
    validate_expected_fold(config, data.split.fold)
    if config.get("objective_mode") == STANDALONE_LABELS_OBJECTIVE_MODE:
        _record_standalone_labels_contract(config, data)
    return True


def _record_standalone_labels_contract(config: dict[str, Any], data: SequenceData) -> None:
    """Require qualified primary data and publish the fixed WMSE ancestry contract."""
    split = data.split
    expected_fit = [("train", group) for group in split.train_groups]
    expected_evaluation = [("valid", group) for group in split.evaluation_groups]
    normalization = data.provenance.get("normalization")
    model_config = config.get("model")
    valid_data = (
        split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        and split.train_groups
        and not split.valid_groups
        and split.evaluation_source == "valid"
        and split.evaluation_role == "valid"
        and split.evaluation_mask == "is_scored"
        and data.fit.members == expected_fit
        and data.evaluation.members == expected_evaluation
        and data.fit.training_mask == "all_after_warmup"
        and data.evaluation.training_mask == "is_scored"
        and data.provenance.get("split_sha256") == split.sha256
        and data.provenance.get("fit_members") == [list(member) for member in expected_fit]
        and data.provenance.get("evaluation_members")
        == [list(member) for member in expected_evaluation]
        and isinstance(normalization, dict)
        and normalization.get("scope") == "all_fit"
        and normalization.get("sequence_count") == len(expected_fit)
        and "teacher" in data.provenance
        and data.provenance["teacher"] is None
        and data.fit.cache is None
        and data.fit.hint_members is None
        and data.fit.hint_dim == 0
        and "hint_members" in data.provenance
        and data.provenance["hint_members"] is None
        and "initializer" not in data.provenance
        and config.get("training_mask") == "all_after_warmup"
        and config.get("normalization_scope") == "all_fit"
        and isinstance(model_config, dict)
        and type(model_config.get("hint_dim")) is int
        and model_config.get("hint_dim") == 0
        and config.get("hint_mode") == "aligned"
        and type(config.get("kd_weight")) in (int, float)
        and type(config.get("hint_weight")) in (int, float)
        and config.get("kd_weight") == 0
        and config.get("hint_weight") == 0
    )
    if not valid_data:
        raise ValueError(
            "standalone labels objective requires full official-valid primary evaluation "
            "and fitting-only label ancestry"
        )
    data.provenance["objective_contract"] = {
        "kind": "standalone_label_weighted_mse",
        "mode": STANDALONE_LABELS_OBJECTIVE_MODE,
        "target_clip": [-2.0, 2.0],
        "target_weight": "abs(clipped_target)",
        "aggregation": "global_ddp_weighted_wmse_per_actual_optimizer_window",
        "pearson_term": "none",
        "reference_training_ancestry": "none",
    }
    data.provenance["objective_admission"] = {
        "split_schema": split.schema,
        "fit_membership": "official_train_fit_only",
        "evaluation_source": split.evaluation_source,
        "evaluation_role": split.evaluation_role,
        "evaluation_mask": split.evaluation_mask,
        "training_mask": config["training_mask"],
        "normalization_scope": config["normalization_scope"],
        "normalization_membership": "all_fitting_members",
        "teacher_source": "none",
        "initializer_source": "none",
    }
