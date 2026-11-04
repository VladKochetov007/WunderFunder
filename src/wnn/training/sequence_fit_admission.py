"""Common CLI admission and the opt-in native all-labeled final-fit branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.galam_adapter import validate_galam_batch_plan
from wnn.training.sequence_config import load_config, validate_expected_fold
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
    return True
