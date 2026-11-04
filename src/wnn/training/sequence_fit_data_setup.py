"""Route data construction through the ordinary or explicit final-fit schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.sequence_data import SequenceData, build_data
from wnn.training.sequence_final_fit import build_final_sequence_data


def build_fit_data(
    config: dict[str, Any],
    data_root: Path,
    split_manifest: Path | None,
    teacher_cache: Path | None,
    final_membership: FinalFitMembership | None,
    final_source_validation_receipt: Path | None,
) -> SequenceData:
    """Use a dedicated no-evaluation adapter only when final membership was admitted."""
    if final_membership is not None:
        if final_source_validation_receipt is None:
            raise ValueError("final-fit source-validation receipt path was not admitted")
        return build_final_sequence_data(
            config, data_root, final_membership, final_source_validation_receipt
        )
    if split_manifest is None:
        raise ValueError("ordinary sequence fitting requires --split-manifest")
    return build_data(config, data_root, split_manifest, teacher_cache)
