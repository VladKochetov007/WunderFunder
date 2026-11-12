from __future__ import annotations

from pathlib import Path
from typing import Any

from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_config import validate_expected_fold
from wnn.training.sequence_data import SequenceData, build_data
from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_pretraining import prepare_future_innovation_records


def prepare_sequence_fit_data(
    config: dict[str, Any],
    data_root: Path,
    split_manifest: Path,
    output: Path,
    context: DistributedContext,
    *,
    teacher_cache: Path | None = None,
    reference_cache: Path | None = None,
    monitor_reference_root: Path | None = None,
    resume: bool = False,
    profile_updates: int = 0,
) -> tuple[SequenceData, ComplementMonitor | None]:
    """Build canonical members, optional pretext targets, and any report-only monitor."""
    data = coordinated_call(
        lambda: build_data(config, data_root, split_manifest, teacher_cache, reference_cache),
        "sequence data construction",
        context,
    )
    if config.get("pretraining") is not None:
        coordinated_call(
            lambda: prepare_future_innovation_records(data, data_root),
            "train-only future-innovation data preparation",
            context,
        )
    coordinated_call(
        lambda: validate_expected_fold(config, data.split.fold),
        "split fold validation",
        context,
    )
    monitor = (
        coordinated_call(
            lambda: ComplementMonitor(
                data,
                monitor_reference_root,
                config,
                output,
                resume=resume,
                primary=context.is_primary,
                report_epochs=not profile_updates,
            ),
            "report-only monitor setup",
            context,
        )
        if "monitoring" in config
        else None
    )
    return data, monitor
