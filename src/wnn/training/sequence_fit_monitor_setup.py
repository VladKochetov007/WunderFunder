from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.sequence_data import SequenceData, SequenceRecords
from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_provenance import canonical_sha256


def select_train_eval_members(monitor_config: object, data: SequenceData) -> list[tuple[str, int]]:
    """Select the fixed source-qualified prefix from actual chronological fit membership."""
    if monitor_config is None:
        return []
    if not isinstance(monitor_config, Mapping):
        raise TypeError("validated monitoring config must remain a mapping")
    if data.split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        raise ValueError("online monitoring requires the official chronological primary split")
    requested = int(monitor_config["train_eval_sequences"])
    members = list(data.fit.members[:requested])
    if not members or any(member not in data.fit.members for member in members):
        raise ValueError("fixed train-eval cohort is not a subset of actual fit membership")
    return members


def initialize_monitor(
    data_root: Path,
    output: Path,
    monitor_config: Mapping[str, object] | None,
    train_eval_members: list[tuple[str, int]],
    data: SequenceData,
    identity: dict[str, object],
    resume: bool,
    context: DistributedContext,
    input_feature_mode: str = "raw112",
) -> tuple[SequenceDataset | None, SequenceMonitor | None]:
    """Build the declared train-eval cohort and rank-zero append-only monitor writer."""
    if monitor_config is None:
        return None, None
    train_eval_mask = monitor_config.get("train_eval_mask", "is_scored")
    if train_eval_mask not in ("is_scored", "all_after_warmup"):
        raise ValueError("monitored train-eval has an unsupported row mask")
    if train_eval_mask != data.fit.training_mask:
        raise ValueError("train-eval row mask must match the actual fitting objective mask")

    def build_train_eval() -> SequenceRecords:
        """Read fitting sequences with their actual all-after-warmup training mask."""
        cohort = SequenceRecords(
            data_root,
            train_eval_members,
            0,
            training_mask=str(train_eval_mask),
            input_feature_mode=input_feature_mode,
        )
        cohort.validate_source_schema()
        return cohort

    train_eval = coordinated_call(build_train_eval, "monitor train-eval cohort setup", context)
    monitor_holder: list[SequenceMonitor | None] = [None]

    def build_monitor() -> bool:
        """Create the shared event writer on rank zero only."""
        if context.is_primary:
            monitor_holder[0] = SequenceMonitor(
                output,
                monitor_config,
                train_eval_members,
                list(data.evaluation.members),
                data.split.sha256,
                canonical_sha256(identity),
                resume=resume,
                input_feature_mode=input_feature_mode,
            )
        return True

    coordinated_call(build_monitor, "monitor event stream setup", context)
    return train_eval, monitor_holder[0]
