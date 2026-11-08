from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
)
from wnn.training.sequence_data import SequenceData, SequenceRecords
from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_budget import (
    MonitorEventBudget,
    validate_monitor_event_budget,
    validate_profile_monitor_event_budget,
)
from wnn.training.sequence_monitor_events import (
    MONITOR_EVALUATION_FILENAME,
    MONITOR_EVENT_FILENAME,
)
from wnn.training.sequence_monitor_identity import (
    TRAIN_EVAL_HASH_RANK_NAMESPACE,
    hash_rank_train_eval_members,
    membership_sha256,
    train_eval_comparability,
)
from wnn.training.sequence_provenance import canonical_sha256


def select_train_eval_members(monitor_config: object, data: SequenceData) -> list[tuple[str, int]]:
    """Select a fitting-only diagnostic cohort without interpreting sequence IDs as time."""
    if monitor_config is None:
        return []
    if not isinstance(monitor_config, Mapping):
        raise TypeError("validated monitoring config must remain a mapping")
    if data.split.schema not in (
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
        MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    ):
        raise ValueError("online monitoring requires a primary or auxiliary source-bound split")
    requested = int(monitor_config["train_eval_sequences"])
    selection = str(monitor_config.get("train_eval_selection", "prefix"))
    if selection == "prefix":
        members = list(data.fit.members[:requested])
    elif selection == "hash_rank_v1":
        members = hash_rank_train_eval_members(data.fit.members)[:requested]
    else:
        raise ValueError(f"unsupported train-eval selection: {selection}")
    if not members or any(member not in data.fit.members for member in members):
        raise ValueError("fixed train-eval cohort is not a subset of actual fit membership")
    evaluation_members = set(getattr(data.evaluation, "members", ()))
    if evaluation_members.intersection(members):
        raise ValueError("train-eval diagnostic must not contain official evaluation members")
    return members


def prepare_monitor_event_budgets(
    monitor_config: object,
    schedule_updates: int,
    epochs: int,
    profile_updates: int,
) -> tuple[MonitorEventBudget | None, MonitorEventBudget | None]:
    """Preflight the complete fit journal and the smaller profile journal separately."""
    if not isinstance(monitor_config, Mapping):
        return None, None
    if "training_telemetry_every_steps" not in monitor_config:
        return None, None
    training_every_steps = int(monitor_config["training_telemetry_every_steps"])
    diagnostics_every_steps = int(monitor_config["diagnostics_every_steps"])
    full_budget = validate_monitor_event_budget(
        schedule_updates, epochs, training_every_steps, diagnostics_every_steps
    )
    profile_budget = None
    if profile_updates:
        profile_budget = validate_profile_monitor_event_budget(
            profile_updates,
            schedule_updates // epochs,
            training_every_steps,
            diagnostics_every_steps,
        )
    return full_budget, profile_budget


def record_monitoring_plan(
    data: SequenceData,
    monitor_config: Mapping[str, object],
    train_eval_members: list[tuple[str, int]],
    input_feature_mode: str,
    event_budget: MonitorEventBudget | None,
) -> None:
    """Record mask, selection, and event-journal policy in fit provenance."""
    early_policy = monitor_config.get("early_stopping")
    early_action = (
        str(early_policy.get("action", "stop")) if isinstance(early_policy, Mapping) else "disabled"
    )
    evaluation_label = (
        "full official valid" if data.split.evaluation_role == "valid" else "inner control"
    )
    if early_action == "report_only":
        early_semantics = (
            "report-only action: min_delta resets the advisory patience counter only; the plateau "
            "threshold is recorded without stopping, and all configured epochs/checkpoints are retained"
        )
    elif early_action == "stop":
        early_semantics = (
            "min_delta resets patience only; count consecutive evaluation checks from the first "
            "check; stopping is eligible after min_epoch; scheduler horizon remains the configured "
            "epoch cap"
        )
    else:
        early_semantics = "no early-stopping or plateau policy configured"
    plan: dict[str, object] = {
        "evaluation_reporting": (
            "full official valid after each epoch; development-exposed report"
            if data.split.evaluation_role == "valid"
            else "inner train control after each epoch; selection proxy, not primary validation"
        ),
        "train_eval_mask": monitor_config.get("train_eval_mask", "is_scored"),
        "evaluation_role": data.split.evaluation_role,
        "evaluation_source": data.split.evaluation_source,
        "evaluation_mask": data.split.evaluation_mask,
        "input_feature_mode": input_feature_mode,
        "checkpoint_selection": (
            f"best-observed {evaluation_label} WP is exposed, role-bound and exploratory; "
            "the best pointer is not an independent or fixed inference endpoint"
        ),
        "early_stopping": early_policy,
        "early_stopping_semantics": early_semantics,
        "train_eval_members": [list(member) for member in train_eval_members],
        "train_eval_selection": monitor_config.get("train_eval_selection", "prefix"),
        "train_eval_selection_algorithm": (
            TRAIN_EVAL_HASH_RANK_NAMESPACE
            if monitor_config.get("train_eval_selection", "prefix") == "hash_rank_v1"
            else "legacy_fit_membership_prefix"
        ),
        "train_eval_selection_order_basis": (
            "SHA256 rank of opaque source-qualified group keys; no temporal order inferred"
            if monitor_config.get("train_eval_selection", "prefix") == "hash_rank_v1"
            else "source-qualified fit-membership order; no temporal order inferred"
        ),
        "train_eval_member_sha256": membership_sha256(train_eval_members),
        "train_eval_comparability": train_eval_comparability(
            str(monitor_config.get("train_eval_mask", "is_scored")),
            str(data.split.evaluation_mask),
            str(data.split.evaluation_role),
            str(monitor_config.get("train_eval_selection", "prefix")),
            len(train_eval_members),
            len(data.evaluation.members),
        ),
        "event_files": [MONITOR_EVENT_FILENAME, MONITOR_EVALUATION_FILENAME],
    }
    if event_budget is not None:
        plan["event_journal_budget"] = event_budget.as_dict()
    data.provenance["monitoring_plan"] = plan


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
                evaluation_role=data.split.evaluation_role,
                evaluation_mask=data.split.evaluation_mask,
                primary_split_sha256=data.split.primary_split_sha256,
            )
        return True

    coordinated_call(build_monitor, "monitor event stream setup", context)
    return train_eval, monitor_holder[0]
