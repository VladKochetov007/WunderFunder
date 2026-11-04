from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.training.sequence_epoch_prediction import validate_index


def select_best_control_epoch(fit_dir: Path, split_sha256: str) -> dict[str, object]:
    """Select the highest logged train-only inner-control WP and bind its safe checkpoint."""
    evaluation_path = fit_dir / "monitor_evaluations.jsonl"
    if not evaluation_path.is_file() or evaluation_path.is_symlink():
        raise FileNotFoundError("fit is missing its producer-native evaluation journal")
    index_path = fit_dir / "epoch_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("all-epoch index is unavailable or unsafe")
    index = read_json(index_path)
    index_sha = index.get("index_sha256")
    if not isinstance(index_sha, str):
        raise ValueError("all-epoch index has no digest")
    validate_index(index, index_sha)
    identity = index.get("identity")
    identity_data = identity.get("data") if isinstance(identity, Mapping) else None
    control_members = (
        identity_data.get("evaluation_members") if isinstance(identity_data, Mapping) else None
    )
    fit_members = identity_data.get("fit_members") if isinstance(identity_data, Mapping) else None
    if (
        not isinstance(identity_data, Mapping)
        or identity_data.get("split_sha256") != split_sha256
        or identity_data.get("evaluation_role") != "control"
        or identity_data.get("evaluation_source") != "train"
        or identity_data.get("evaluation_mask") != "all_after_warmup"
        or not _source_members(control_members, source="train")
        or not _source_members(fit_members, source="train")
        or set(map(tuple, control_members)) & set(map(tuple, fit_members))
    ):
        raise ValueError("epoch index does not bind disjoint train-only fit/control memberships")
    member_count = len(control_members)
    member_sha256 = _members_sha256(control_members)
    selected_rows = member_count * (SEQUENCE_LENGTH - WARMUP)
    eligible: list[dict[str, object]] = []
    for line_number, line in enumerate(evaluation_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluation event on line {line_number}") from error
        if not isinstance(event, dict) or event.get("schema") != "wnn.monitor.v1":
            raise ValueError("evaluation journal contains an unsupported event")
        if event.get("kind") != "evaluation":
            continue
        context, metrics = event.get("context"), event.get("metrics")
        if not isinstance(context, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("evaluation event is missing context or metric values")
        control = context.get("control")
        if not isinstance(control, Mapping):
            raise ValueError("evaluation event does not identify the inner control cohort")
        if (
            control.get("scope") != "inner_train_proxy"
            or control.get("source") != "train"
            or control.get("role") != "control"
            or control.get("row_mask") != "all_after_warmup"
            or control.get("mask_definition")
            != "all rows after row 99 warmup; actual fitting objective mask"
            or ("is_scored_used" in control and control.get("is_scored_used") is not False)
            or control.get("split_sha256") != split_sha256
            or control.get("primary_split_sha256") != identity_data.get("primary_split_sha256")
            or control.get("member_count") != member_count
            or control.get("member_sha256") != member_sha256
        ):
            raise ValueError("control event does not match the train-only proxy policy")
        values = {
            name: metrics.get(name)
            for name in ("control/wp", "control/loss", "control/wp/t0", "control/wp/t1")
        }
        if not all(_finite_number(value) for value in values.values()):
            raise ValueError("control event is missing finite pooled WP/loss/target metrics")
        if control.get("selected_rows") != selected_rows:
            raise ValueError("control event does not cover every post-warmup row in its members")
        epoch, step = event.get("epoch"), event.get("step")
        if not _finite_number(epoch) or not _nonnegative_int(step):
            raise ValueError("control evaluation has invalid epoch or update count")
        eligible.append(
            {
                "epoch": int(epoch),
                "step": step,
                "wp": float(values["control/wp"]),
                "loss": float(values["control/loss"]),
                "t0": float(values["control/wp/t0"]),
                "t1": float(values["control/wp/t1"]),
                "selected_rows": selected_rows,
                "sequence_count": member_count,
                "event_sha256": canonical_json_sha256(event),
            }
        )
    if not eligible:
        raise ValueError("fit has no completed inner-control evaluation")
    best = max(eligible, key=lambda item: float(item["wp"]))
    entries = index.get("entries")
    if not isinstance(entries, list) or not isinstance(identity, Mapping):
        raise ValueError("all-epoch index is missing its identity or safe endpoint entries")
    index_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("epoch") == best["epoch"] - 1
    ]
    if len(index_entries) != 1:
        raise ValueError("selected control epoch has no unique immutable checkpoint")
    entry = dict(index_entries[0])
    if entry.get("global_step") != best["step"]:
        raise ValueError(
            "selected control event does not match its epoch checkpoint update boundary"
        )
    checkpoint = fit_dir / str(entry.get("checkpoint_path"))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("selected checkpoint is unavailable or unsafe")
    if file_sha256(checkpoint) != entry.get("checkpoint_sha256"):
        raise ValueError("selected checkpoint bytes differ from the epoch index")
    best.update(
        {
            "eligible_epoch_count": len(eligible),
            "selection_role": "inner_control_selected",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "epoch_index_path": str(index_path),
            "epoch_index_sha256": index_sha,
            "epoch_index_file_sha256": file_sha256(index_path),
            "identity_sha256": canonical_json_sha256(identity),
        }
    )
    return best


def validate_inner_objective(
    objective_path: Path,
    *,
    auxiliary_split_sha256: str,
    primary_split_sha256: str,
    expected_checkpoint_sha256: str,
    expected_identity_sha256: str,
) -> dict[str, object]:
    """Accept only the source-qualified train-only control score for the selected checkpoint."""
    objective = read_json(objective_path)
    if objective.get("schema") != "wnn.multistage-inner-objective.v1":
        raise ValueError("score output has an unsupported inner-objective schema")
    if objective.get("split_sha256") != auxiliary_split_sha256:
        raise ValueError("inner objective uses a different auxiliary split")
    if objective.get("primary_split_sha256") != primary_split_sha256:
        raise ValueError("inner objective does not bind the official primary split")
    if objective.get("model_sha256") != expected_checkpoint_sha256:
        raise ValueError("inner objective model bytes differ from the selected checkpoint")
    producer = objective.get("producer")
    provenance = producer.get("provenance") if isinstance(producer, Mapping) else None
    producer_identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    if not isinstance(producer_identity, Mapping):
        raise ValueError("inner objective must retain producer provenance identity")
    identity_sha256 = canonical_json_sha256(producer_identity)
    if identity_sha256 != expected_identity_sha256:
        raise ValueError("inner objective producer identity differs from the profiled recipe")
    if (
        producer.get("checkpoint_sha256") != expected_checkpoint_sha256
        or producer.get("split_sha256") != auxiliary_split_sha256
        or producer.get("fit_members_sha256") != objective.get("fit_members_sha256")
    ):
        raise ValueError("inner objective producer checkpoint, split, or fit membership differs")
    evaluation, metric = objective.get("evaluation"), objective.get("objective")
    if not isinstance(evaluation, Mapping) or not isinstance(metric, Mapping):
        raise ValueError("inner objective is missing evaluation or metric details")
    if (
        evaluation.get("role") != "control"
        or evaluation.get("scope") != "inner_train_proxy"
        or evaluation.get("source") != "train"
        or evaluation.get("row_mask") != "all_after_warmup AND row_index>=99"
        or evaluation.get("is_scored_used") is not False
    ):
        raise ValueError("objective must be a train-only all-after-warmup proxy")
    if metric.get("name") != "wp" or not _finite_number(metric.get("value")):
        raise ValueError("inner objective value is not a finite pooled WP")
    if metric.get("reduction") != "mean_of_two_pooled_clipped_weighted_Pearson_targets":
        raise ValueError("inner WP reduction differs from the frozen competition metric")
    for name in ("t0", "t1"):
        if not _finite_number(metric.get(name)):
            raise ValueError(f"inner objective omits finite target score {name}")
    if not _nonnegative_int(objective.get("selected_rows")) or objective["selected_rows"] <= 0:
        raise ValueError("inner objective selected-row count must be positive")
    if not _nonnegative_int(objective.get("sequence_count")) or objective["sequence_count"] <= 0:
        raise ValueError("inner objective sequence count must be positive")
    if not _is_sha256(objective.get("prediction_manifest_sha256")):
        raise ValueError("inner objective is missing the prediction-manifest digest")
    if not _is_sha256(objective.get("fit_members_sha256")) or not _is_sha256(
        objective.get("evaluation_members_sha256")
    ):
        raise ValueError("inner objective does not bind both source-qualified memberships")
    if (
        not isinstance(objective.get("interpretation"), str)
        or "not chronological validation" not in objective["interpretation"]
    ):
        raise ValueError("inner objective interpretation must retain its selection-only scope")
    objective["identity_sha256"] = identity_sha256
    return objective


def _finite_number(value: object) -> bool:
    """Accept finite metric numbers and reject Boolean values."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_int(value: object) -> bool:
    """Accept nonnegative integer counters and reject Boolean values."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    """Accept a lowercase 64-character SHA256 digest only."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_members(value: object, *, source: str) -> bool:
    """Require a nonempty unique ordered list of source-qualified row groups."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(member, list)
            and len(member) == 2
            and member[0] == source
            and type(member[1]) is int
            and member[1] >= 0
            for member in value
        )
        and len({tuple(member) for member in value}) == len(value)
    )


def _members_sha256(members: list[list[object]]) -> str:
    """Hash the ordered source-qualified membership in monitor journal format."""
    encoded = json.dumps(members, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
