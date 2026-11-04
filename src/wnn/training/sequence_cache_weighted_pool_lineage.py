"""Provenance validation for heterogeneous fitting-only teacher pools."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from wnn.training.sequence_cache_ensemble_provenance import (
    canonical_sha256,
    common_provenance,
)

WEIGHTED_POOL_KIND = "weighted_teacher_pool_v1"
WEIGHTED_POOL_METHOD = "ordered_fp32_weighted_sum"
WEIGHTED_POOL_TARGET_CONTRACT = {
    "formula": "sequential float32 sum(weight_i * prediction_i) in parent order",
    "parent_clipping": "none",
    "pool_clipping": "none",
    "student_target_clipping": "clamp pooled teacher target to [-2,2] in weighted MSE",
    "student_objective": "label_loss + kd_weight * teacher_loss",
}
_PARENT_REFERENCE_FIELDS = (
    "trial_id",
    "weight",
    "manifest_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "producer_identity_sha256",
    "producer_sha256",
    "provenance_sha256",
)


def _is_sha256(value: object) -> bool:
    """Check lowercase SHA256 hex strings without accepting lookalikes."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_initializer(
    initializer: object,
    split_sha256: str,
    primary_split_sha256: str,
    fit_members: list[list[object]],
    evaluation_members: list[list[object]],
    expected_model: Mapping[str, object],
    expected_sources: Mapping[str, object] | None,
) -> None:
    """Require an initializer to bind its direct label-only parent and exact cohorts."""
    if initializer is None:
        return
    if not isinstance(initializer, Mapping):
        raise ValueError("teacher initializer ancestry is malformed")
    parent = initializer.get("parent_producer")
    if not isinstance(parent, dict) or not isinstance(parent.get("provenance"), dict):
        raise ValueError("teacher initializer omits its direct parent producer receipt")
    parent_provenance = parent["provenance"]
    parent_config = parent.get("config")
    parent_identity = parent_provenance.get("identity")
    parent_data = parent_identity.get("data") if isinstance(parent_identity, dict) else None
    if (
        initializer.get("parent_producer_sha256") != canonical_sha256(parent)
        or initializer.get("checkpoint_sha256") != parent.get("checkpoint_sha256")
        or initializer.get("parent_identity_sha256") != canonical_sha256(parent_identity)
        or initializer.get("parent_config_sha256") != canonical_sha256(parent_config)
        or not isinstance(parent_config, dict)
        or not isinstance(parent_data, dict)
        or parent.get("kind") not in (None, "single_checkpoint")
        or parent.get("teacher") is not None
        or parent_config.get("precision_mode") != "fp32"
        or type(parent_config.get("kd_weight")) not in (int, float)
        or parent_config["kd_weight"] != 0
        or type(parent_config.get("hint_weight")) not in (int, float)
        or parent_config["hint_weight"] != 0
        or not isinstance(parent_config.get("model"), dict)
        or parent_config["model"] != dict(expected_model)
        or parent_config["model"].get("hint_dim") != 0
        or not isinstance(parent_provenance.get("precision"), dict)
    ):
        raise ValueError(
            "teacher initializer direct parent is not bound to a label-only strict-FP32 fit"
        )
    common_provenance(parent, parent_provenance)
    expected_fit = fit_members
    expected_evaluation = evaluation_members
    if (
        parent.get("split_sha256") != split_sha256
        or parent.get("fit_members") != expected_fit
        or parent_data.get("split_sha256") != split_sha256
        or parent_data.get("primary_split_sha256") != primary_split_sha256
        or parent_data.get("fit_members") != expected_fit
        or parent_data.get("evaluation_members") != expected_evaluation
        or parent_data.get("evaluation_role") != "control"
        or parent_data.get("evaluation_source") != "train"
        or parent_data.get("evaluation_mask") != "all_after_warmup"
        or parent_data.get("training_mask") != "all_after_warmup"
        or parent_data.get("mask") != "all_after_warmup AND row_index>=99"
        or parent_data.get("teacher") is not None
        or parent_data.get("initializer") is not None
        or (expected_sources is not None and parent_data.get("sources") != dict(expected_sources))
    ):
        raise ValueError(
            "teacher initializer direct parent differs from the frozen auxiliary protocol"
        )
    if (
        initializer.get("schema") != "wnn.sequence-initializer.v1"
        or not _is_sha256(initializer.get("checkpoint_sha256"))
        or not _is_sha256(initializer.get("parent_identity_sha256"))
        or not _is_sha256(initializer.get("parent_config_sha256"))
        or initializer.get("parent_split_sha256") != split_sha256
        or initializer.get("primary_split_sha256") != primary_split_sha256
        or initializer.get("fit_members_sha256") != canonical_sha256(fit_members)
        or initializer.get("evaluation_members_sha256") != canonical_sha256(evaluation_members)
        or initializer.get("evaluation_role") != "control"
        or initializer.get("evaluation_source") != "train"
        or initializer.get("training_mask") != "all_after_warmup"
        or initializer.get("mask") != "all_after_warmup AND row_index>=99"
    ):
        raise ValueError("teacher initializer ancestry differs from the auxiliary fit/control")


def _validate_parent(
    parent: object,
    split_sha256: str,
    fit_members: list[list[object]],
    evaluation_members: list[list[object]] | None,
    primary_split_sha256: str | None,
    expected_sources: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Verify one direct stage-one/two model and its complete fixed-final lineage."""
    if not isinstance(parent, dict) or not isinstance(parent.get("producer"), dict):
        raise ValueError("weighted teacher parent record is malformed")
    producer, provenance = parent["producer"], parent.get("provenance")
    if not isinstance(provenance, dict) or producer.get("provenance") != provenance:
        raise ValueError("weighted teacher parent provenance is detached")
    if (
        producer.get("kind") not in (None, "single_checkpoint")
        or producer.get("teacher") is not None
    ):
        raise ValueError("weighted pools accept only direct label-trained stage-one/two models")
    checkpoint_sha256 = producer.get("checkpoint_sha256")
    config, identity = producer.get("config"), provenance.get("identity")
    if (
        not _is_sha256(checkpoint_sha256)
        or not isinstance(config, dict)
        or not isinstance(identity, dict)
        or not isinstance(identity.get("data"), dict)
        or not isinstance(config.get("model"), dict)
        or config.get("precision_mode") != "fp32"
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or config["model"].get("hint_dim") != 0
    ):
        raise ValueError("weighted teacher parent must be a label-trained strict-FP32 checkpoint")
    data = identity["data"]
    expected_evaluation = data.get("evaluation_members")
    if (
        producer.get("split_sha256") != split_sha256
        or producer.get("fit_members") != fit_members
        or data.get("split_sha256") != split_sha256
        or data.get("fit_members") != fit_members
        or data.get("evaluation_role") != "control"
        or data.get("evaluation_source") != "train"
        or data.get("evaluation_mask") != "all_after_warmup"
        or data.get("training_mask") != "all_after_warmup"
        or data.get("teacher") is not None
        or not isinstance(expected_evaluation, list)
        or not expected_evaluation
        or any(
            not isinstance(member, list)
            or len(member) != 2
            or member[0] != "train"
            or type(member[1]) is not int
            for member in expected_evaluation
        )
        or set(map(tuple, expected_evaluation)) & set(map(tuple, fit_members))
    ):
        raise ValueError(
            "weighted teacher fit/control provenance is not component-disjoint train data"
        )
    if evaluation_members is not None and expected_evaluation != evaluation_members:
        raise ValueError("weighted teacher control membership differs from the frozen split")
    parent_primary = data.get("primary_split_sha256")
    if not _is_sha256(parent_primary) or (
        primary_split_sha256 is not None and parent_primary != primary_split_sha256
    ):
        raise ValueError("weighted teacher primary-split identity differs")
    if expected_sources is not None and data.get("sources") != dict(expected_sources):
        raise ValueError("weighted teacher source identities differ from the frozen split")
    common_provenance(producer, provenance)
    _validate_initializer(
        data.get("initializer"),
        split_sha256,
        parent_primary,
        fit_members,
        expected_evaluation,
        config["model"],
        expected_sources,
    )
    return producer


def validate_weighted_pool_node(
    current: dict[str, Any],
    split_sha256: str,
    members: list[tuple[str, int]],
    evaluation_members: list[tuple[str, int]] | None,
    primary_split_sha256: str | None,
    expected_sources: Mapping[str, object] | None,
) -> list[dict[str, Any]]:
    """Validate pinned heterogeneous parents, coefficients, arithmetic and ancestry."""
    if "checkpoint_sha256" in current or current.get("teacher") is not None:
        raise ValueError("weighted pool identity cannot claim one checkpoint or teacher")
    fit_members = [list(member) for member in members]
    expected_control = None if evaluation_members is None else [list(x) for x in evaluation_members]
    parents, composition = current.get("parents"), current.get("composition")
    if (
        not isinstance(parents, list)
        or not 2 <= len(parents) <= 8
        or not isinstance(composition, dict)
    ):
        raise ValueError("weighted pool requires two to eight complete teacher parents")
    validated: list[dict[str, Any]] = []
    references: list[dict[str, object]] = []
    for parent in parents:
        producer = _validate_parent(
            parent,
            split_sha256,
            fit_members,
            expected_control,
            primary_split_sha256,
            expected_sources,
        )
        if not isinstance(parent, dict):
            raise ValueError("weighted teacher parent record is malformed")
        for field in _PARENT_REFERENCE_FIELDS:
            if field not in parent:
                raise ValueError(f"weighted teacher parent omits {field}")
        weight = parent.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or weight <= 0
        ):
            raise ValueError("weighted teacher coefficients must be finite and positive")
        provenance = parent["provenance"]
        identity = provenance.get("identity")
        if (
            parent.get("checkpoint_sha256") != producer.get("checkpoint_sha256")
            or parent.get("config_sha256") != canonical_sha256(producer.get("config"))
            or parent.get("producer_identity_sha256") != canonical_sha256(identity)
            or parent.get("producer_sha256") != canonical_sha256(producer)
            or parent.get("provenance_sha256") != canonical_sha256(provenance)
            or not _is_sha256(parent.get("manifest_sha256"))
            or not isinstance(parent.get("trial_id"), str)
            or not parent["trial_id"]
        ):
            raise ValueError("weighted teacher parent hashes do not match its source records")
        validated.append(producer)
        references.append({field: parent[field] for field in _PARENT_REFERENCE_FIELDS})
    if (
        len({item["checkpoint_sha256"] for item in references}) != len(parents)
        or len({item["manifest_sha256"] for item in references}) != len(parents)
        or len({item["trial_id"] for item in references}) != len(parents)
        or abs(math.fsum(float(item["weight"]) for item in references) - 1.0) > 1e-7
    ):
        raise ValueError("weighted teacher parents must be unique and coefficients must sum to one")
    expected_composition = {
        "schema": "sequence-weighted-teacher-pool-v1",
        "method": WEIGHTED_POOL_METHOD,
        "pool_id": current.get("pool_id"),
        "pool_request_sha256": current.get("pool_request_sha256"),
        "target_contract": WEIGHTED_POOL_TARGET_CONTRACT,
        "parents": references,
    }
    if (
        not isinstance(current.get("pool_id"), str)
        or not current["pool_id"]
        or not _is_sha256(current.get("pool_request_sha256"))
        or current.get("split_sha256") != split_sha256
        or current.get("fit_members") != fit_members
        or composition != expected_composition
        or current.get("composition_sha256") != canonical_sha256(composition)
    ):
        raise ValueError("weighted pool composition or clipping arithmetic is not pinned")
    return validated
