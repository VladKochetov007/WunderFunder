"""Assemble exact weighted-pool ancestry from validated teacher cache exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from wnn.splits import MULTISTAGE_AUXILIARY_SPLIT_SCHEMA, CampaignSplit
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_cache_ensemble_lineage import validate_producer
from wnn.training.sequence_cache_ensemble_provenance import canonical_sha256
from wnn.training.sequence_cache_weighted_pool_lineage import (
    WEIGHTED_POOL_KIND,
    WEIGHTED_POOL_METHOD,
    WEIGHTED_POOL_TARGET_CONTRACT,
)

_PIN_FIELDS = (
    "trial_id",
    "weight",
    "cache_manifest_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "producer_identity_sha256",
    "producer_sha256",
    "provenance_sha256",
)


def build_weighted_pool_producer(
    caches: list[TeacherCache],
    expected_parents: list[Mapping[str, object]],
    split: CampaignSplit,
    pool_id: str,
    request_sha256: str,
) -> tuple[dict[str, Any], list[tuple[str, int]], list[tuple[str, int]]]:
    """Bind pinned model producers, fixed coefficients and the exact aux cohorts."""
    fit_members, control_members = auxiliary_members(split)
    sources = _read_split_sources(split)
    parents: list[dict[str, Any]] = []
    references: list[dict[str, object]] = []
    for cache, expected in zip(caches, expected_parents, strict=True):
        if not set(_PIN_FIELDS).issubset(expected):
            raise ValueError("each teacher pool parent must pin every source identity hash")
        producer = cache.metadata.get("producer")
        if not isinstance(producer, dict):
            raise ValueError("teacher cache producer is malformed")
        provenance = producer.get("provenance")
        identity = provenance.get("identity") if isinstance(provenance, dict) else None
        actual = {
            "trial_id": expected["trial_id"],
            "weight": expected["weight"],
            "cache_manifest_sha256": cache.identity["manifest_sha256"],
            "checkpoint_sha256": producer.get("checkpoint_sha256"),
            "config_sha256": canonical_sha256(producer.get("config")),
            "producer_identity_sha256": canonical_sha256(identity),
            "producer_sha256": canonical_sha256(producer),
            "provenance_sha256": canonical_sha256(provenance),
        }
        if any(expected.get(key) != value for key, value in actual.items()):
            raise ValueError("teacher export differs from the frozen trial/source receipt")
        parent = {
            "trial_id": actual["trial_id"],
            "weight": actual["weight"],
            "manifest_sha256": actual["cache_manifest_sha256"],
            "checkpoint_sha256": actual["checkpoint_sha256"],
            "config_sha256": actual["config_sha256"],
            "producer_identity_sha256": actual["producer_identity_sha256"],
            "producer_sha256": actual["producer_sha256"],
            "provenance_sha256": actual["provenance_sha256"],
            "producer": producer,
            "provenance": provenance,
        }
        parents.append(parent)
        references.append(
            {
                key: parent[key]
                for key in (
                    "trial_id",
                    "weight",
                    "manifest_sha256",
                    "checkpoint_sha256",
                    "config_sha256",
                    "producer_identity_sha256",
                    "producer_sha256",
                    "provenance_sha256",
                )
            }
        )
    composition = {
        "schema": "sequence-weighted-teacher-pool-v1",
        "method": WEIGHTED_POOL_METHOD,
        "pool_id": pool_id,
        "pool_request_sha256": request_sha256,
        "target_contract": WEIGHTED_POOL_TARGET_CONTRACT,
        "parents": references,
    }
    producer = {
        "kind": WEIGHTED_POOL_KIND,
        "split_sha256": split.sha256,
        "fit_members": [list(member) for member in fit_members],
        "teacher": None,
        "pool_id": pool_id,
        "pool_request_sha256": request_sha256,
        "composition": composition,
        "composition_sha256": canonical_sha256(composition),
        "parents": parents,
    }
    validate_producer(
        producer,
        split.sha256,
        fit_members,
        evaluation_members=control_members,
        primary_split_sha256=split.primary_split_sha256,
        expected_sources=sources,
    )
    return producer, fit_members, control_members


def _read_split_sources(split: CampaignSplit) -> dict[str, object]:
    """Load raw source fingerprints from the exact manifest already validated."""
    record = json.loads(split.path.read_text(encoding="utf-8"))
    sources = record.get("sources") if isinstance(record, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise ValueError("auxiliary split manifest must bind raw source fingerprints")
    return sources


def auxiliary_members(split: CampaignSplit) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Require the exact train-only auxiliary fit/control protocol."""
    if (
        split.schema != MULTISTAGE_AUXILIARY_SPLIT_SCHEMA
        or split.valid_groups
        or split.evaluation_source != "train"
        or split.evaluation_role != "control"
        or split.evaluation_mask != "all_after_warmup"
        or not isinstance(split.primary_split_sha256, str)
        or len(split.primary_split_sha256) != 64
    ):
        raise ValueError("weighted teacher pools require the frozen train-only auxiliary split")
    fit_members = [("train", group) for group in split.train_groups]
    control_members = [("train", group) for group in split.evaluation_groups]
    if (
        not fit_members
        or not control_members
        or set(fit_members) & set(control_members)
        or len(set(fit_members)) != len(fit_members)
        or len(set(control_members)) != len(control_members)
    ):
        raise ValueError("auxiliary fit and control members must be unique and disjoint")
    return fit_members, control_members
