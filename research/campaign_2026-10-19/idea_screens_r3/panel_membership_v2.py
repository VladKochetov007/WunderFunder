"""Canonical R3 membership checks and fixed component-first selection, schema v2."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
FIT_COMPONENTS = 64
LOW_FIT_COMPONENTS = 32
EVAL_COMPONENTS = 32
SEEDS = {
    "fit_components": 19021,
    "fit_groups": 19022,
    "eval_components": 19023,
    "eval_groups": 19024,
}
R3_DIR = Path(__file__).resolve().parent
GroupPool = dict[int, list[dict[str, object]]]


def _read_group_list(
    role_map: dict[str, object], source: str, name: str, known: set[int]
) -> list[int]:
    raw = role_map.get(source, [])
    if not isinstance(raw, list):
        raise ValueError(f"invalid {name}.{source} group list")
    groups = [int(group) for group in raw]
    if len(groups) != len(set(groups)):
        raise ValueError(f"duplicate {name}.{source} group")
    if any(group not in known for group in groups):
        raise ValueError(f"unknown {name}.{source} source/group metadata")
    return groups


def _split_index(split: dict[str, object]) -> tuple[GroupPool, GroupPool, dict[str, object]]:
    sequences = split.get("sequences")
    if not isinstance(sequences, dict) or set(sequences) != {"train", "valid"}:
        raise ValueError("canonical split must index train and valid sequence metadata")
    index: dict[str, dict[int, dict[str, object]]] = {}
    for source, raw_records in sequences.items():
        if not isinstance(raw_records, list):
            raise ValueError(f"invalid sequence metadata for {source}")
        source_index: dict[int, dict[str, object]] = {}
        for record in raw_records:
            if (
                not isinstance(record, dict)
                or not {"row_group", "seq_ix", "component"} <= record.keys()
            ):
                raise ValueError(f"incomplete sequence metadata for {source}")
            group = int(record["row_group"])
            sequence_id = int(record["seq_ix"])
            component = int(record["component"])
            if group < 0 or sequence_id < 0 or component < 0:
                raise ValueError(f"negative source/group metadata for {source}/{group}")
            if group in source_index:
                raise ValueError(f"duplicate sequence metadata: {source}/{group}")
            source_index[group] = {
                **record,
                "row_group": group,
                "seq_ix": sequence_id,
                "component": component,
            }
        index[source] = source_index

    fit, evaluation, excluded = (split.get(key) for key in ("fit", "evaluation", "excluded"))
    if any(not isinstance(value, dict) for value in (fit, evaluation, excluded)):
        raise ValueError("canonical split lacks fit, evaluation, or excluded memberships")
    if set(fit) != {"train", "valid"} or set(evaluation) != {"valid"}:
        raise ValueError("unexpected canonical fit/evaluation role keys")
    if set(excluded) != {"train", "valid", "corrupt_train"}:
        raise ValueError("unexpected canonical excluded role keys")

    train_groups = set(index["train"])
    valid_groups = set(index["valid"])
    fit_train = set(_read_group_list(fit, "train", "fit", train_groups))
    fit_valid = set(_read_group_list(fit, "valid", "fit", valid_groups))
    eval_valid = set(_read_group_list(evaluation, "valid", "evaluation", valid_groups))
    excluded_train = set(_read_group_list(excluded, "train", "excluded", train_groups))
    excluded_valid = set(_read_group_list(excluded, "valid", "excluded", valid_groups))
    corrupt_raw = excluded.get("corrupt_train")
    if not isinstance(corrupt_raw, list):
        raise ValueError("invalid excluded.corrupt_train group list")
    corrupt_values = [int(group) for group in corrupt_raw]
    if any(group < 0 for group in corrupt_values) or len(set(corrupt_values)) != len(
        corrupt_values
    ):
        raise ValueError("duplicate or negative excluded.corrupt_train marker")
    corrupt_train = set(corrupt_values)

    if fit_train & excluded_train or fit_train | excluded_train != train_groups:
        raise ValueError("fit.train and excluded.train must partition train metadata")
    if fit_valid & eval_valid or fit_valid | eval_valid != valid_groups:
        raise ValueError("fit.valid and evaluation.valid must partition valid metadata")
    if excluded_valid != eval_valid:
        raise ValueError("excluded.valid must be the evaluation.valid alias")
    corrupt_fit = corrupt_train & fit_train
    if corrupt_fit:
        raise ValueError(f"corrupt train markers are fit-eligible: {sorted(corrupt_fit)}")

    evaluation_components = {int(index["valid"][group]["component"]) for group in eval_valid}
    expected_excluded_train = {
        group
        for group, record in index["train"].items()
        if int(record["component"]) in evaluation_components
    }
    if excluded_train != expected_excluded_train:
        raise ValueError("excluded.train does not match evaluation-linked train metadata")

    fit_components = {
        int(index[source][group]["component"])
        for source, groups in (("train", fit_train), ("valid", fit_valid))
        for group in groups
    }
    if fit_components & evaluation_components:
        raise ValueError("full fitting/evaluation component overlap before selection")

    fit_pool: GroupPool = defaultdict(list)
    eval_pool: GroupPool = defaultdict(list)
    for group in sorted(fit_train):
        record = index["train"][group]
        component = int(record["component"])
        fit_pool[component].append({"source": "train", "group": group, **record})
    for group in sorted(eval_valid):
        record = index["valid"][group]
        component = int(record["component"])
        eval_pool[component].append({"source": "valid", "group": group, **record})
    if len(fit_pool) < FIT_COMPONENTS or len(eval_pool) < EVAL_COMPONENTS:
        raise ValueError("insufficient eligible train or evaluation components")

    role_groups = {"fit": {"train": fit_train, "valid": fit_valid}, "eval": {"valid": eval_valid}}
    role_counts: dict[str, dict[str, object]] = {}
    for role, by_source in role_groups.items():
        role_components = {
            int(index[source][group]["component"])
            for source, groups in by_source.items()
            for group in groups
        }
        role_counts[role] = {
            "components": len(role_components),
            "groups": sum(len(groups) for groups in by_source.values()),
            "train_components": len(
                {int(index["train"][group]["component"]) for group in by_source.get("train", set())}
            ),
            "valid_components": len(
                {int(index["valid"][group]["component"]) for group in by_source.get("valid", set())}
            ),
            "source_groups": {source: len(groups) for source, groups in by_source.items()},
        }
    counts = {
        "roles": role_counts,
        "eligible_train_components": len(fit_pool),
        "eligible_eval_components": len(eval_pool),
        "coverage_checks": {
            "train_metadata_groups": len(train_groups),
            "fit_train_groups": len(fit_train),
            "excluded_train_groups": len(excluded_train),
            "train_partition_complete": True,
            "valid_metadata_groups": len(valid_groups),
            "fit_valid_groups": len(fit_valid),
            "evaluation_valid_groups": len(eval_valid),
            "excluded_valid_alias_groups": len(excluded_valid),
            "valid_partition_complete": True,
            "excluded_valid_is_eval_alias": True,
            "excluded_train_matches_eval_components": True,
            "fit_eval_component_disjoint": True,
            "corrupt_train_markers": len(corrupt_train),
            "corrupt_train_markers_fit_eligible": sorted(corrupt_fit),
            "corrupt_train_markers_absent_from_train_metadata": sorted(
                corrupt_train - train_groups
            ),
        },
    }
    return dict(fit_pool), dict(eval_pool), counts


def _check_panel_metadata(panel: dict[str, object], counts: dict[str, object]) -> None:
    if panel.get("split_sha256") != SPLIT_SHA:
        raise ValueError("panel metadata split identity differs")
    observed_roles = panel.get("roles")
    if not isinstance(observed_roles, dict):
        raise ValueError("panel metadata has no role summaries")
    expected_roles = counts["roles"]
    for role, expected in expected_roles.items():
        observed = observed_roles.get(role)
        if not isinstance(observed, dict):
            raise ValueError(f"panel metadata missing role {role}")
        for key in (
            "components",
            "groups",
            "train_components",
            "valid_components",
            "source_groups",
        ):
            if observed.get(key) != expected[key]:
                raise ValueError(f"panel metadata mismatch at {role}.{key}")


def select_membership(
    split: dict[str, object], panel_metadata: dict[str, object] | None = None
) -> dict[str, object]:
    """Select the fixed R3 panel after validating explicit exclusions and aliases."""
    fit_pool, eval_pool, counts = _split_index(split)
    if panel_metadata is not None:
        _check_panel_metadata(panel_metadata, counts)
    fit_order = sorted(fit_pool)
    random.Random(SEEDS["fit_components"]).shuffle(fit_order)
    fit_order = fit_order[:FIT_COMPONENTS]
    eval_order = sorted(eval_pool)
    random.Random(SEEDS["eval_components"]).shuffle(eval_order)
    eval_order = eval_order[:EVAL_COMPONENTS]
    fit_rng, eval_rng = random.Random(SEEDS["fit_groups"]), random.Random(SEEDS["eval_groups"])
    fit_probability = FIT_COMPONENTS / len(fit_pool)
    low_probability = LOW_FIT_COMPONENTS / len(fit_pool)
    eval_probability = EVAL_COMPONENTS / len(eval_pool)
    records = []
    for order, component in enumerate(fit_order):
        candidates = sorted(fit_pool[component], key=lambda item: (item["source"], item["group"]))
        chosen = fit_rng.choice(candidates)
        conditional = 1.0 / len(candidates)
        records.append(
            {
                "role": "fit",
                "source": chosen["source"],
                "group": chosen["group"],
                "row_group": chosen["row_group"],
                "seq_ix": chosen["seq_ix"],
                "component": component,
                "included_in_low32": order < LOW_FIT_COMPONENTS,
                "fit_low32_order": order if order < LOW_FIT_COMPONENTS else None,
                "eligible_groups_in_component": len(candidates),
                "selection_probabilities": {
                    "fit_full64_component": fit_probability,
                    "fit_low32_component": low_probability,
                    "low32_component_given_full64": LOW_FIT_COMPONENTS / FIT_COMPONENTS,
                    "group_given_component": conditional,
                    "fit_full64_sequence": fit_probability * conditional,
                    "fit_low32_sequence": low_probability * conditional,
                },
            }
        )
    for component in eval_order:
        candidates = sorted(eval_pool[component], key=lambda item: (item["source"], item["group"]))
        chosen = eval_rng.choice(candidates)
        conditional = 1.0 / len(candidates)
        records.append(
            {
                "role": "eval",
                "source": chosen["source"],
                "group": chosen["group"],
                "row_group": chosen["row_group"],
                "seq_ix": chosen["seq_ix"],
                "component": component,
                "included_in_low32": False,
                "fit_low32_order": None,
                "eligible_groups_in_component": len(candidates),
                "selection_probabilities": {
                    "eval32_component": eval_probability,
                    "group_given_component": conditional,
                    "eval32_sequence": eval_probability * conditional,
                },
            }
        )
    selection = {
        "scheme": "uniform_component_sample_then_uniform_one_group_per_component",
        "seeds": SEEDS,
        "available_component_counts": {"fit_train": len(fit_pool), "eval_valid": len(eval_pool)},
        "selected_component_counts": {
            "fit_full64": len(fit_order),
            "fit_nested_low32": LOW_FIT_COMPONENTS,
            "eval32": len(eval_order),
        },
        "fit_full64_component_order": fit_order,
        "fit_low32_components": fit_order[:LOW_FIT_COMPONENTS],
        "eval32_component_order": eval_order,
        "probability_stages": {
            "fit_full64": {
                "component": fit_probability,
                "within_component_group": "1/k; k is recorded per sequence",
            },
            "fit_nested_low32": {
                "component": low_probability,
                "within_component_group": "1/k; same full64 group draw",
            },
            "eval32": {
                "component": eval_probability,
                "within_component_group": "1/k; k is recorded per sequence",
            },
        },
    }
    return {
        "schema": "idea_screens_r3.membership.v2",
        "split_sha256": SPLIT_SHA,
        "metadata_counts": counts,
        "selection": selection,
        "records": records,
    }
