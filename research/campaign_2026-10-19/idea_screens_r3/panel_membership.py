"""Metadata validation and deterministic R3 component-first selection."""

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


def _split_index(
    split: dict[str, object],
) -> tuple[
    dict[int, list[dict[str, object]]], dict[int, list[dict[str, object]]], dict[str, object]
]:
    """Validate full role membership and return train/eval eligible group pools."""
    sequences = split.get("sequences")
    if not isinstance(sequences, dict):
        raise ValueError("canonical split has no sequence metadata map")
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
            if group in source_index:
                raise ValueError(f"duplicate sequence metadata: {source}/{group}")
            source_index[group] = record
        index[str(source)] = source_index

    memberships: dict[str, dict[str, list[int]]] = {"fit": {}, "evaluation": {}}
    role_components: dict[str, set[int]] = {"fit": set(), "evaluation": set()}
    role_group_counts: dict[str, dict[str, int]] = {"fit": {}, "evaluation": {}}
    for role in ("fit", "evaluation"):
        mapping = split.get(role)
        if not isinstance(mapping, dict):
            raise ValueError(f"canonical split has no {role} membership")
        for source, raw_groups in mapping.items():
            groups = [int(group) for group in raw_groups]
            if len(groups) != len(set(groups)):
                raise ValueError(f"duplicate {role} group for source {source}")
            if source not in index or any(group not in index[source] for group in groups):
                raise ValueError(f"missing {role} source/group metadata for {source}")
            memberships[role][str(source)] = groups
            role_group_counts[role][str(source)] = len(groups)
            role_components[role].update(int(index[source][group]["component"]) for group in groups)
    for source, source_index in index.items():
        fit_groups = set(memberships["fit"].get(source, []))
        eval_groups = set(memberships["evaluation"].get(source, []))
        if fit_groups & eval_groups:
            raise ValueError(f"source-qualified group occurs in both roles: {source}")
        if fit_groups | eval_groups != set(source_index):
            raise ValueError(f"canonical membership does not cover source metadata: {source}")
    if role_components["fit"] & role_components["evaluation"]:
        raise ValueError("full fitting/evaluation component overlap before selection")

    fit_pool: dict[int, list[dict[str, object]]] = defaultdict(list)
    eval_pool: dict[int, list[dict[str, object]]] = defaultdict(list)
    for group in memberships["fit"].get("train", []):
        item = index["train"][group]
        fit_pool[int(item["component"])].append({"source": "train", "group": group, **item})
    for group in memberships["evaluation"].get("valid", []):
        item = index["valid"][group]
        eval_pool[int(item["component"])].append({"source": "valid", "group": group, **item})
    if len(fit_pool) < FIT_COMPONENTS or len(eval_pool) < EVAL_COMPONENTS:
        raise ValueError("insufficient eligible train or evaluation components")
    counts = {
        "roles": {
            role: {
                "components": len(role_components[role]),
                "groups": sum(role_group_counts[role].values()),
                "train_components": len(
                    {
                        int(index["train"][g]["component"])
                        for g in memberships[role].get("train", [])
                    }
                ),
                "valid_components": len(
                    {
                        int(index["valid"][g]["component"])
                        for g in memberships[role].get("valid", [])
                    }
                ),
                "source_groups": role_group_counts[role],
            }
            for role in ("fit", "evaluation")
        },
        "eligible_train_components": len(fit_pool),
        "eligible_eval_components": len(eval_pool),
    }
    return dict(fit_pool), dict(eval_pool), counts


def _check_panel_metadata(panel: dict[str, object], counts: dict[str, object]) -> None:
    """Require the metadata-only panel preflight to match the canonical split."""
    if panel.get("split_sha256") != SPLIT_SHA:
        raise ValueError("panel metadata split identity differs")
    observed_roles = panel.get("roles")
    if not isinstance(observed_roles, dict):
        raise ValueError("panel metadata has no role summaries")
    expected_roles = counts["roles"]
    for role, expected in expected_roles.items():
        observed = observed_roles.get("fit" if role == "fit" else "eval")
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
    """Select64 train and32 eval components, then one eligible sequence per component."""
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
        "schema": "idea_screens_r3.membership.v1",
        "split_sha256": SPLIT_SHA,
        "metadata_counts": counts,
        "selection": selection,
        "records": records,
    }
