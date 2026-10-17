"""Select every eligible complete component for geometry comparison membership."""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias

import numpy as np

MembershipRow: TypeAlias = tuple[str, int, int, int]


def _component_rows(rows: dict[str, np.ndarray], components: set[int]) -> list[MembershipRow]:
    """Materialize source-qualified rows for the requested cached components."""
    selected = np.isin(rows["chain"], np.fromiter(components, dtype=np.int64))
    return [
        (str(source), int(group), int(component), int(fold))
        for source, group, component, fold in zip(
            rows["split"][selected],
            rows["row_group"][selected],
            rows["chain"][selected],
            rows["fold"][selected],
            strict=True,
        )
    ]


def _fold_counts(rows: list[MembershipRow]) -> dict[str, int]:
    """Count selected sequences by source and cached fold."""
    return {
        f"{source}:fold{fold}": count
        for (source, fold), count in sorted(Counter((r[0], r[3]) for r in rows).items())
    }


def select_all_eligible(
    rows: dict[str, np.ndarray], bad_train_row_group: int
) -> tuple[list[MembershipRow], list[MembershipRow], dict[str, object]]:
    """Select all whole fold-2/3 fit components and fold-0/1 valid eval components."""
    split, group, component, fold = (
        rows["split"],
        rows["row_group"],
        rows["chain"],
        rows["fold"],
    )
    fit_components = set(map(int, np.unique(component[np.isin(fold, (2, 3))])))
    eval_components = set(
        map(
            int,
            np.unique(component[np.isin(fold, (0, 1)) & (split == "valid")]),
        )
    )
    bad_rows = (split == "train") & (group == bad_train_row_group)
    bad_components = set(map(int, np.unique(component[bad_rows])))
    if int(np.sum(bad_rows)) > 1 or len(bad_components) > 1:
        raise ValueError("known bad train row group is not uniquely indexed")
    if fit_components & eval_components:
        raise ValueError("fit and evaluation folds share a cached component")
    fit_components -= bad_components

    fitting = _component_rows(rows, fit_components)
    eval_members = [
        row
        for row in _component_rows(rows, eval_components)
        if not (row[0] == "train" and row[1] == bad_train_row_group)
    ]
    evaluation = [row for row in eval_members if row[0] == "valid"]
    held_out_train = [row for row in eval_members if row[0] == "train"]
    if not evaluation or not fitting:
        raise ValueError("all-eligible membership unexpectedly has an empty split")
    if any(row[3] not in (2, 3) for row in fitting):
        raise ValueError("fitting membership contains a component outside folds 2 and 3")
    if any(row[3] not in (0, 1) for row in eval_members):
        raise ValueError("evaluation membership contains a component outside folds 0 and 1")
    if any(row[0] != "valid" for row in evaluation):
        raise ValueError("evaluation scoring rows must be valid source rows")
    if {row[2] for row in fitting} & {row[2] for row in eval_members}:
        raise ValueError("fit and evaluation component membership overlaps")
    if len({(row[0], row[1]) for row in fitting + eval_members}) != len(fitting + eval_members):
        raise ValueError("source-qualified row group appears more than once")

    return (
        fitting,
        evaluation,
        {
            "selection": "all eligible complete components; no random sampling",
            "membership_seed": None,
            "fit_component_count": len(fit_components),
            "fit_components_by_fold": {
                str(value): len(set(row[2] for row in fitting if row[3] == value))
                for value in (2, 3)
            },
            "fit_sequence_counts_by_source_fold": _fold_counts(fitting),
            "eval_component_count": len(eval_components),
            "eval_components_by_fold": {
                str(value): len(set(row[2] for row in eval_members if row[3] == value))
                for value in (0, 1)
            },
            "eval_component_member_sequence_count": len(eval_members),
            "eval_component_member_counts_by_source_fold": _fold_counts(eval_members),
            "eval_scored_sequence_counts_by_source_fold": _fold_counts(evaluation),
            "held_out_train_counterpart_count": len(held_out_train),
            "held_out_train_counterpart_components": len({row[2] for row in held_out_train}),
            "held_out_train_counterparts": held_out_train,
            "excluded_train_row": {"source": "train", "row_group": bad_train_row_group},
            "excluded_train_component": next(iter(bad_components)) if bad_components else None,
            "known_bad_row_status": "indexed_and_excluded"
            if bad_components
            else "absent_from_chain_cache",
        },
    )
