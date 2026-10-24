"""Validate frozen outer splits and deterministic component-disjoint inner wrappers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from campaign_io import read_object, require, sha256_file

INNER_SCHEMA = "wnn.sequence-inner-split.v1"


def _group_map(manifest: dict[str, Any]) -> dict[tuple[str, int], int]:
    """Map each split/row-group identity to its connected-component ID."""
    sequences = manifest.get("sequences")
    require(isinstance(sequences, dict), "split sequences must be an object")
    result: dict[tuple[str, int], int] = {}
    for split, records in sequences.items():
        require(
            split in ("train", "valid") and isinstance(records, list),
            "bad split sequence section",
        )
        for record in records:
            require(isinstance(record, dict), "split sequence record must be an object")
            key = (split, int(record["row_group"]))
            require(key not in result, f"duplicate split sequence: {key}")
            result[key] = int(record["component"])
    return result


def _group_list(mapping: dict[str, Any], section: str, split: str) -> set[int]:
    """Read one nonempty, unique row-group membership list."""
    values = mapping.get(section, {}).get(split)
    require(isinstance(values, list) and values, f"{section}.{split} must be nonempty")
    result = {int(value) for value in values}
    require(len(result) == len(values), f"duplicate group in {section}.{split}")
    require(min(result) >= 0, f"negative group in {section}.{split}")
    return result


def validate_outer_inner_pair(
    outer_path: Path,
    outer: dict[str, Any],
    inner_path: Path,
    inner: dict[str, Any],
    fold: int,
) -> None:
    """Bind the wrapper to its outer parent and validate deterministic partitioning."""
    del inner_path
    outer_hash = sha256_file(outer_path)
    parent = inner.get("parent")
    require(
        inner.get("schema") == INNER_SCHEMA and inner.get("version") == 1,
        "unsupported inner split schema",
    )
    require(isinstance(parent, dict), "inner split parent binding must be an object")
    require(
        Path(parent.get("path", "")).resolve() == outer_path.resolve(),
        "inner split names a different outer parent",
    )
    require(
        parent.get("sha256") == outer_hash,
        "inner split parent hash differs from supplied outer split",
    )
    require(
        int(outer.get("fold", -1)) == fold and int(inner.get("fold", -1)) == fold,
        "split fold mismatch",
    )
    require(
        int(inner.get("n_folds", -1)) == int(outer.get("n_folds", -2)),
        "inner split fold count mismatch",
    )
    require(
        inner.get("sources") == outer.get("sources"),
        "inner split source identities differ from outer split",
    )
    require(
        inner.get("sequences") == outer.get("sequences"),
        "inner split component map differs from outer split",
    )
    selection = inner.get("selection")
    require(isinstance(selection, dict), "inner split selection metadata is missing")
    require(
        selection.get("algorithm") == "sha256-component-order-v1",
        "unsupported inner selection algorithm",
    )
    require(
        type(selection.get("seed")) is int and selection["seed"] >= 0,
        "invalid inner selection seed",
    )
    fraction = selection.get("evaluation_fraction")
    require(
        isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and 0 < fraction < 1,
        "invalid inner evaluation fraction",
    )

    components = _group_map(outer)
    outer_train = _group_list(outer, "fit", "train")
    outer_valid = _group_list(outer, "fit", "valid")
    fit_train = _group_list(inner, "fit", "train")
    fit_valid = _group_list(inner, "fit", "valid")
    eval_valid = _group_list(inner, "evaluation", "valid")
    outer_eval = _group_list(outer, "evaluation", "valid")
    require(not outer_eval.intersection(outer_valid), "outer fit/evaluation membership overlaps")
    outer_eval_components = {components[("valid", group)] for group in outer_eval}
    outer_fit_components = {components[("train", group)] for group in outer_train} | {
        components[("valid", group)] for group in outer_valid
    }
    require(
        outer_eval_components.isdisjoint(outer_fit_components),
        "outer fit/evaluation components overlap",
    )
    require(
        fit_train <= outer_train and fit_valid <= outer_valid,
        "inner fit exceeds outer fitting membership",
    )
    require(eval_valid <= outer_valid, "inner evaluation exceeds outer fitting membership")
    eval_components = {components[("valid", group)] for group in eval_valid}
    require(
        eval_components.isdisjoint(components[("train", group)] for group in fit_train)
        and eval_components.isdisjoint(components[("valid", group)] for group in fit_valid),
        "inner fit/evaluation components overlap",
    )

    valid_by_component: dict[int, set[int]] = {}
    train_by_component: dict[int, set[int]] = {}
    for group in outer_valid:
        valid_by_component.setdefault(components[("valid", group)], set()).add(group)
    for group in outer_train:
        train_by_component.setdefault(components[("train", group)], set()).add(group)
    candidates = sorted(
        valid_by_component,
        key=lambda component: (
            hashlib.sha256(
                f"wnn.sequence-inner.v1:{selection['seed']}:{fold}:{component}".encode()
            ).hexdigest(),
            component,
        ),
    )
    target = max(1, math.ceil(float(fraction) * len(outer_valid)))
    chosen_components: set[int] = set()
    chosen_valid_count = 0
    chosen_train_count = 0
    for component in candidates:
        valid_count = len(valid_by_component[component])
        train_count = len(train_by_component.get(component, set()))
        if chosen_valid_count + valid_count >= len(outer_valid):
            continue
        if chosen_train_count + train_count >= len(outer_train):
            continue
        chosen_components.add(component)
        chosen_valid_count += valid_count
        chosen_train_count += train_count
        if chosen_valid_count >= target:
            break
    expected_eval = {
        group for group in outer_valid if components[("valid", group)] in chosen_components
    }
    require(
        eval_valid == expected_eval, "inner evaluation differs from its frozen component-order seed"
    )
    expected_train = {
        group for group in outer_train if components[("train", group)] not in chosen_components
    }
    expected_valid = {
        group for group in outer_valid if components[("valid", group)] not in chosen_components
    }
    require(
        fit_train == expected_train, "inner training membership omits or adds eligible components"
    )
    require(
        fit_valid == expected_valid,
        "inner validation-fit membership omits or adds eligible components",
    )


def validate_role_pair(
    role_name: str,
    role: dict[str, Any],
    expected_fold: int,
    expected_schema: str,
    outer: dict[str, Any] | None = None,
    outer_path: Path | None = None,
) -> dict[str, Any]:
    """Re-read one source split and verify its campaign role and parent bindings."""
    source_path = Path(role["source_path"]).resolve(strict=True)
    require(sha256_file(source_path) == role["sha256"], f"source split changed: {source_path}")
    value = read_object(source_path)
    if expected_schema == INNER_SCHEMA:
        require(
            outer is not None and outer_path is not None, "inner split has no supplied outer parent"
        )
        validate_outer_inner_pair(outer_path, outer, source_path, value, expected_fold)
    else:
        require(
            value.get("version") == 1 and int(value.get("fold", -1)) == expected_fold,
            f"invalid outer split: {role_name}",
        )
    require(
        role["fold"] == expected_fold and role["schema"] == expected_schema,
        f"campaign split role changed: {role_name}",
    )
    return value
