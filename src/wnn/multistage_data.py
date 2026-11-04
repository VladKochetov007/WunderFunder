"""Fitting-only component holdouts for the bounded multistage search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MAX_CAMPAIGN_SPLIT_BYTES,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    CampaignSplit,
    _read_campaign_manifest,
    load_campaign_split,
)

SELECTION_ALGORITHM = "sha256-primary-train-component-order-v1"
_POLICY = "train_only_whole_component_auxiliary_control_v1"
_COMPONENT_METHOD = "undirected connected components of feature-only successor graph"


def _validate_selection(seed: int, evaluation_fraction: float) -> tuple[int, float]:
    """Validate the deterministic auxiliary holdout parameters."""
    if type(seed) is not int or seed < 0:
        raise ValueError("auxiliary split seed must be a nonnegative integer")
    if isinstance(evaluation_fraction, bool) or not isinstance(evaluation_fraction, (int, float)):
        raise ValueError("evaluation fraction must be finite and strictly between zero and one")
    try:
        fraction = float(evaluation_fraction)
    except OverflowError as exc:
        raise ValueError(
            "evaluation fraction must be finite and strictly between zero and one"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("evaluation fraction must be finite and strictly between zero and one")
    return seed, fraction


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality ambiguity."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _selected_membership(
    split: CampaignSplit, seed: int, evaluation_fraction: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select whole train-only components and retain fitting members."""
    seed, fraction = _validate_selection(seed, evaluation_fraction)
    component_of = {(source, group): component for source, group, component in split.components}
    eligible = set(split.train_groups)
    valid_components = {
        component for (source, _), component in component_of.items() if source == "valid"
    }
    groups_by_component: dict[int, list[int]] = {}
    for group in split.train_groups:
        component = component_of[("train", group)]
        if component in valid_components:
            raise ValueError("primary fitting membership contains a validation-linked component")
        groups_by_component.setdefault(component, []).append(group)
    all_train_by_component: dict[int, list[int]] = {}
    for (source, group), component in component_of.items():
        if source == "train":
            all_train_by_component.setdefault(component, []).append(group)
    if any(
        set(all_train_by_component[component]) != set(groups)
        for component, groups in groups_by_component.items()
    ):
        raise ValueError("primary train membership splits a known connected component")
    if len(groups_by_component) < 2:
        raise ValueError("auxiliary holdout needs at least two primary-train components")
    ordered = sorted(
        groups_by_component,
        key=lambda component: (
            hashlib.sha256(
                f"wnn.multistage.aux.v1:{split.sha256}:{seed}:{component}".encode()
            ).hexdigest(),
            component,
        ),
    )
    target = max(1, math.ceil(fraction * len(eligible)))
    held_components: set[int] = set()
    held_count = 0
    for component in ordered:
        if held_count >= target:
            break
        count = len(groups_by_component[component])
        if held_count + count >= len(eligible):
            continue
        held_components.add(component)
        held_count += count
    held = tuple(
        sorted(group for component in held_components for group in groups_by_component[component])
    )
    fit = tuple(sorted(eligible - set(held)))
    if not fit or not held:
        raise ValueError("auxiliary split must retain nonempty fit and control memberships")
    return fit, held


def _ordered_groups(value: object, label: str) -> tuple[int, ...]:
    """Validate one sorted unique source row-group list."""
    if (
        not isinstance(value, list)
        or any(type(group) is not int or group < 0 for group in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{label} membership must be sorted unique nonnegative row groups")
    return tuple(value)


def load_multistage_auxiliary_split(
    path: Path, root: Path, raw: bytes, manifest: dict[str, Any]
) -> CampaignSplit:
    """Revalidate a train-only auxiliary split against its primary parent and graph."""
    required = {
        "schema",
        "version",
        "policy",
        "chronology_basis",
        "component_method",
        "fold",
        "n_folds",
        "parent",
        "selection",
        "sources",
        "sequences",
        "fit",
        "evaluation",
        "excluded",
    }
    if set(manifest) != required or manifest.get("schema") != MULTISTAGE_AUXILIARY_SPLIT_SCHEMA:
        raise ValueError("invalid multistage auxiliary split schema")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("unsupported multistage auxiliary split version")
    if manifest.get("policy") != _POLICY or manifest.get("component_method") != _COMPONENT_METHOD:
        raise ValueError("unsupported multistage auxiliary split policy or graph method")
    if manifest.get("chronology_basis") != (
        "none; deterministic whole-component auxiliary partition, not a time split"
    ):
        raise ValueError("auxiliary manifest makes an unsupported chronology claim")
    parent_record = manifest.get("parent")
    if not isinstance(parent_record, dict) or set(parent_record) != {"path", "sha256"}:
        raise ValueError("invalid primary split parent identity")
    parent_text, parent_hash = parent_record.get("path"), parent_record.get("sha256")
    if not isinstance(parent_text, str) or not Path(parent_text).is_absolute():
        raise ValueError("primary split parent path must be absolute")
    parent_path, parent_raw, parent_manifest = _read_campaign_manifest(Path(parent_text))
    if str(parent_path) != parent_text or parent_path == path.resolve():
        raise ValueError("invalid or recursive primary split parent path")
    if not isinstance(parent_hash, str) or hashlib.sha256(parent_raw).hexdigest() != parent_hash:
        raise ValueError("primary split parent hash changed")
    primary = load_campaign_split(parent_path, root)
    if primary.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or primary.valid_groups:
        raise ValueError("auxiliary split parent must be the official chronological primary split")
    if type(manifest.get("fold")) is not int or type(manifest.get("n_folds")) is not int:
        raise ValueError("auxiliary fold identity must be integer-valued")
    if (manifest["fold"], manifest["n_folds"]) != (primary.fold, primary.n_folds):
        raise ValueError("auxiliary split must preserve the primary fold identity")
    if not _json_equal(manifest.get("sources"), parent_manifest.get("sources")) or not _json_equal(
        manifest.get("sequences"), parent_manifest.get("sequences")
    ):
        raise ValueError("auxiliary sources or component graph differ from primary parent")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "algorithm",
        "seed",
        "evaluation_fraction",
    }:
        raise ValueError("invalid auxiliary selection metadata")
    if selection.get("algorithm") != SELECTION_ALGORITHM or type(selection.get("seed")) is not int:
        raise ValueError("unsupported auxiliary selection algorithm or seed")
    seed, fraction = _validate_selection(selection["seed"], selection["evaluation_fraction"])
    expected_fit, expected_control = _selected_membership(primary, seed, fraction)
    fit_record, evaluation_record = manifest.get("fit"), manifest.get("evaluation")
    if not isinstance(fit_record, dict) or set(fit_record) != {"train", "valid"}:
        raise ValueError("invalid auxiliary fitting membership")
    if not isinstance(evaluation_record, dict) or set(evaluation_record) != {"source", "train"}:
        raise ValueError("invalid auxiliary control membership")
    fit_train = _ordered_groups(fit_record.get("train"), "fit train")
    if fit_record.get("valid") != []:
        raise ValueError("auxiliary inner fitting must not include official valid sequences")
    control = _ordered_groups(evaluation_record.get("train"), "control train")
    if evaluation_record.get("source") != "train" or (fit_train, control) != (
        expected_fit,
        expected_control,
    ):
        raise ValueError("auxiliary membership differs from deterministic primary-train selection")
    if set(fit_train).intersection(control) or set(fit_train).union(control) != set(
        primary.train_groups
    ):
        raise ValueError("auxiliary fit/control membership does not partition primary training")
    excluded = manifest.get("excluded")
    expected_excluded = parent_manifest.get("excluded", {}).get("train_component_overlap")
    if not isinstance(excluded, dict) or set(excluded) != {"primary_component_overlap"}:
        raise ValueError("invalid auxiliary excluded membership")
    if _ordered_groups(excluded.get("primary_component_overlap"), "excluded train") != tuple(
        expected_excluded or ()
    ):
        raise ValueError("auxiliary exclusions differ from the primary split")
    result = CampaignSplit(
        path.resolve(),
        hashlib.sha256(raw).hexdigest(),
        primary.fold,
        primary.n_folds,
        fit_train,
        (),
        control,
        primary.components,
        MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
        "train",
        "control",
        "all_after_warmup",
        primary.sha256,
    )
    result.assert_fit_groups(result.train_groups, result.valid_groups)
    return result


def prepare_multistage_auxiliary_split(
    primary_manifest: Path,
    root: Path,
    output: Path,
    *,
    seed: int = 20261031,
    evaluation_fraction: float = 0.2,
) -> CampaignSplit:
    """Freeze a deterministic component-disjoint train/control split under root."""
    seed, fraction = _validate_selection(seed, evaluation_fraction)
    root = root.resolve()
    parent_path = primary_manifest.resolve()
    parent_path, parent_raw, parent_manifest = _read_campaign_manifest(parent_path)
    primary = load_campaign_split(parent_path, root)
    if primary.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or primary.valid_groups:
        raise ValueError("multistage search requires the official chronological primary manifest")
    fit, control = _selected_membership(primary, seed, fraction)
    payload: dict[str, object] = {
        "schema": MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
        "version": 1,
        "policy": _POLICY,
        "chronology_basis": "none; deterministic whole-component auxiliary partition, not a time split",
        "component_method": _COMPONENT_METHOD,
        "fold": primary.fold,
        "n_folds": primary.n_folds,
        "parent": {"path": str(parent_path), "sha256": hashlib.sha256(parent_raw).hexdigest()},
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "evaluation_fraction": fraction,
        },
        "sources": parent_manifest["sources"],
        "sequences": parent_manifest["sequences"],
        "fit": {"train": list(fit), "valid": []},
        "evaluation": {"source": "train", "train": list(control)},
        "excluded": {
            "primary_component_overlap": parent_manifest["excluded"]["train_component_overlap"]
        },
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_CAMPAIGN_SPLIT_BYTES:
        raise ValueError("multistage auxiliary manifest exceeds its bounded size")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise FileExistsError(
                "existing auxiliary split differs; refusing to replace it"
            ) from None
    return load_campaign_split(output, root)
