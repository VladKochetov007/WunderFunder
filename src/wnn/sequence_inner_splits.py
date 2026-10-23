"""Prepare and validate component-clean inner validation splits for sequence fits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from wnn.splits import (
    SEQUENCE_INNER_SPLIT_SCHEMA,
    CampaignSplit,
    _read_campaign_manifest,
    load_campaign_split,
)

SELECTION_ALGORITHM = "sha256-component-order-v1"


def _validate_selection(seed: int, evaluation_fraction: float) -> tuple[int, float]:
    """Validate and normalize the deterministic selection parameters."""
    if type(seed) is not int or seed < 0:
        raise ValueError("inner split seed must be a nonnegative integer")
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
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Select whole valid-containing components while retaining both fit groups."""
    seed, evaluation_fraction = _validate_selection(seed, evaluation_fraction)
    component_of = {(source, group): component for source, group, component in split.components}
    valid_by_component: dict[int, list[int]] = {}
    train_by_component: dict[int, list[int]] = {}
    for group in split.valid_groups:
        valid_by_component.setdefault(component_of[("valid", group)], []).append(group)
    for group in split.train_groups:
        train_by_component.setdefault(component_of[("train", group)], []).append(group)
    if len(valid_by_component) < 2:
        raise ValueError("inner split requires at least two fitting validation components")

    ordered = sorted(
        valid_by_component,
        key=lambda component: (
            hashlib.sha256(
                f"wnn.sequence-inner.v1:{seed}:{split.fold}:{component}".encode()
            ).hexdigest(),
            component,
        ),
    )
    target = max(1, math.ceil(evaluation_fraction * len(split.valid_groups)))
    selected: set[int] = set()
    selected_valid = 0
    selected_train_groups: set[int] = set()
    for component in ordered:
        if selected_valid >= target:
            break
        candidate_valid = len(valid_by_component[component])
        candidate_train = set(train_by_component.get(component, ()))
        if selected_valid + candidate_valid >= len(split.valid_groups):
            continue
        if len(selected_train_groups | candidate_train) >= len(split.train_groups):
            continue
        selected.add(component)
        selected_valid += candidate_valid
        selected_train_groups.update(candidate_train)
    if selected_valid < target:
        raise ValueError("selection cannot reach the requested fraction and retain fit groups")

    evaluation = tuple(
        sorted(group for component in selected for group in valid_by_component[component])
    )
    train = tuple(sorted(set(split.train_groups) - selected_train_groups))
    valid = tuple(sorted(set(split.valid_groups) - set(evaluation)))
    if not train or not valid or not evaluation:
        raise ValueError("inner split must retain nonempty train, valid and evaluation groups")
    return train, valid, evaluation


def _strict_integer_list(value: object, label: str) -> list[int]:
    """Validate a canonical JSON list of nonnegative row-group integers."""
    if not isinstance(value, list) or any(type(group) is not int or group < 0 for group in value):
        raise ValueError(f"invalid {label} membership list")
    if value != sorted(set(value)):
        raise ValueError(f"{label} membership must be unique and sorted")
    return value


def load_inner_campaign_split(
    path: Path, root: Path, raw: bytes, manifest: dict[str, Any]
) -> CampaignSplit:
    """Revalidate the parent and derive inner membership from its frozen graph."""
    required = {
        "schema",
        "version",
        "fold",
        "n_folds",
        "parent",
        "selection",
        "sources",
        "sequences",
        "fit",
        "evaluation",
    }
    if set(manifest) != required or manifest.get("schema") != SEQUENCE_INNER_SPLIT_SCHEMA:
        raise ValueError("invalid sequence inner split wrapper schema")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("unsupported sequence inner split version")

    parent_record = manifest.get("parent")
    if not isinstance(parent_record, dict) or set(parent_record) != {"path", "sha256"}:
        raise ValueError("invalid sequence inner split parent identity")
    parent_text, parent_hash = parent_record.get("path"), parent_record.get("sha256")
    if not isinstance(parent_text, str) or not Path(parent_text).is_absolute():
        raise ValueError("inner split parent path must be absolute")
    if (
        not isinstance(parent_hash, str)
        or len(parent_hash) != 64
        or any(character not in "0123456789abcdef" for character in parent_hash)
    ):
        raise ValueError("invalid inner split parent hash")
    parent_path, parent_raw, parent_manifest = _read_campaign_manifest(Path(parent_text))
    if str(parent_path) != parent_text:
        raise ValueError("inner split parent path must use its resolved absolute form")
    if parent_path == path.resolve():
        raise ValueError("recursive sequence inner split wrapper")
    if parent_manifest.get("schema") == SEQUENCE_INNER_SPLIT_SCHEMA:
        raise ValueError("recursive sequence inner split wrapper")
    parent = load_campaign_split(parent_path, root)
    observed_parent_hash = hashlib.sha256(parent_raw).hexdigest()
    if parent.sha256 != parent_hash or parent.sha256 != observed_parent_hash:
        raise ValueError("inner split parent manifest hash changed")

    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "algorithm",
        "seed",
        "evaluation_fraction",
    }:
        raise ValueError("invalid sequence inner split selection metadata")
    if selection.get("algorithm") != SELECTION_ALGORITHM:
        raise ValueError("unsupported sequence inner split selection algorithm")
    seed, fraction = selection.get("seed"), selection.get("evaluation_fraction")
    if type(seed) is not int:
        raise ValueError("inner split seed must be a nonnegative integer")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("invalid inner split evaluation fraction")
    seed, fraction = _validate_selection(seed, fraction)

    if (
        type(manifest.get("fold")) is not int
        or type(manifest.get("n_folds")) is not int
        or manifest["fold"] != parent.fold
        or manifest["n_folds"] != parent.n_folds
    ):
        raise ValueError("inner split must preserve its canonical parent fold")
    if not _json_equal(manifest.get("sources"), parent_manifest.get("sources")):
        raise ValueError("inner split sources differ from its parent manifest")
    if not _json_equal(manifest.get("sequences"), parent_manifest.get("sequences")):
        raise ValueError("inner split graph differs from its parent manifest")

    expected_train, expected_valid, expected_evaluation = _selected_membership(
        parent, seed, fraction
    )
    fit, evaluation = manifest.get("fit"), manifest.get("evaluation")
    if not isinstance(fit, dict) or set(fit) != {"train", "valid"}:
        raise ValueError("invalid inner split fit membership")
    if not isinstance(evaluation, dict) or set(evaluation) != {"valid"}:
        raise ValueError("invalid inner split evaluation membership")
    train = _strict_integer_list(fit.get("train"), "fit train")
    valid = _strict_integer_list(fit.get("valid"), "fit valid")
    held_out = _strict_integer_list(evaluation.get("valid"), "evaluation valid")
    if (train, valid, held_out) != (
        list(expected_train),
        list(expected_valid),
        list(expected_evaluation),
    ):
        raise ValueError("inner split membership differs from the deterministic parent selection")

    component_of = {(source, group): component for source, group, component in parent.components}
    outer_components = {component_of[("valid", group)] for group in parent.evaluation_groups}
    inner_components = {
        component_of[(source, group)]
        for source, groups in (("train", train), ("valid", valid), ("valid", held_out))
        for group in groups
    }
    if outer_components & inner_components:
        raise ValueError("inner split includes a parent outer-evaluation component")
    result = CampaignSplit(
        path.resolve(),
        hashlib.sha256(raw).hexdigest(),
        parent.fold,
        parent.n_folds,
        tuple(train),
        tuple(valid),
        tuple(held_out),
        parent.components,
    )
    result.assert_fit_groups(result.train_groups, result.valid_groups)
    return result


def _publish_exclusive(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without replacing an existing manifest."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_sequence_inner_split(
    parent_manifest: Path,
    data_root: Path,
    output: Path,
    seed: int = 41001,
    evaluation_fraction: float = 0.2,
) -> CampaignSplit:
    """Create an immutable inner split bound to a validated canonical outer fold."""
    parent_path, parent_raw, parent_data = _read_campaign_manifest(parent_manifest)
    if parent_data.get("schema") == SEQUENCE_INNER_SPLIT_SCHEMA:
        raise ValueError("recursive sequence inner split wrapper")
    parent = load_campaign_split(parent_path, data_root)
    if parent.sha256 != hashlib.sha256(parent_raw).hexdigest():
        raise ValueError("parent manifest changed while validating the canonical outer split")
    seed, evaluation_fraction = _validate_selection(seed, evaluation_fraction)
    output = output.resolve()
    if output == parent_path:
        raise ValueError("inner split output cannot replace its parent manifest")
    if output.exists():
        raise FileExistsError(f"refusing to replace sequence inner split: {output}")
    train, valid, evaluation = _selected_membership(parent, seed, evaluation_fraction)
    manifest = {
        "schema": SEQUENCE_INNER_SPLIT_SCHEMA,
        "version": 1,
        "fold": parent.fold,
        "n_folds": parent.n_folds,
        "parent": {"path": str(parent_path), "sha256": parent.sha256},
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "evaluation_fraction": evaluation_fraction,
        },
        "sources": parent_data["sources"],
        "sequences": parent_data["sequences"],
        "fit": {"train": list(train), "valid": list(valid)},
        "evaluation": {"valid": list(evaluation)},
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    result = load_inner_campaign_split(output, data_root, payload, manifest)
    _publish_exclusive(output, payload)
    return result
