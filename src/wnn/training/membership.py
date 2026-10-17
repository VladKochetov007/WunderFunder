from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.splits import CampaignSplit


def load_component_fit_subset(
    path: Path, campaign_split: CampaignSplit, *, expected_experiment_id: str | None = None
) -> dict[str, object]:
    """Validate component-complete fitting members against a frozen campaign split."""
    if path.stat().st_size > 5000000:
        raise ValueError("fit subset metadata exceeds the 5 MB bound")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("fit subset manifest must be a version-1 JSON object")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("fit subset must name its experiment")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError("fit subset belongs to a different experiment")
    if manifest.get("split_manifest_sha256") != campaign_split.sha256:
        raise ValueError("fit subset was built from a different campaign split")
    groups_by_split = manifest.get("selected_groups")
    if not isinstance(groups_by_split, dict):
        raise ValueError("fit subset has no selected_groups object")
    allowed = {"train": set(campaign_split.train_groups), "valid": set(campaign_split.valid_groups)}
    selected: dict[str, list[int]] = {}
    for split in ("train", "valid"):
        raw_groups = groups_by_split.get(split)
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"fit subset must include selected {split} groups")
        if any(type(group) is not int or group < 0 for group in raw_groups):
            raise ValueError(f"selected {split} groups must be nonnegative integers")
        groups = [int(group) for group in raw_groups]
        if len(set(groups)) != len(groups) or not set(groups).issubset(allowed[split]):
            raise ValueError(f"selected {split} groups are duplicated or outside the fit split")
        selected[split] = groups
    members = sorted(((split, group) for split, groups in selected.items() for group in groups))
    member_bytes = json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if hashlib.sha256(member_bytes).hexdigest() != manifest.get("selected_members_sha256"):
        raise ValueError("fit subset membership digest is invalid")
    component_list = manifest.get("selected_components")
    if (
        not isinstance(component_list, list)
        or any(type(component) is not int or component < 0 for component in component_list)
        or component_list != sorted(set(component_list))
    ):
        raise ValueError("selected component IDs must be unique sorted nonnegative integers")
    component_bytes = np.asarray(component_list, dtype="<i8").tobytes()
    if hashlib.sha256(component_bytes).hexdigest() != manifest.get("selected_component_sha256"):
        raise ValueError("selected component digest is invalid")
    component_of = {
        (split, int(group)): int(component) for split, group, component in campaign_split.components
    }
    selected_components = {component_of[split, group] for split, group in members}
    if sorted(selected_components) != component_list:
        raise ValueError("selected component IDs disagree with their group membership")
    evaluation_components = {
        component_of["valid", group] for group in campaign_split.evaluation_groups
    }
    if selected_components.intersection(evaluation_components):
        raise ValueError("fit subset overlaps an evaluated component")
    evaluation_bytes = np.asarray(sorted(campaign_split.evaluation_groups), dtype="<i8").tobytes()
    evaluation_groups = manifest.get("evaluation_groups")
    if (
        not isinstance(evaluation_groups, list)
        or any(type(group) is not int or group < 0 for group in evaluation_groups)
        or evaluation_groups != sorted(set(evaluation_groups))
        or (evaluation_groups != sorted(campaign_split.evaluation_groups))
        or (type(manifest.get("evaluation_group_count")) is not int)
        or (manifest.get("evaluation_group_count") != len(evaluation_groups))
    ):
        raise ValueError("fit subset exact evaluation groups differ from the campaign split")
    if hashlib.sha256(evaluation_bytes).hexdigest() != manifest.get("evaluation_membership_sha256"):
        raise ValueError("fit subset was built against a different evaluation membership")
    expected_members = {
        (split, group)
        for split, group, component in campaign_split.components
        if group in allowed[split] and component in selected_components
    }
    if set(members) != expected_members:
        raise ValueError("fit subset splits an eligible fitting component")
    campaign_split.assert_fit_groups(selected["train"], selected["valid"])
    return {
        "manifest": manifest,
        "train_groups": selected["train"],
        "valid_groups": selected["valid"],
    }


def load_e540_fit_subset(path: Path, campaign_split: CampaignSplit) -> dict[str, object]:
    """Keep the legacy E540 loader bound to its original experiment identity."""
    return load_component_fit_subset(path, campaign_split, expected_experiment_id="E540")
