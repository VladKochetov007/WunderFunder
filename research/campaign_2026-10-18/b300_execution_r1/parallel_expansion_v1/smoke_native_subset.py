"""Build and validate a tiny component-complete fit subset for engineering probes."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

from feature_common import sha, write  # noqa: E402

SUBSET_BUILDER_SHA256 = "ef0025c1233c3cfd613143fbdecb3d24a38275ab3c778278a8abfd66b96c5395"
SELECTION_SEEDS = tuple(range(20261019, 20261051))
TARGET_GROUP_COUNTS = {"train": 2, "valid": 1}


def _validate_subset(manifest: dict[str, Any], evaluation: list[int]) -> dict[str, Any]:
    """Check requested fit counts and preserve the complete original eval roster."""
    groups = manifest.get("selected_groups")
    counts = manifest.get("selected_group_counts")
    if not isinstance(groups, dict) or counts != TARGET_GROUP_COUNTS:
        raise ValueError("subset must contain two train and one fit-valid groups")
    if any(
        not isinstance(groups.get(name), list) or len(groups[name]) != count
        for name, count in TARGET_GROUP_COUNTS.items()
    ):
        raise ValueError("subset group lists disagree with the requested counts")
    if manifest.get("evaluation_groups") != sorted(evaluation):
        raise ValueError("fit subset changed the full original fold evaluation roster")
    if manifest.get("evaluation_group_count") != len(evaluation):
        raise ValueError("subset evaluation count differs from the original fold")
    if not isinstance(manifest.get("selected_members_sha256"), str):
        raise ValueError("subset lacks its canonical membership digest")
    return {
        "selected_groups": groups,
        "selected_group_counts": counts,
        "selected_components": manifest.get("selected_components"),
        "selected_members_sha256": manifest["selected_members_sha256"],
        "evaluation_group_count": len(evaluation),
    }


def make_subset(work: Path, source: Path, split_path: Path, job_id: str) -> dict[str, Any]:
    """Search a bounded seed set using the pinned canonical builder and loader."""
    sys.path.insert(0, str(source / "src"))
    splits = importlib.import_module("wnn.splits")
    membership = importlib.import_module("wnn.training.membership")
    campaign = splits.load_campaign_split(split_path, work)
    helper_path = source / "scripts/prepare_e540_fit_subset.py"
    if helper_path.is_symlink() or sha(helper_path) != SUBSET_BUILDER_SHA256:
        raise ValueError("component-subset builder differs from its pinned source hash")
    spec = importlib.util.spec_from_file_location("_smoke_fit_subset_builder", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned component-subset builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    experiment_id = f"native-smoke-{job_id}"
    selected: dict[str, Any] | None = None
    attempts = 0
    for seed in SELECTION_SEEDS:
        for count in (1, 2, 3):
            attempts += 1
            try:
                candidate = builder.build_manifest(
                    split_path, None, seed, work, component_count=count, experiment_id=experiment_id
                )
            except ValueError as error:
                if "must include train and valid fitting groups" in str(error):
                    continue
                raise
            if candidate.get("selected_group_counts") == TARGET_GROUP_COUNTS:
                selected = candidate
                break
            counts = candidate.get("selected_group_counts", {})
            if any(counts.get(key, 0) > value for key, value in TARGET_GROUP_COUNTS.items()):
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("bounded subset search found no component-complete 2+1 fit roster")

    subset_path = work / "fit_subset_manifest.json"
    write(subset_path, selected, fresh=True)
    validated = membership.load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=experiment_id
    )
    if (
        validated["train_groups"] != selected["selected_groups"]["train"]
        or validated["valid_groups"] != selected["selected_groups"]["valid"]
    ):
        raise ValueError("canonical subset loader returned different fit membership")
    return {
        "path": str(subset_path),
        "sha256": sha(subset_path),
        "builder_path": str(helper_path),
        "builder_sha256": sha(helper_path),
        "search_attempts": attempts,
        "selection_seed": selected["selection_seed"],
        "component_count": selected["selected_component_count"],
        "experiment_id": experiment_id,
        **_validate_subset(selected, list(campaign.evaluation_groups)),
    }
