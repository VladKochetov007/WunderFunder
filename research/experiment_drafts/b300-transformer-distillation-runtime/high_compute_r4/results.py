"""Manual recipe selections and their conditional dependency receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import (
    campaign_digest,
    canonical_sha256,
    read_object,
    require,
    sha256_file,
    write_bytes_new,
    write_json_new,
)
from comparisons import (
    _key,
    _outcomes,
    _stage_path,
    validate_comparison_file,
)
from materialize import SELECTABLE_STAGES, load_stage


def _selection_evidence(
    root: Path,
    manifest: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Return the score map and complete/failure evidence for every declared arm."""
    score_hashes: dict[str, str] = {}
    evidence = []
    for key, outcome in sorted(outcomes.items()):
        if outcome["state"] == "completed":
            score_hashes[key] = str(outcome["score_sha256"])
        evidence.append({"endpoint": key, **outcome})
    return score_hashes, evidence


def select_variant(
    campaign_root: Path,
    stage: str,
    variant: str,
    reason: str,
) -> dict[str, Any]:
    """Publish a user-named variant selection after validating every arm and fold."""
    require(stage in SELECTABLE_STAGES, "only B-F have manual recipe selection")
    require(bool(reason.strip()), "selection reason must be nonempty")
    root = campaign_root.expanduser().resolve(strict=True)
    from campaign_io import load_campaign

    campaign = load_campaign(root)
    manifest = load_stage(root, stage)
    require(variant in manifest["variants"], f"unknown {stage} variant: {variant}")
    outcomes, _ = _outcomes(root, campaign, stage, manifest)
    variants = set(manifest["variants"])
    for arm in variants:
        arm_records = [record for record in manifest["cells"] if record["variant"] == arm]
        require(
            sorted(int(record["fold"]) for record in arm_records) == [0, 2],
            f"{arm} must declare one fixed endpoint on both folds",
        )
    selected: dict[int, dict[str, Any]] = {}
    records = {_key(record): record for record in manifest["cells"]}
    for selected_fold in (0, 2):
        endpoint_key = next(
            (key for key in outcomes if key.startswith(f"{variant}/F{selected_fold}/")), None
        )
        require(endpoint_key is not None, f"selected variant lacks F{selected_fold}: {variant}")
        require(
            outcomes[endpoint_key]["state"] == "completed",
            f"selected variant has a failed endpoint: {variant} F{selected_fold}",
        )
        selected[selected_fold] = records[endpoint_key]
    require(
        all(
            any(key.startswith(f"{arm}/F{fold}/") for key in outcomes)
            for arm in variants
            for fold in (0, 2)
        ),
        "stage does not declare every variant on both folds",
    )
    config_paths = {fold: root / selected[fold]["config"] for fold in (0, 2)}
    candidate_configs = {fold: read_object(config_paths[fold]) for fold in (0, 2)}
    left, right = candidate_configs[0].copy(), candidate_configs[2].copy()
    left.pop("expected_fold")
    right.pop("expected_fold")
    left.pop("seed")
    right.pop("seed")
    require(left == right, "selected F0/F2 recipes differ beyond their frozen fold and seed")
    score_hashes, endpoint_evidence = _selection_evidence(root, manifest, outcomes)
    pair = []
    for fold in (0, 2):
        record = selected[fold]
        key = _key(record)
        receipt = outcomes[key]
        pair.append(
            {
                "fold": fold,
                "id": record["id"],
                "seed": record["seed"],
                "endpoint": key,
                "config": record["config"],
                "config_sha256": record["config_sha256"],
                "run_dir": record["run_dir"],
                "split_manifest": record["split_manifest"],
                "split_sha256": record["split_sha256"],
                "score_sha256": receipt["score_sha256"],
            }
        )
    selection = {
        "schema": "wnn.high-compute-r4-selection.v1",
        "stage": stage,
        "campaign_sha256": campaign["campaign_sha256"],
        "stage_manifest_sha256": sha256_file(_stage_path(root, stage, "stage.json")),
        "variant": variant,
        "reason": reason.strip(),
        "automatic_selection": False,
        "selected_pair": pair,
        "candidate_config": candidate_configs[0],
        "candidate_config_sha256": canonical_sha256(candidate_configs[0]),
        "score_sha256": score_hashes,
        "endpoint_evidence": endpoint_evidence,
        "failed_attempts": sum(item["state"] == "failed" for item in outcomes.values()),
    }
    path = _stage_path(root, stage, "selection.json")
    raw_hash = write_json_new(path, selection)
    write_bytes_new(path.with_suffix(".sha256"), (raw_hash + "\n").encode("ascii"))
    validate_selection_receipt(root, campaign, selection)
    return selection


def validate_selection_receipt(
    campaign_root: Path,
    campaign: dict[str, Any],
    selection: dict[str, Any],
    stage_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate every referenced score and endpoint before a conditional child stage."""
    root = campaign_root.resolve(strict=True)
    stage = str(selection.get("stage"))
    require(stage in SELECTABLE_STAGES, "selection stage is not selectable")
    path = _stage_path(root, stage, "selection.json")
    require(
        selection.get("schema") == "wnn.high-compute-r4-selection.v1",
        "unsupported selection receipt",
    )
    require(read_object(path) == selection, "selection object differs from its immutable file")
    require(
        selection.get("campaign_sha256") == campaign_digest(root),
        "selection campaign binding changed",
    )
    require(
        selection.get("stage_manifest_sha256")
        == sha256_file(_stage_path(root, stage, "stage.json")),
        "selection stage manifest changed",
    )
    sidecar = path.with_suffix(".sha256")
    require(
        sidecar.read_text(encoding="utf-8").strip() == sha256_file(path),
        "selection receipt hash mismatch",
    )
    manifest = stage_manifest if stage_manifest is not None else load_stage(root, stage)
    require(
        selection.get("variant") in manifest["variants"], "selection variant is no longer declared"
    )
    outcomes, _ = _outcomes(root, campaign, stage, manifest)
    score_hashes, endpoint_evidence = _selection_evidence(root, manifest, outcomes)
    require(selection.get("score_sha256") == score_hashes, "selection score hash inventory changed")
    require(
        selection.get("endpoint_evidence") == endpoint_evidence,
        "selection endpoint or failure evidence changed",
    )
    require(
        selection.get("failed_attempts")
        == sum(item["state"] == "failed" for item in outcomes.values()),
        "selection failed-attempt count changed",
    )
    variant = selection["variant"]
    records = {_key(record): record for record in manifest["cells"]}
    selected = []
    for fold in (0, 2):
        key = next((item for item in outcomes if item.startswith(f"{variant}/F{fold}/")), None)
        require(
            key is not None and outcomes[key]["state"] == "completed",
            f"selected endpoint is not complete: {variant} F{fold}",
        )
        selected.append(records[key])
    require(
        len(selection.get("selected_pair", [])) == 2, "selection must name exactly the F0/F2 pair"
    )
    for item, record in zip(selection["selected_pair"], selected, strict=True):
        outcome = outcomes[_key(record)]
        require(item.get("endpoint") == _key(record), "selected pair endpoint changed")
        require(
            item.get("id") == record["id"]
            and item.get("fold") == record["fold"]
            and item.get("seed") == record["seed"]
            and item.get("config") == record["config"]
            and item.get("run_dir") == record["run_dir"]
            and item.get("split_manifest") == record["split_manifest"]
            and item.get("split_sha256") == record["split_sha256"],
            "selected pair identity changed",
        )
        require(
            item.get("config_sha256") == record["config_sha256"]
            and item.get("score_sha256") == outcome["score_sha256"],
            "selected pair endpoint hashes changed",
        )
    config = read_object(root / selected[0]["config"])
    other_config = read_object(root / selected[1]["config"])
    left, right = config.copy(), other_config.copy()
    left.pop("expected_fold")
    right.pop("expected_fold")
    left.pop("seed")
    right.pop("seed")
    require(left == right, "selected F0/F2 recipes differ beyond fold and seed")
    require(
        selection.get("candidate_config") == config
        and selection.get("candidate_config_sha256") == canonical_sha256(config),
        "selected candidate config changed",
    )
    require(
        bool(str(selection.get("reason", "")).strip())
        and selection.get("automatic_selection") is False,
        "selection reason or manual policy is missing",
    )
    return selection


def validate_stage_dependencies(
    root: Path, campaign: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Recheck parent stage, manual selection, G comparison, and their endpoint hashes."""
    dependencies = manifest.get("dependencies", {})
    if not dependencies:
        return
    parent = dependencies.get("parent_stage")
    require(parent in SELECTABLE_STAGES, "stage parent selection is missing or invalid")
    parent_manifest = load_stage(root, str(parent))
    parent_manifest_path = _stage_path(root, str(parent), "stage.json")
    require(
        sha256_file(parent_manifest_path) == dependencies["stage_manifest"]["sha256"],
        "parent stage manifest changed",
    )
    selection_path = _stage_path(root, str(parent), "selection.json")
    require(
        sha256_file(selection_path) == dependencies["selection"]["sha256"],
        "parent selection changed",
    )
    selection = read_object(selection_path)
    validate_selection_receipt(root, campaign, selection, stage_manifest=parent_manifest)
    require(
        selection["variant"] == dependencies["selected_variant"], "parent selected variant changed"
    )
    require(parent_manifest["stage"] == parent, "stage dependency points to a different parent")
    comparison = dependencies.get("G_comparison")
    if comparison is not None:
        comparison_path = _stage_path(root, "G", "comparison.json")
        require(
            sha256_file(comparison_path) == comparison["sha256"],
            "G comparison changed after H preparation",
        )
        validate_comparison_file(root, campaign, "G", comparison_path)
