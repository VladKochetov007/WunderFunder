"""Fixed-endpoint comparisons and immutable descriptive receipts."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from campaign_io import (
    read_object,
    require,
    sha256_file,
    write_bytes_new,
    write_json_new,
)
from endpoints import validate_completed_endpoint
from failure_receipts import failure_path, validate_failure_receipt
from materialize import load_stage
from paired_metrics import _paired


def _relative(root: Path, path: Path) -> str:
    """Serialize a campaign-local receipt path using POSIX separators."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _stage_path(root: Path, stage: str, filename: str) -> Path:
    """Return one stable stage receipt path."""
    return root / "stages" / stage / filename


def _key(record: dict[str, Any]) -> str:
    """Create a stable identity for every attempted variant/fold/seed endpoint."""
    return f"{record['variant']}/F{record['fold']}/S{record['seed']}"


def _outcomes(
    root: Path,
    campaign: dict[str, Any],
    stage: str,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate every declared endpoint or terminal failure in one stage."""
    stage_manifest = manifest if manifest is not None else load_stage(root, stage)
    outcomes: dict[str, dict[str, Any]] = {}
    for record in stage_manifest["cells"]:
        key = _key(record)
        require(key not in outcomes, f"duplicate stage endpoint key: {key}")
        failed = failure_path(root, record).is_file()
        if failed:
            outcome = validate_failure_receipt(root, campaign, record)
            require(
                not (root / record["run_dir"] / "fit" / "score.json").exists(),
                f"failed cell also has a score: {record['id']}",
            )
        else:
            outcome = validate_completed_endpoint(root, campaign, record)
        outcomes[key] = outcome
    return outcomes, stage_manifest


def _score(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Read one score receipt after endpoint validation has bound its hash."""
    path = root / record["run_dir"] / "fit" / "score.json"
    return read_object(path)


def _with_scores(
    root: Path,
    manifest: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach validated score payloads while preserving failure outcomes."""
    scored: dict[str, dict[str, Any]] = {}
    records = {_key(record): record for record in manifest["cells"]}
    for key, outcome in outcomes.items():
        if outcome["state"] == "completed":
            scored[key] = {"evidence": outcome, "score": _score(root, records[key])}
    return scored


def _build_comparison(
    root: Path,
    campaign: dict[str, Any],
    stage: str,
    manifest: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build descriptive paired outcomes without choosing a recipe."""
    scored = _with_scores(root, manifest, outcomes)
    variants = list(manifest["variants"])
    comparisons: list[dict[str, Any]] = []
    if stage in "BCDEF":
        for left_variant, right_variant in itertools.combinations(variants, 2):
            for fold in (0, 2):
                left = next(
                    (
                        value
                        for key, value in scored.items()
                        if key.startswith(f"{left_variant}/F{fold}/")
                    ),
                    None,
                )
                right = next(
                    (
                        value
                        for key, value in scored.items()
                        if key.startswith(f"{right_variant}/F{fold}/")
                    ),
                    None,
                )
                if left is None or right is None:
                    comparisons.append(
                        {
                            "fold": fold,
                            "candidate": left_variant,
                            "reference": right_variant,
                            "state": "not_comparable_due_to_failed_attempt",
                        }
                    )
                    continue
                comparisons.append(
                    {
                        "fold": fold,
                        "candidate": left_variant,
                        "reference": right_variant,
                        "state": "paired",
                        **_paired(left, right, 1729 + fold),
                    }
                )
    elif stage == "G":
        require(
            set(variants) == {"candidate", "reference"},
            "G must compare candidate and reference arms",
        )
        alias = bool(manifest["accounting"].get("candidate_aliases_reference"))
        for fold in (0, 2):
            fold_deltas = []
            for seed in (301001, 301002, 301003):
                candidate = scored[f"candidate/F{fold}/S{seed}"]
                reference = scored[f"reference/F{fold}/S{seed}"]
                if alias:
                    require(
                        candidate["evidence"]["score_sha256"]
                        == reference["evidence"]["score_sha256"],
                        "G alias arms do not share the same score artifact",
                    )
                    pair = {
                        "candidate_score_sha256": candidate["evidence"]["score_sha256"],
                        "reference_score_sha256": reference["evidence"]["score_sha256"],
                        "candidate_wp": candidate["evidence"]["score_wp"],
                        "reference_wp": reference["evidence"]["score_wp"],
                        "candidate_minus_reference": 0.0,
                        "component_bootstrap": None,
                        "interpretation": "Exact config alias; this is one fit, not an independent paired replication.",
                    }
                else:
                    pair = _paired(candidate, reference, 1729 + fold * 10 + seed % 10)
                fold_deltas.append(pair["candidate_minus_reference"])
                comparisons.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "state": "exact_alias" if alias else "paired",
                        **pair,
                    }
                )
            comparisons.append(
                {
                    "fold": fold,
                    "state": "fold_descriptive_mean",
                    "candidate_minus_reference": sum(fold_deltas) / len(fold_deltas),
                }
            )
    else:
        require(
            stage == "H" and set(variants) == {"labels", "single_kd", "mean3_kd"},
            "unsupported H comparison arms",
        )
        for fold in (0, 2):
            label = scored[f"labels/F{fold}/S{401001 if fold == 0 else 401002}"]
            for variant in ("single_kd", "mean3_kd"):
                student = scored[f"{variant}/F{fold}/S{401001 if fold == 0 else 401002}"]
                comparisons.append(
                    {
                        "fold": fold,
                        "candidate": variant,
                        "reference": "labels",
                        "state": "paired",
                        **_paired(student, label, 1729 + fold),
                    }
                )
    score_hashes = {
        key: value["score_sha256"]
        for key, value in outcomes.items()
        if value["state"] == "completed"
    }
    return {
        "schema": "wnn.high-compute-r4-comparison.v1",
        "stage": stage,
        "campaign_sha256": campaign["campaign_sha256"],
        "stage_manifest_sha256": sha256_file(_stage_path(root, stage, "stage.json")),
        "automatic_selection": False,
        "variants": variants,
        "endpoint_outcomes": outcomes,
        "score_sha256": score_hashes,
        "paired_outcomes": comparisons,
        "interpretation": "Fixed-endpoint paired evidence; intervals condition on the declared components and fits. F0/F2 are exposed outer folds. No automatic winner or generalization claim is produced.",
    }


def compare_stage(campaign_root: Path, stage: str) -> dict[str, Any]:
    """Write one immutable comparison receipt for every completed paired outcome."""
    root = campaign_root.expanduser().resolve(strict=True)
    from campaign_io import load_campaign

    campaign = load_campaign(root)
    manifest = load_stage(root, stage)
    path = _stage_path(root, stage, "comparison.json")
    if path.exists():
        return validate_comparison_file(root, campaign, stage, path)
    outcomes, _ = _outcomes(root, campaign, stage, manifest)
    if stage in ("G", "H"):
        require(
            all(item["state"] == "completed" for item in outcomes.values()),
            f"{stage} comparison needs every fixed endpoint completed",
        )
    comparison = _build_comparison(root, campaign, stage, manifest, outcomes)
    raw_hash = write_json_new(path, comparison)
    write_bytes_new(path.with_suffix(".sha256"), (raw_hash + "\n").encode("ascii"))
    return comparison


def validate_comparison_file(
    campaign_root: Path,
    campaign: dict[str, Any],
    stage: str,
    path: Path,
) -> dict[str, Any]:
    """Rebuild and verify a comparison against current immutable endpoint hashes."""
    root = campaign_root.resolve(strict=True)
    expected_path = _stage_path(root, stage, "comparison.json")
    require(
        path.resolve() == expected_path.resolve(),
        "comparison path is not the campaign's fixed stage receipt",
    )
    comparison = read_object(path)
    require(
        comparison.get("schema") == "wnn.high-compute-r4-comparison.v1"
        and comparison.get("stage") == stage,
        "invalid comparison receipt",
    )
    sidecar = path.with_suffix(".sha256")
    require(
        sidecar.read_text(encoding="utf-8").strip() == sha256_file(path),
        f"comparison hash mismatch: {stage}",
    )
    manifest = load_stage(root, stage)
    outcomes, _ = _outcomes(root, campaign, stage, manifest)
    if stage in ("G", "H"):
        require(
            all(item["state"] == "completed" for item in outcomes.values()),
            f"{stage} comparison contains an incomplete fixed endpoint",
        )
    require(
        comparison.get("campaign_sha256") == campaign["campaign_sha256"],
        "comparison campaign changed",
    )
    require(
        comparison.get("stage_manifest_sha256")
        == sha256_file(_stage_path(root, stage, "stage.json")),
        f"comparison stage changed: {stage}",
    )
    require(
        comparison.get("endpoint_outcomes") == outcomes,
        f"comparison endpoint evidence changed: {stage}",
    )
    expected_scores = {
        key: value["score_sha256"]
        for key, value in outcomes.items()
        if value["state"] == "completed"
    }
    require(
        comparison.get("score_sha256") == expected_scores,
        f"comparison score hashes changed: {stage}",
    )
    require(comparison.get("automatic_selection") is False, "comparison cannot select a winner")
    pairs = comparison.get("paired_outcomes")
    require(isinstance(pairs, list) and pairs, f"comparison pair outcomes are missing: {stage}")
    if stage == "G":
        require(len(pairs) == 8, "G comparison must include six seed pairs and two fold summaries")
    elif stage == "H":
        require(len(pairs) == 4, "H comparison must include both KD arms on both folds")
    return comparison
