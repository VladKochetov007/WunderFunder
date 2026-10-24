"""Write and validate complete per-cell configs and planned output paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import (
    canonical_sha256,
    read_object,
    relative_campaign_path,
    require,
    sha256_file,
    write_json_new,
)
from catalog import profile_one_config, profile_peak_config, validate_config


def _cell_path(root: Path, value: str) -> Path:
    """Resolve a campaign-relative cell artifact and reject path traversal."""
    path = relative_campaign_path(root, value).resolve()
    require(path.is_relative_to(root.resolve()), f"cell path escapes campaign root: {value}")
    return path


def _relative(root: Path, path: Path) -> str:
    """Serialize a campaign artifact path using stable POSIX separators."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _split_role(campaign: dict[str, Any], fold: int, inner: bool) -> dict[str, Any]:
    """Return the exact outer or inner split binding for a fold."""
    collection = campaign["inner_splits"] if inner else campaign["outer_splits"]
    return collection[f"F{fold}"]


def _split_path(campaign: dict[str, Any], fold: int, inner: bool) -> tuple[str, str]:
    """Return the normalized absolute split path and frozen hash."""
    role = _split_role(campaign, fold, inner)
    path = Path(role["source_path"]).resolve(strict=True)
    require(sha256_file(path) == role["sha256"], f"split changed: {path}")
    return str(path), str(role["sha256"])


def _new_cell(
    campaign_root: Path,
    campaign: dict[str, Any],
    cell_id: str,
    variant: str,
    fold: int,
    seed: int,
    kind: str,
    config: dict[str, Any],
    *,
    inner_split: bool,
    cache_inputs: list[str] | None = None,
    cache_producers: list[dict[str, Any]] | None = None,
    teacher_cache: str | None = None,
) -> dict[str, Any]:
    """Write one full config and its two profile variants to a new cell folder."""
    validate_config(config)
    cell_dir = campaign_root / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=False)
    config_path = cell_dir / "config.json"
    one_path = cell_dir / "profile_one.json"
    peak_path = cell_dir / "profile_peak.json"
    config_raw_hash = write_json_new(config_path, config)
    one_config = profile_one_config(config)
    peak_config = profile_peak_config(config)
    one_raw_hash = write_json_new(one_path, one_config)
    peak_raw_hash = write_json_new(peak_path, peak_config)
    split_path, split_hash = _split_path(campaign, fold, inner_split)
    record: dict[str, Any] = {
        "id": cell_id,
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "kind": kind,
        "config": _relative(campaign_root, config_path),
        "profile_one": _relative(campaign_root, one_path),
        "profile_peak": _relative(campaign_root, peak_path),
        "split_manifest": split_path,
        "split_sha256": split_hash,
        "split_manifest_sha256": split_hash,
        "run_dir": f"runs/{cell_id}",
        "teacher_cache": teacher_cache
        if teacher_cache is not None
        else (f"runs/{cell_id}/teacher-cache" if kind == "teacher" else None),
        "reused": False,
        "artifact_state": "planned_output_paths",
        "config_sha256": canonical_sha256(config),
        "config_file_sha256": config_raw_hash,
        "profile_one_sha256": one_raw_hash,
        "profile_peak_sha256": peak_raw_hash,
        "cache_inputs": cache_inputs or [],
        "cache_producers": cache_producers or [],
        "parent_run_dirs": list(
            dict.fromkeys(item["producer_run_dir"] for item in cache_producers or [])
        ),
    }
    return record


def _reused_cell(source: dict[str, Any], variant: str, stage: str) -> dict[str, Any]:
    """Declare exact prior endpoint reuse with the same ID and every artifact path."""
    result = source.copy()
    result["variant"] = variant
    result["reused"] = True
    result["reuse_of"] = source["id"]
    result["reuse_source_stage"] = stage
    result["failure_origin_stage"] = source.get("failure_origin_stage", source.get("stage", stage))
    return result


def _validate_record(campaign_root: Path, campaign: dict[str, Any], record: dict[str, Any]) -> None:
    """Check config, profile, split and campaign-relative output bindings."""
    require(record["kind"] in ("teacher", "student"), "unknown cell kind")
    config = read_object(_cell_path(campaign_root, record["config"]))
    validate_config(config)
    require(
        canonical_sha256(config) == record["config_sha256"],
        f"config identity changed: {record['id']}",
    )
    require(
        sha256_file(_cell_path(campaign_root, record["config"])) == record["config_file_sha256"],
        f"config bytes changed: {record['id']}",
    )
    for name, digest in (
        ("profile_one", "profile_one_sha256"),
        ("profile_peak", "profile_peak_sha256"),
    ):
        path = _cell_path(campaign_root, record[name])
        require(
            sha256_file(path) == record[digest], f"profile config changed: {record['id']} {name}"
        )
        profile = read_object(path)
        validate_config(profile)
        expected_profile = (
            profile_one_config(config) if name == "profile_one" else profile_peak_config(config)
        )
        require(
            canonical_sha256(profile) == canonical_sha256(expected_profile),
            f"profile config mismatch: {record['id']} {name}",
        )
    split_path = Path(record["split_manifest"]).resolve(strict=True)
    require(
        sha256_file(split_path) == record["split_manifest_sha256"],
        f"split manifest changed: {record['id']}",
    )
    require(
        int(config["expected_fold"]) == int(record["fold"]), f"config fold mismatch: {record['id']}"
    )
    require(
        Path(record["run_dir"]).parts[0] == "runs",
        f"run directory outside campaign namespace: {record['id']}",
    )
    _cell_path(campaign_root, record["run_dir"])
    if record["teacher_cache"] is not None:
        _cell_path(campaign_root, record["teacher_cache"])
    require(
        record.get("cache_inputs", [])
        == [item["path"] for item in record.get("cache_producers", [])],
        "cache inputs differ from bound G producers",
    )
    for cache_path in record.get("cache_inputs", []):
        _cell_path(campaign_root, cache_path)
    for producer in record.get("cache_producers", []):
        _cell_path(campaign_root, producer["producer_run_dir"])
        _cell_path(campaign_root, producer["checkpoint_path"])
