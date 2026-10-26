"""Revalidate fit, sweep, split, and paired-score bytes for R5 retention."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import (
    INDEX_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    read_object,
    safe_file,
    validate_split_pair,
)
from retention_contract import _plan_path
from retention_inventory import _tree_artifacts, _validate_prediction_inventory


def _verify_artifacts(root: Path, expected: list[dict[str, Any]], *, fit: bool) -> None:
    """Require the retained file inventory and every recorded size and digest."""
    actual = _tree_artifacts(root, fit=fit)
    if fit:
        actual = [
            item
            for item in actual
            if item["path"] not in {"retention_intent.json", "retention_complete.json"}
        ]
    if [item["path"] for item in actual] != [item["path"] for item in expected]:
        raise ValueError(f"retained metadata inventory changed under {root}")
    if any(
        actual_item != expected_item
        for actual_item, expected_item in zip(actual, expected, strict=True)
    ):
        raise ValueError(f"retained metadata hash or size changed under {root}")


def _verify_checkpoints(fold: dict[str, Any], allow_missing: bool) -> None:
    """Verify selected and scheduled checkpoint files; accept only intent-listed deletions."""
    fit_dir = Path(fold["fit_dir"]).resolve(strict=True)
    directory = fit_dir / "epoch_checkpoints"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint directory is missing or unsafe")
    selected = fold["selected_checkpoint"]
    remove = {item["path"]: item for item in fold["remove"]}
    expected = {selected["path"]} | set(remove)
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
            raise ValueError("checkpoint directory contains an unsafe or unexpected file")
        observed.add(path.relative_to(fit_dir).as_posix())
    if not observed <= expected or selected["path"] not in observed:
        raise ValueError("checkpoint directory contains an unplanned or missing selected file")
    if not allow_missing and observed != expected:
        raise ValueError("checkpoint deletion is incomplete without a durable retention intent")
    for relative, entry in [(selected["path"], selected), *remove.items()]:
        path = fit_dir / relative
        if not path.exists():
            if relative in remove and allow_missing:
                continue
            raise ValueError("selected checkpoint is missing")
        safe = safe_file(fit_dir, relative)
        if safe.stat().st_size != entry["size"] or file_sha256(safe) != entry["sha256"]:
            raise ValueError(f"checkpoint bytes changed: {relative}")


def _verify_sweep_artifacts(fold: dict[str, Any]) -> None:
    """Recheck the immutable curve metadata and every manifest-listed prediction shard."""
    root = Path(fold["sweep_dir"]).resolve(strict=True)
    _verify_artifacts(root, fold["sweep_artifacts"], fit=False)
    for item in fold["curve"]:
        completed = item["completed_epochs"]
        relative = f"epochs/epoch-{completed:06d}/predictions/manifest.json"
        manifest, raw = read_object(safe_file(root, relative))
        if hashlib.sha256(raw).hexdigest() != item["prediction_manifest_sha256"]:
            raise ValueError("prediction manifest changed after the retention plan")
        prefix = f"epochs/epoch-{completed:06d}/predictions"
        seen: set[str] = set()
        sequences = manifest.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError("prediction manifest lacks its sequence inventory")
        _validate_prediction_inventory(root, completed, manifest)
        for sequence in sequences:
            if not isinstance(sequence, dict) or not isinstance(sequence.get("file"), str):
                raise ValueError("prediction manifest sequence entry is malformed")
            name = sequence["file"]
            if name in seen:
                raise ValueError("prediction manifest contains duplicate sequence files")
            seen.add(name)
            path = safe_file(root, f"{prefix}/{name}")
            if file_sha256(path) != sequence.get("sha256"):
                raise ValueError(f"prediction shard changed: {relative}/{name}")


def _verify_index_snapshots(fit_dir: Path, index: dict[str, Any]) -> None:
    """Verify every immutable prefix, its canonical bytes, and hash chain."""
    previous_hash = None
    entries = index["entries"]
    for completed in range(1, 31):
        path = safe_file(fit_dir, f"epoch_indexes/epoch-{completed:06d}.json")
        prefix, raw = read_object(path)
        unsigned = {key: value for key, value in prefix.items() if key != "index_sha256"}
        if (
            set(prefix) != set(index)
            or raw != canonical_bytes(prefix) + b"\n"
            or prefix.get("schema") != INDEX_SCHEMA
            or prefix.get("completed_epochs") != completed
            or prefix.get("identity") != index["identity"]
            or prefix.get("entries") != entries[:completed]
            or prefix.get("global_step") != entries[completed - 1]["global_step"]
            or prefix.get("previous_index_sha256") != previous_hash
            or prefix.get("index_sha256") != canonical_sha256(unsigned)
        ):
            raise ValueError(f"immutable epoch index prefix changed at epoch {completed}")
        previous_hash = prefix["index_sha256"]


def _verify_fit_identity(fold: dict[str, Any]) -> None:
    """Revalidate component splits, config, complete index chain, and fixed-final model."""
    fit_dir = Path(fold["fit_dir"]).resolve(strict=True)
    _verify_artifacts(fit_dir, fold["fit_artifacts"], fit=True)
    split = validate_split_pair(Path(fold["outer_split"]), Path(fold["inner_split"]), fold["fold"])
    if (
        split["outer_sha256"] != fold["outer_split_sha256"]
        or split["inner_sha256"] != fold["inner_split_sha256"]
    ):
        raise ValueError("retention split ancestry changed")
    config_path = _plan_path(fold["config_path"])
    config, raw_config = read_object(config_path)
    if hashlib.sha256(raw_config).hexdigest() != fold["config_file_sha256"]:
        raise ValueError("retention config bytes changed")
    model = config.get("model")
    if not isinstance(model, dict) or config.get("precision_mode") != "bf16_bf16x9":
        raise ValueError("retention config precision identity is malformed")
    normalized = dict(config) | {
        "model": dict(model) | {"precision_mode": config["precision_mode"]}
    }
    if canonical_sha256(normalized) != fold["identity"].get("config_sha256"):
        raise ValueError("retention config differs from the indexed fit identity")

    index, index_raw = read_object(safe_file(fit_dir, "epoch_index.json"))
    unsigned_index = {key: value for key, value in index.items() if key != "index_sha256"}
    if (
        hashlib.sha256(index_raw).hexdigest() != fold["index_file_sha256"]
        or canonical_sha256(unsigned_index) != fold["index_sha256"]
        or index.get("identity") != fold["identity"]
        or index.get("completed_epochs") != 30
        or index.get("entries") != fold["index_entries"]
        or len(index.get("entries", [])) != 30
        or index.get("global_step") != index["entries"][-1].get("global_step")
    ):
        raise ValueError("retention epoch index changed or is incomplete")
    immutable = safe_file(fit_dir, "epoch_indexes/epoch-000030.json")
    if immutable.read_bytes() != index_raw:
        raise ValueError("latest and immutable terminal epoch indexes differ")
    _verify_index_snapshots(fit_dir, index)
    terminal, _ = read_object(safe_file(fit_dir, "terminal_status.json"))
    final_model = safe_file(fit_dir, "model.pt")
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != 30
        or terminal.get("identity") != fold["identity"]
        or terminal.get("model_sha256") != fold["final_model_sha256"]
        or file_sha256(final_model) != fold["final_model_sha256"]
    ):
        raise ValueError("fixed-final model/terminal receipt changed")


def _verify_selection_curve(plan: dict[str, Any]) -> None:
    """Recompute the shared epoch from both bound 30-epoch inner WP curves."""
    selection_path = _plan_path(plan["selection_path"])
    selection, raw = read_object(selection_path)
    if (
        hashlib.sha256(raw).hexdigest() != plan["selection_file_sha256"]
        or selection != plan["selection"]
    ):
        raise ValueError("selection file differs from the immutable retention plan")
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if canonical_sha256(unsigned) != selection.get("selection_sha256"):
        raise ValueError("selection self-hash is invalid")
    score_curves: dict[int, list[float]] = {}
    for fold in plan["folds"]:
        root = Path(fold["sweep_dir"])
        values = []
        for record in fold["curve"]:
            relative = f"epochs/epoch-{record['completed_epochs']:06d}/score.json"
            score, raw_score = read_object(safe_file(root, relative))
            if hashlib.sha256(raw_score).hexdigest() != record["score_sha256"]:
                raise ValueError("score bytes changed after selection")
            metric = score.get("score")
            if metric != record["score"]:
                raise ValueError("curve score differs from its original pooled score receipt")
            value = metric.get("wp") if isinstance(metric, dict) else None
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("retention score curve contains a nonfinite WP")
            values.append(float(value))
        score_curves[fold["fold"]] = values
    means = [
        (left + right) / 2.0 for left, right in zip(score_curves[0], score_curves[2], strict=True)
    ]
    selected = max(range(len(means)), key=lambda index: (means[index], -index)) + 1
    if (
        selection.get("selected_completed_epochs") != selected
        or selection.get("mean_inner_wp_by_completed_epoch") != means
        or selection.get("selected_mean_inner_wp") != means[selected - 1]
        or selection.get("curve_score_hashes")
        != {
            f"F{fold['fold']}": [item["score_sha256"] for item in fold["curve"]]
            for fold in plan["folds"]
        }
    ):
        raise ValueError("shared-epoch selection no longer matches the full paired score curve")
    for fold in plan["folds"]:
        chosen = selection["folds"][f"F{fold['fold']}"]["selected_model"]
        selected_record = fold["curve"][selected - 1]
        if chosen != {
            "checkpoint_path": fold["selected_checkpoint"]["path"],
            "checkpoint_sha256": fold["selected_checkpoint"]["sha256"],
            "score_sha256": selected_record["score_sha256"],
            "prediction_manifest_sha256": selected_record["prediction_manifest_sha256"],
            "score": selected_record["score"],
        }:
            raise ValueError("selected fold checkpoint differs from the paired epoch curve")


def _verify_plan_state(plan: dict[str, Any], *, allow_missing: bool) -> None:
    """Check every retained byte and ensure only intent-listed checkpoints are absent."""
    for fold in plan["folds"]:
        _verify_fit_identity(fold)
        _verify_checkpoints(fold, allow_missing)
        _verify_sweep_artifacts(fold)
    _verify_selection_curve(plan)


def _verify_removed_absent(plan: dict[str, Any]) -> None:
    """Require every explicitly pruned checkpoint to remain absent after completion."""
    for fold in plan["folds"]:
        fit_dir = Path(fold["fit_dir"])
        for entry in fold["remove"]:
            path = fit_dir / entry["path"]
            if path.is_symlink() or path.exists():
                raise ValueError(
                    f"completed retention checkpoint unexpectedly exists: {entry['path']}"
                )
