"""Validate immutable outputs from one retained-epoch inference and scoring pass."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from epoch_contract import (
    canonical_sha256,
    file_sha256,
    read_object,
    safe_file,
)


def validate_prediction(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    index: dict[str, Any],
    index_sha256: str,
    inner_sha256: str,
) -> dict[str, Any]:
    """Require an evaluation export bound to the selected safe epoch entry."""
    producer = manifest.get("producer")
    admission = producer.get("epoch_admission") if isinstance(producer, dict) else None
    if (
        manifest.get("schema") != "sequence-predictions-v1"
        or manifest.get("complete") is not True
        or manifest.get("membership") != "evaluation"
        or manifest.get("split_sha256") != inner_sha256
        or not isinstance(producer, dict)
        or producer.get("kind") != "retained_epoch"
        or producer.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        or producer.get("split_sha256") != inner_sha256
        or not isinstance(admission, dict)
        or admission.get("index_sha256") != index_sha256
        or admission.get("index") != index
        or admission.get("entry") != entry
    ):
        raise ValueError(
            f"prediction export does not bind retained epoch {entry['completed_epochs']}"
        )
    return producer


def validate_score(
    score: dict[str, Any],
    producer: dict[str, Any],
    prediction_manifest: Path,
    inner_sha256: str,
) -> dict[str, Any]:
    """Require a finite pooled official score linked to the exact export manifest."""
    metric = score.get("score")
    if (
        score.get("split_sha256") != inner_sha256
        or score.get("predictions_manifest_sha256") != file_sha256(prediction_manifest)
        or score.get("producer") != producer
        or not isinstance(metric, dict)
        or metric.get("metric_id") is None
        or type(metric.get("selected_rows")) is not int
        or metric["selected_rows"] < 1
        or any(
            not isinstance(metric.get(name), (float, int))
            or isinstance(metric.get(name), bool)
            or not math.isfinite(metric[name])
            for name in ("wp", "t0", "t1")
        )
        or not -1.000000001 <= metric["wp"] <= 1.000000001
    ):
        raise ValueError("official score receipt failed validation")
    return metric


def load_epoch_record(
    output: Path,
    entry: dict[str, Any],
    fold: int,
    fit: dict[str, Any],
    split_binding: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate an already completed epoch before allowing a sweep to resume past it."""
    completed = entry["completed_epochs"]
    relative_dir = f"epochs/epoch-{completed:06d}"
    receipt_path = safe_file(output, f"{relative_dir}/epoch.json")
    prediction_path = safe_file(output, f"{relative_dir}/predictions/manifest.json")
    score_path = safe_file(output, f"{relative_dir}/score.json")
    receipt, _ = read_object(receipt_path)
    manifest, _ = read_object(prediction_path)
    producer = validate_prediction(
        manifest,
        entry,
        fit["index"],
        fit["index_sha256"],
        split_binding["inner_sha256"],
    )
    sequences = manifest.get("sequences")
    expected_members = split_binding.get("evaluation_members")
    if not isinstance(sequences, list) or not isinstance(expected_members, list):
        raise ValueError("completed prediction export is missing sequence membership")
    members = []
    for item in sequences:
        if not isinstance(item, dict):
            raise ValueError("completed prediction sequence entry is malformed")
        member = [item.get("split"), item.get("row_group")]
        if type(member[1]) is not int or not isinstance(member[0], str):
            raise ValueError("completed prediction sequence member is malformed")
        path = safe_file(output, f"{relative_dir}/predictions/{item.get('file')}")
        if not isinstance(item.get("sha256"), str) or file_sha256(path) != item["sha256"]:
            raise ValueError("completed prediction shard hash mismatch")
        members.append(member)
    if members != expected_members:
        raise ValueError(
            "completed prediction export does not cover the bound inner evaluation set"
        )

    score, _ = read_object(score_path)
    metric = validate_score(score, producer, prediction_path, split_binding["inner_sha256"])
    prediction_hash, score_hash = file_sha256(prediction_path), file_sha256(score_path)
    expected_receipt = {
        "schema": "wnn.recovery-r5-epoch-score.v1",
        "fold": fold,
        "completed_epochs": completed,
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "config_identity_sha256": fit["identity"]["config_sha256"],
        "outer_split_sha256": split_binding["outer_sha256"],
        "inner_split_sha256": split_binding["inner_sha256"],
        "epoch_index_sha256": fit["index_sha256"],
        "prediction_manifest_sha256": prediction_hash,
        "score_sha256": score_hash,
        "score": metric,
    }
    if receipt != expected_receipt:
        raise ValueError(f"completed epoch receipt changed at completed epoch {completed}")
    return {
        "epoch": entry["epoch"],
        "completed_epochs": completed,
        "checkpoint_path": entry["checkpoint_path"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "prediction_manifest_sha256": prediction_hash,
        "score_sha256": score_hash,
        "score": metric,
        "receipt_sha256": file_sha256(receipt_path),
    }


def completed_sweep(
    output: Path, plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Accept an existing final sweep only if it exactly lists verified receipts."""
    path = output / "sweep.json"
    if not path.exists():
        return None
    sweep, _ = read_object(safe_file(output, "sweep.json"))
    unsigned = {
        "schema": "wnn.recovery-r5-epoch-sweep.v1",
        "status": "complete",
        "fold": plan["fold"],
        "expected_epochs": plan["expected_epochs"],
        "plan_sha256": file_sha256(output / "plan.json"),
        "runner_sha256": plan["runner_sha256"],
        "epoch_index_sha256": plan["epoch_index_sha256"],
        "config_identity_sha256": plan["config_identity_sha256"],
        "outer_split_sha256": plan["outer_split_sha256"],
        "inner_split_sha256": plan["inner_split_sha256"],
        "epochs": records,
    }
    expected = unsigned | {"sweep_sha256": canonical_sha256(unsigned)}
    if len(records) != plan["expected_epochs"] or sweep != expected:
        raise ValueError("existing completed sweep is detached from verified epoch receipts")
    return sweep
