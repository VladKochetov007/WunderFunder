"""Validate fixed fit endpoints and record immutable failed attempts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from campaign_io import (
    canonical_sha256,
    read_object,
    require,
    sha256_file,
)


def run_path(root: Path, record: dict[str, Any], relative: str = "") -> Path:
    """Resolve a run artifact and reject paths that leave the campaign root."""
    base = root / str(record["run_dir"])
    path = (base / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"run artifact escapes campaign: {path}")
    return path


def _fit_members(split: dict[str, Any]) -> list[list[str | int]]:
    """Return the canonical ordered fit-membership identity for a split."""
    return [["train", int(group)] for group in split["fit"]["train"]] + [
        ["valid", int(group)] for group in split["fit"]["valid"]
    ]


def validate_completed_endpoint(
    campaign_root: Path,
    campaign: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Require a complete final-epoch model, score, and source-bound evaluation export."""
    root = campaign_root.resolve(strict=True)
    config_path = (root / str(record["config"])).resolve(strict=True)
    config = read_object(config_path)
    require(
        canonical_sha256(config) == record["config_sha256"],
        f"endpoint config changed: {record['id']}",
    )
    split_path = Path(str(record["split_manifest"])).resolve(strict=True)
    split = read_object(split_path)
    split_hash = sha256_file(split_path)
    require(split_hash == record["split_sha256"], f"endpoint split changed: {record['id']}")
    require(int(split.get("fold", -1)) == int(record["fold"]), "endpoint split fold mismatch")

    status_path = run_path(root, record, "fit/terminal_status.json")
    model_path = run_path(root, record, "fit/model.pt")
    score_path = run_path(root, record, "fit/score.json")
    evaluation_path = run_path(root, record, "evaluation/manifest.json")
    status = read_object(status_path)
    identity = status.get("identity")
    require(isinstance(identity, dict), f"endpoint identity missing: {record['id']}")
    data_identity = identity.get("data")
    require(isinstance(data_identity, dict), f"endpoint data identity missing: {record['id']}")
    require(status.get("status") == "completed", f"endpoint is not completed: {record['id']}")
    require(
        status.get("completed_epochs") == config["epochs"],
        f"endpoint is not the fixed final epoch: {record['id']}",
    )
    require(
        identity.get("config_sha256") == canonical_sha256(config),
        f"endpoint config identity mismatch: {record['id']}",
    )
    require(
        identity.get("source_sha256") == campaign["runtime_source_sha256"],
        f"endpoint source identity mismatch: {record['id']}",
    )
    require(
        identity.get("world_size") == config["required_world_size"],
        f"endpoint world size mismatch: {record['id']}",
    )
    require(
        data_identity.get("fold") == record["fold"], f"endpoint data fold mismatch: {record['id']}"
    )
    require(
        data_identity.get("split_sha256") == split_hash,
        f"endpoint fit split mismatch: {record['id']}",
    )
    require(
        data_identity.get("fit_members") == _fit_members(split),
        f"endpoint fit membership mismatch: {record['id']}",
    )
    weight_hash = sha256_file(model_path)
    require(status.get("model_sha256") == weight_hash, f"endpoint weights changed: {record['id']}")

    score = read_object(score_path)
    score_body = score.get("score")
    producer = score.get("producer")
    require(
        isinstance(score_body, dict) and isinstance(producer, dict),
        f"endpoint score is incomplete: {record['id']}",
    )
    metric = score_body.get("metric_id")
    value = score_body.get("wp")
    require(metric == "global_wp_clip2_v1", f"endpoint metric mismatch: {record['id']}")
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"endpoint score is nonfinite: {record['id']}",
    )
    require(
        score.get("split_sha256") == split_hash, f"endpoint score split mismatch: {record['id']}"
    )
    require(
        score.get("sequences") == len(split["evaluation"]["valid"]),
        f"endpoint score coverage mismatch: {record['id']}",
    )
    require(
        producer.get("checkpoint_sha256") == weight_hash,
        f"score producer weights mismatch: {record['id']}",
    )
    require(producer.get("config") == config, f"score producer config mismatch: {record['id']}")
    provenance = producer.get("provenance")
    require(
        isinstance(provenance, dict) and provenance.get("identity") == identity,
        f"score producer identity mismatch: {record['id']}",
    )
    require(
        producer.get("split_sha256") == split_hash, f"score producer split mismatch: {record['id']}"
    )
    require(
        producer.get("fit_members") == _fit_members(split),
        f"score producer fit membership mismatch: {record['id']}",
    )

    evaluation = read_object(evaluation_path)
    require(
        evaluation.get("schema") == "sequence-predictions-v1"
        and evaluation.get("complete") is True,
        f"evaluation export incomplete: {record['id']}",
    )
    require(
        evaluation.get("membership") == "evaluation"
        and evaluation.get("split_sha256") == split_hash,
        f"evaluation export split mismatch: {record['id']}",
    )
    require(
        evaluation.get("producer") == producer,
        f"score and prediction producer differ: {record['id']}",
    )
    entries = evaluation.get("sequences")
    require(isinstance(entries, list), f"evaluation export has no sequence list: {record['id']}")
    observed = [(item.get("split"), int(item.get("row_group", -1))) for item in entries]
    expected = [("valid", int(group)) for group in split["evaluation"]["valid"]]
    require(
        len(set(observed)) == len(observed) and set(observed) == set(expected),
        f"evaluation export coverage mismatch: {record['id']}",
    )
    require(
        score.get("predictions_manifest_sha256") == sha256_file(evaluation_path),
        f"scored prediction manifest changed: {record['id']}",
    )
    components = {
        str(int(item["component"]))
        for item in split["sequences"]["valid"]
        if int(item["row_group"]) in set(split["evaluation"]["valid"])
    }
    moments = score.get("component_centered_moments")
    require(
        isinstance(moments, dict) and set(moments) == components,
        f"endpoint component moments mismatch: {record['id']}",
    )
    for component, values in moments.items():
        require(
            isinstance(values, list) and len(values) == 2,
            f"invalid component moments: {record['id']} {component}",
        )
        for row in values:
            require(
                isinstance(row, list) and len(row) == 6,
                f"invalid component moments: {record['id']} {component}",
            )
            require(
                all(isinstance(item, (int, float)) and math.isfinite(item) for item in row),
                f"nonfinite component moments: {record['id']} {component}",
            )

    return {
        "state": "completed",
        "id": record["id"],
        "variant": record["variant"],
        "fold": int(record["fold"]),
        "seed": int(record["seed"]),
        "run_dir": record["run_dir"],
        "config_sha256": record["config_sha256"],
        "split_sha256": split_hash,
        "score_wp": float(value),
        "metric_id": metric,
        "score_sha256": sha256_file(score_path),
        "status_sha256": sha256_file(status_path),
        "weights_sha256": weight_hash,
        "evaluation_manifest_sha256": sha256_file(evaluation_path),
    }
