from __future__ import annotations

from pathlib import Path
from typing import Any

from packet_contract import (
    digest,
    file_hash,
    members,
    normalized_config,
    read,
    require,
    source_files,
)

# These heterogeneous JSON identities originate in the canonical trainer/exporter.


def validate_identity(root: Path, cell: str, identity: dict[str, Any]) -> None:
    """Bind a cell to the nominated config, source and full source-holdout membership."""
    fit, evaluation = members(root)
    require(identity.get("config_sha256") == digest(normalized_config(cell)), "Config mismatch")
    require(identity.get("source_files") == source_files(), "Source files mismatch")
    require(identity.get("source_sha256") == digest(source_files()), "Source digest mismatch")
    require(identity.get("world_size") == 2, "Two ranks required")
    data = identity["data"]
    require(identity.get("data_sha256") == digest(data), "Data identity digest mismatch")
    require(data.get("split_sha256") == file_hash(root / "split.json"), "Split digest mismatch")
    require(data.get("fold") == 0, "Source holdout must use protocol slot zero")
    require(data.get("fit_members") == fit, "Fit membership differs from source holdout")
    require(data.get("evaluation_members") == evaluation, "Evaluation must cover full valid")
    require(data.get("input_schema") == "raw112-v1", "Unexpected input features")
    require(data.get("training_mask") == "all_after_warmup", "Training mask changed")
    require(data.get("normalization", {}).get("scope") == "all_fit", "Fit-only scaling required")
    if not cell.startswith("kd_"):
        require(data.get("teacher") is None, "Teacher and label control cannot reuse a teacher")


def validate_terminal(root: Path, cell: str) -> dict[str, Any]:
    """Accept only this cell's newly completed fixed-final five-epoch checkpoint."""
    directory = root / cell
    terminal = read(directory / "fit/terminal_status.json")
    require(terminal.get("status") == "completed", "Cell has not completed")
    require(terminal.get("completed_epochs") == 5, "Only fixed FINAL5 is admitted")
    require("epoch_index_sha256" not in terminal, "No validation-selected retained epoch")
    identity = terminal["identity"]
    validate_identity(root, cell, identity)
    approved = read(directory / "approved_profile.json")
    require(approved.get("approved") is True, "Missing reviewed profile")
    require(identity == approved.get("identity"), "Terminal differs from reviewed profile")
    require(
        terminal.get("model_sha256") == file_hash(directory / "fit/model.pt"), "Final model changed"
    )
    return terminal


def _validate_entries(meta: dict[str, Any], expected: list[list[object]]) -> None:
    """Reject duplicated, foreign or missing members in an exported target set."""
    entries = meta.get("sequences")
    require(isinstance(entries, list), "Missing prediction entries")
    observed = [[item["split"], item["row_group"]] for item in entries]
    require(len(observed) == len(expected), "Prediction member count differs")
    require(sorted(observed) == sorted(expected), "Prediction membership differs")
    names = [item["file"] for item in entries]
    require(len(set(names)) == len(names), "Prediction members reuse files")
    require(
        all(Path(name).name == name and name not in (".", "..") for name in names),
        "Unsafe array path",
    )


def validate_cache(root: Path, cell: str) -> dict[str, Any]:
    """Bind KD exclusively to the designated same-seed teacher's fresh final model."""
    teacher = "teacher_" + cell.split("_", 1)[1]
    terminal = validate_terminal(root, teacher)
    directory = root / teacher / "cache"
    meta = read(directory / "manifest.json")
    fit, _ = members(root)
    require(meta.get("schema") == "sequence-teacher-cache-v1", "Canonical teacher cache required")
    require(
        meta.get("complete") is True and meta.get("membership") == "fit", "Incomplete fit cache"
    )
    require(
        meta.get("input_schema") == "raw112-v1" and meta.get("hint_dim") == 0, "Wrong cache layout"
    )
    require(meta.get("sequence_length") == 20_000, "Wrong sequence length")
    require(meta.get("split_sha256") == file_hash(root / "split.json"), "Foreign cache split")
    require(meta.get("world_size") == 2, "Cache must use the frozen two-rank exporter")
    producer = meta["producer"]
    require(
        producer.get("checkpoint_sha256") == terminal["model_sha256"], "Foreign teacher weights"
    )
    require(producer.get("config") == normalized_config(teacher), "Foreign teacher recipe/seed")
    require(producer.get("fit_members") == fit, "Teacher fit members differ")
    require(producer.get("teacher") is None, "Teacher must be trained from labels")
    require(
        producer.get("provenance", {}).get("identity") == terminal["identity"],
        "Foreign teacher provenance",
    )
    require(producer.get("provenance", {}).get("completed_epochs") == 5, "Teacher is not FINAL5")
    _validate_entries(meta, fit)
    for entry in meta["sequences"]:
        path = directory / entry["file"]
        require(not path.is_symlink(), "Cache arrays cannot be symlinks")
        require(file_hash(path) == entry["sha256"], "Teacher cache array changed")
    return {
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "checkpoint_sha256": terminal["model_sha256"],
        "producer": producer,
    }


def validate_evaluation(root: Path, cell: str) -> dict[str, Any]:
    """Tie canonical full-valid scores to this cell's terminal weights and split."""
    terminal = validate_terminal(root, cell)
    directory = root / cell
    meta = read(directory / "evaluation/manifest.json")
    _, evaluation = members(root)
    require(meta.get("schema") == "sequence-predictions-v1", "Canonical evaluation required")
    require(
        meta.get("complete") is True and meta.get("membership") == "evaluation",
        "Incomplete evaluation",
    )
    require(meta.get("split_sha256") == file_hash(root / "split.json"), "Foreign evaluation split")
    producer = meta["producer"]
    require(
        producer.get("checkpoint_sha256") == terminal["model_sha256"], "Wrong evaluation checkpoint"
    )
    require(producer.get("config") == normalized_config(cell), "Wrong evaluation config")
    require(
        producer.get("provenance", {}).get("identity") == terminal["identity"],
        "Wrong evaluation lineage",
    )
    _validate_entries(meta, evaluation)
    score = read(directory / "score.json")
    manifest_hash = file_hash(directory / "evaluation/manifest.json")
    require(score.get("predictions_manifest_sha256") == manifest_hash, "Score export changed")
    require(score.get("producer") == producer, "Score producer changed")
    require(score.get("split_sha256") == meta["split_sha256"], "Score split changed")
    require(score.get("sequences") == len(evaluation), "Score coverage incomplete")
    return {
        "schema": "wnn.source-kd-evaluation.v1",
        "cell": cell,
        "config_sha256": terminal["identity"]["config_sha256"],
        "checkpoint_sha256": terminal["model_sha256"],
        "source_sha256": terminal["identity"]["source_sha256"],
        "split_sha256": meta["split_sha256"],
        "git_commit": read(root / "packet.json")["git_commit"],
        "prediction_manifest_sha256": manifest_hash,
        "score_sha256": file_hash(directory / "score.json"),
    }
