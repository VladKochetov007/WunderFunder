"""Admit PR81 fit and score artifacts into a hash-bound evidence record."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from midas_evidence_binding import (
    SHA,
    bind,
    cell_identity,
    pins,
    run_identity,
    training_rows,
    validate_epoch_index,
    validate_source_map,
)
from midas_evidence_io import _canonical, _hash_file, _read, _read_bytes

METRIC_LIMIT = 64 << 20
EXPECTED_SCORE_ROWS = 3_528_273
EXPECTED_SCORE_SEQUENCES = 1_873


@dataclass(frozen=True)
class PreparedFit:
    """Validated producer metadata needed to assemble a portable bundle."""

    fit: Path
    meta: dict[str, Any]
    meta_raw: bytes
    config: dict[str, Any]
    expected: dict[str, Any]
    owner_raw: bytes
    cell: dict[str, Any]
    fold: int
    commit: str
    split_raw: bytes
    split_digest: str
    epochs: int
    model_sha: str | None
    model_bytes: int | None
    terminal_raw: bytes | None
    status: str
    index_raw: bytes | None
    train_raw: bytes
    train_rows: list[tuple[int, float, bytes]]
    score_obj: dict[str, Any] | None
    score_raw: bytes | None
    external_id: str


def prepare_fit(fit: Path, score: Path | None) -> PreparedFit:
    """Validate source, split, terminal, metrics, score and fixed endpoint claims."""
    fit = Path(fit).expanduser()
    if fit.is_symlink() or not fit.is_dir():
        raise ValueError("fit must be an existing non-symlink directory")
    fit = fit.resolve(strict=True)
    meta, meta_raw = _read(fit / "run_metadata.json")
    config, provenance = meta.get("config"), meta.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("run metadata lacks normalized config/provenance")
    expected = run_identity(meta)
    owner, owner_raw = _read(fit / "owner.json")
    data = provenance["data"]
    fold, split_sha = data.get("fold"), data.get("split_sha256")
    if type(fold) is not int or not isinstance(split_sha, str) or SHA.fullmatch(split_sha) is None:
        raise ValueError("run metadata fold/split identity is invalid")
    cell = cell_identity(config, fold)
    if (
        owner.get("schema") != "impact81.owner.v1"
        or owner.get("cell_id") != cell["id"]
        or owner.get("mode") != "fit"
        or owner.get("optimization_changes") is not False
    ):
        raise ValueError("owner receipt does not identify this frozen PR81 fit")
    commit, pinned_split, bindings_sha = pins(Path(__file__).resolve().parent)
    validate_source_map(
        Path(__file__).resolve().parent,
        expected["source_files"],
        commit,
        bindings_sha,
    )
    split, split_raw = _read(fit / "split.json")
    split_digest = hashlib.sha256(split_raw).hexdigest()
    if split_digest != split_sha or split_digest != pinned_split or split.get("fold") != fold:
        raise ValueError("fit/split.json differs from the run or PR81 pin")
    epochs = config.get("epochs")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("config.epochs must be positive")

    model_path = fit / "model.pt"
    terminal_path = fit / "terminal_status.json"
    if model_path.is_symlink() or terminal_path.is_symlink():
        raise ValueError("fit model/terminal artifacts must not be symlinks")
    model_sha, model_bytes = (
        _hash_file(model_path, 16 << 30) if model_path.exists() else (None, None)
    )
    terminal_raw = None
    status = "partial"
    if terminal_path.exists():
        terminal, terminal_raw = _read(terminal_path)
        terminal_state = terminal.get("status")
        if terminal_state == "completed":
            if (
                model_sha is None
                or terminal.get("model_sha256") != model_sha
                or terminal.get("completed_epochs") != epochs
                or type(terminal.get("global_updates")) is not int
                or terminal["global_updates"] < 1
            ):
                raise ValueError("terminal status does not bind the fixed model endpoint")
            bind(terminal.get("identity"), expected, "terminal status")
            status = "completed"
        elif terminal_state in {"failed", "cancelled"}:
            status = terminal_state

    if score is not None and status != "completed":
        raise ValueError("outer score export requires a verified completed terminal receipt")
    index_path = fit / "epoch_index.json"
    if index_path.is_symlink():
        raise ValueError("fit/epoch_index.json must not be a symlink")
    index_raw = None
    index = None
    if index_path.exists():
        index, index_raw = _read(index_path)
        validate_epoch_index(index, expected, epochs, status == "completed")
    if config.get("checkpoint_retention") == "all_epochs" and index is None:
        raise ValueError("all-epoch retention requires fit/epoch_index.json")

    train_path = fit / "metrics.jsonl"
    if train_path.is_symlink():
        raise ValueError("fit/metrics.jsonl must not be a symlink")
    train_raw = _read_bytes(train_path, METRIC_LIMIT) if train_path.exists() else b""
    frontier = index["completed_epochs"] if index else (epochs if status == "completed" else None)
    train_rows = training_rows(train_raw, frontier)
    if status == "completed" and {row[0] for row in train_rows} != set(range(1, epochs + 1)):
        raise ValueError("completed fit lacks one aggregate training metric per epoch")

    score_obj = None
    score_raw = None
    if score is not None:
        score_obj, score_raw = _read(Path(score).expanduser())
        producer, reported = score_obj.get("producer"), score_obj.get("score")
        if not isinstance(producer, dict) or not isinstance(reported, dict):
            raise ValueError("score JSON lacks producer/score objects")
        if (
            reported.get("metric_id") != "global_wp_clip2_v1"
            or type(reported.get("selected_rows")) is not int
            or reported.get("selected_rows") != EXPECTED_SCORE_ROWS
            or type(score_obj.get("sequences")) is not int
            or score_obj.get("sequences") != EXPECTED_SCORE_SEQUENCES
            or producer.get("checkpoint_sha256") != model_sha
            or producer.get("config") != config
            or producer.get("split_sha256") != split_digest
            or score_obj.get("split_sha256") != split_digest
            or not isinstance(score_obj.get("predictions_manifest_sha256"), str)
            or SHA.fullmatch(score_obj["predictions_manifest_sha256"]) is None
            or "epoch_admission" in producer
            or producer.get("kind") == "retained_epoch"
        ):
            raise ValueError("score does not bind the fixed official-valid endpoint/config/split")
        score_provenance = producer.get("provenance")
        if not isinstance(score_provenance, dict):
            raise ValueError("score producer lacks provenance identity")
        bind(score_provenance.get("identity"), expected, "score producer")
        for key in ("wp", "t0", "t1"):
            value = reported.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"official score {key} must be finite numeric")

    fingerprint = hashlib.sha256(
        _canonical(
            {
                "metadata": hashlib.sha256(meta_raw).hexdigest(),
                "model": model_sha,
                "owner": hashlib.sha256(owner_raw).hexdigest(),
                "score": hashlib.sha256(score_raw).hexdigest() if score_raw else None,
                "metrics": hashlib.sha256(train_raw).hexdigest() if train_raw else None,
            }
        )
    ).hexdigest()
    external_id = f"{cell['id']}:{fingerprint}"
    return PreparedFit(
        fit=fit,
        meta=meta,
        meta_raw=meta_raw,
        config=config,
        expected=expected,
        owner_raw=owner_raw,
        cell=cell,
        fold=fold,
        commit=commit,
        split_raw=split_raw,
        split_digest=split_digest,
        epochs=epochs,
        model_sha=model_sha,
        model_bytes=model_bytes,
        terminal_raw=terminal_raw,
        status=status,
        index_raw=index_raw,
        train_raw=train_raw,
        train_rows=train_rows,
        score_obj=score_obj,
        score_raw=score_raw,
        external_id=external_id,
    )
