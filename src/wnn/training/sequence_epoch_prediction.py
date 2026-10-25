from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Inference payloads contain heterogeneous JSON provenance, never executable objects here.
EPOCH_KIND = "retained_epoch"
INDEX_SCHEMA = "wnn.sequence-epoch-index.v1"


def digest(value: object) -> str:
    """Hash bounded JSON metadata using the trainer's canonical encoding."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_index(index: dict[str, Any], expected_hash: str) -> None:
    """Validate a captured prefix of safe epochs without scanning any weight files."""
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise ValueError("unsupported epoch index")
    body = {key: value for key, value in index.items() if key != "index_sha256"}
    if index.get("index_sha256") != expected_hash or digest(body) != expected_hash:
        raise ValueError("epoch index hash mismatch")
    count = index.get("completed_epochs")
    entries = index.get("entries")
    if type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("invalid epoch index frontier")
    if not isinstance(entries, list) or len(entries) != count:
        raise ValueError("epoch index must contain its complete safe prefix")
    previous_step = 0
    paths = set()
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or entry.get("epoch") != number - 1:
            raise ValueError("epoch index order mismatch")
        step = entry.get("global_step")
        if (
            entry.get("completed_epochs") != number
            or type(step) is not int
            or step <= previous_step
        ):
            raise ValueError("invalid safe epoch boundary")
        name, sha = entry.get("checkpoint_path"), entry.get("checkpoint_sha256")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("invalid epoch checkpoint digest")
        if name != f"epoch_checkpoints/epoch-{number:06d}-{sha}.pt" or name in paths:
            raise ValueError("noncanonical or duplicate epoch checkpoint path")
        paths.add(name)
        previous_step = step
    if index.get("global_step") != previous_step or not isinstance(index.get("identity"), dict):
        raise ValueError("epoch index terminal boundary mismatch")


def epoch_admission(checkpoint: Path, index_path: Path, expected_hash: str) -> dict[str, Any]:
    """Bind one local weight file to an explicitly pinned immutable index snapshot."""
    if index_path.is_symlink() or index_path.stat().st_size > 16_000_000:
        raise ValueError("unsafe or oversized epoch index")
    index = json.loads(index_path.read_text())
    validate_index(index, expected_hash)
    if index_path.name == "epoch_index.json":
        root = index_path.parent.resolve()
    elif index_path.parent.name == "epoch_indexes":
        root = index_path.parent.parent.resolve()
    else:
        raise ValueError("epoch index must be inside its original fit directory")
    resolved = checkpoint.resolve(strict=True)
    if checkpoint.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("epoch checkpoint escapes the fit directory")
    relative = resolved.relative_to(root).as_posix()
    matches = [entry for entry in index["entries"] if entry["checkpoint_path"] == relative]
    if len(matches) != 1:
        raise ValueError("checkpoint is not in the pinned safe epoch prefix")
    entry = matches[0]
    actual = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            actual.update(block)
    if actual.hexdigest() != entry["checkpoint_sha256"]:
        raise ValueError("epoch checkpoint bytes changed")
    return {"index": index, "index_sha256": expected_hash, "entry": entry}


def validate_epoch_payload(payload: dict[str, Any], admission: dict[str, Any]) -> None:
    """Keep the original training schedule and bind inference weights to their safe epoch."""
    entry, identity = admission["entry"], admission["index"]["identity"]
    config, provenance = payload.get("config"), payload.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("epoch checkpoint lacks config/provenance")
    if payload.get("checkpoint_schema") != "wnn.sequence-epoch.v1":
        raise ValueError("unsupported epoch checkpoint schema")
    if any(payload.get(key) != entry[key] for key in ("epoch", "completed_epochs", "global_step")):
        raise ValueError("epoch checkpoint boundary does not match index")
    if (
        config.get("checkpoint_retention") != "all_epochs"
        or type(config.get("epochs")) is not int
        or not 1 <= payload["completed_epochs"] <= config["epochs"]
    ):
        raise ValueError("epoch checkpoint rewrote the declared schedule")
    if (
        digest(config) != identity.get("config_sha256")
        or provenance.get("identity") != identity
        or payload.get("data") != identity.get("data")
    ):
        raise ValueError("epoch checkpoint identity differs from its index")


def validate_epoch_producer(
    producer: dict[str, Any], split_hash: str, members: list[tuple[str, int]]
) -> None:
    """Admit retained-epoch metrics only; this does not authorize a KD teacher cache."""
    admission = producer.get("epoch_admission")
    if producer.get("kind") != EPOCH_KIND or not isinstance(admission, dict):
        raise ValueError("missing epoch prediction admission")
    index = admission.get("index")
    validate_index(index, admission.get("index_sha256"))
    entry = admission.get("entry")
    if not isinstance(entry, dict) or entry not in index["entries"]:
        raise ValueError("prediction epoch is outside the pinned safe index")
    identity = index["identity"]
    provenance, config = producer.get("provenance"), producer.get("config")
    data = identity.get("data")
    if (
        not isinstance(provenance, dict)
        or not isinstance(data, dict)
        or not isinstance(config, dict)
    ):
        raise ValueError("epoch prediction identity is incomplete")
    if (
        producer.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        or producer.get("split_sha256") != split_hash
        or data.get("split_sha256") != split_hash
        or producer.get("fit_members") != [list(member) for member in members]
        or data.get("fit_members") != producer["fit_members"]
        or digest(data) != identity.get("data_sha256")
        or digest(config) != identity.get("config_sha256")
        or digest(identity.get("source_files")) != identity.get("source_sha256")
        or provenance.get("identity") != identity
    ):
        raise ValueError("epoch prediction source/config/split/weights identity mismatch")
    if (
        producer.get("teacher") is not None
        or data.get("teacher") is not None
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
    ):
        raise ValueError("R5 epoch selection requires a labels-only teacher with no ancestors")
    if provenance.get("completed_epochs") != entry["completed_epochs"]:
        raise ValueError("epoch provenance does not attest the indexed boundary")
