"""Stdlib provenance checks shared by the R5 epoch evaluator and selector."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
R4 = PACKET.parent / "high_compute_r4"
if str(R4) not in sys.path:
    sys.path.insert(0, str(R4))

from split_inputs import validate_outer_inner_pair  # noqa: E402

INNER_SCHEMA = "wnn.sequence-inner-split.v1"
INDEX_SCHEMA = "wnn.sequence-epoch-index.v1"
MAX_JSON_BYTES = 32_000_000
EXPECTED_EPOCHS = 30


def canonical_bytes(value: object) -> bytes:
    """Serialize finite canonical JSON without a trailing newline."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON values used for immutable index and receipt identities."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_object(path: Path, maximum_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    """Read one bounded JSON object and retain the exact bytes for hashing."""
    raw = path.read_bytes()
    if not raw or len(raw) > maximum_bytes:
        raise ValueError(f"JSON size outside the accepted bound: {path}")
    value = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, raw


def write_json_new(path: Path, value: object) -> str:
    """Atomically publish immutable finite JSON without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def validate_split_pair(outer_path: Path, inner_path: Path, fold: int) -> dict[str, Any]:
    """Use the canonical R4 validator to bind an inner split to its outer fold."""
    outer_path = outer_path.expanduser().resolve(strict=True)
    inner_path = inner_path.expanduser().resolve(strict=True)
    outer, outer_raw = read_object(outer_path, 2_000_000)
    inner, inner_raw = read_object(inner_path, 2_000_000)
    if (
        outer.get("schema") != "wnn.sequence-campaign-split.v1"
        or inner.get("schema") != INNER_SCHEMA
        or type(fold) is not int
        or outer.get("fold") != fold
        or inner.get("fold") != fold
    ):
        raise ValueError("epoch selection requires the matching inner split and outer parent")
    validate_outer_inner_pair(outer_path, outer, inner_path, inner, fold)
    fit = inner.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("inner split fitting membership is malformed")
    train, valid = fit.get("train"), fit.get("valid")
    if any(
        not isinstance(groups, list)
        or not groups
        or any(type(group) is not int or group < 0 for group in groups)
        or groups != sorted(set(groups))
        for groups in (train, valid)
    ):
        raise ValueError("inner split fit membership must be sorted, unique and nonempty")
    evaluation_groups = inner["evaluation"]["valid"]
    return {
        "outer_path": str(outer_path),
        "outer_sha256": hashlib.sha256(outer_raw).hexdigest(),
        "inner_path": str(inner_path),
        "inner_sha256": hashlib.sha256(inner_raw).hexdigest(),
        "fold": fold,
        "fit_members": [["train", group] for group in train]
        + [["valid", group] for group in valid],
        "evaluation_members": [["valid", group] for group in evaluation_groups],
    }


def safe_file(root: Path, relative: object) -> Path:
    """Resolve an indexed file inside one root without following symlinks."""
    if not isinstance(relative, str):
        raise ValueError("indexed file path must be text")
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("indexed path escapes its root")
    root = root.resolve(strict=True)
    candidate = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("indexed path must not traverse symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("indexed path escapes its root") from error
    if not resolved.is_file():
        raise ValueError("indexed reference is not a regular file")
    return resolved


def _normalized_recipe(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    model = result.get("model")
    if not isinstance(model, dict) or not isinstance(result.get("precision_mode"), str):
        raise ValueError("R5 config is missing model or precision identity")
    result["model"] = dict(model) | {"precision_mode": result["precision_mode"]}
    return result


def validate_epoch_index(
    fit_dir: Path,
    index_path: Path,
    requested_sha256: str,
    config_path: Path,
    split_binding: dict[str, Any],
) -> dict[str, Any]:
    """Require either canonical full-index path and matching latest/pinned bytes."""
    fit_dir = fit_dir.expanduser().resolve(strict=True)
    index_path = index_path.expanduser().resolve(strict=True)
    latest_pointer = safe_file(fit_dir, "epoch_index.json")
    latest_snapshot = safe_file(fit_dir, f"epoch_indexes/epoch-{EXPECTED_EPOCHS:06d}.json")
    if index_path not in (latest_pointer, latest_snapshot):
        raise ValueError("R5 sweep requires the latest or immutable complete 30-epoch index")
    index, index_raw = read_object(index_path)
    if latest_pointer.read_bytes() != index_raw or latest_snapshot.read_bytes() != index_raw:
        raise ValueError("latest and immutable 30-epoch indexes must match the requested bytes")
    if index_raw != canonical_bytes(index) + b"\n":
        raise ValueError("epoch index bytes are not canonical")
    unsigned = {key: value for key, value in index.items() if key != "index_sha256"}
    index_sha = canonical_sha256(unsigned)
    if (
        index.get("schema") != INDEX_SCHEMA
        or index.get("index_sha256") != index_sha
        or requested_sha256 != index_sha
    ):
        raise ValueError("unsupported or detached epoch index")
    if set(index) != {
        "schema",
        "identity",
        "completed_epochs",
        "global_step",
        "previous_index_sha256",
        "entries",
        "index_sha256",
    }:
        raise ValueError("epoch index fields differ from the retained-epoch contract")
    if (
        type(index.get("completed_epochs")) is not int
        or index["completed_epochs"] != EXPECTED_EPOCHS
    ):
        raise ValueError("selection requires all 30 completed epochs")
    identity, entries = index.get("identity"), index.get("entries")
    if (
        not isinstance(identity, dict)
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_EPOCHS
    ):
        raise ValueError("epoch index must contain its full run identity and 30 snapshots")
    if identity.get("schema") != "wnn.sequence-run-identity.v1":
        raise ValueError("epoch index run identity is malformed")
    data = identity.get("data")
    if (
        not isinstance(data, dict)
        or data.get("split_sha256") != split_binding["inner_sha256"]
        or data.get("fold") != split_binding["fold"]
        or data.get("fit_members") != split_binding["fit_members"]
    ):
        raise ValueError("fit index does not bind this component-clean inner split")

    config, _ = read_object(config_path)
    normalized = _normalized_recipe(config)
    if (
        config.get("epochs") != EXPECTED_EPOCHS
        or config.get("expected_fold") != split_binding["fold"]
        or config.get("precision_mode") != "bf16_bf16x9"
        or config.get("checkpoint_retention") != "all_epochs"
        or identity.get("config_sha256") != canonical_sha256(normalized)
    ):
        raise ValueError("R5 config is not a complete BF16/BF16x9 30-epoch retention recipe")
    previous_step = 0
    for position, entry in enumerate(entries):
        expected = (position, position + 1)
        expected_path = (
            (f"epoch_checkpoints/epoch-{expected[1]:06d}-{entry.get('checkpoint_sha256')}.pt")
            if isinstance(entry, dict)
            else None
        )
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {"epoch", "completed_epochs", "global_step", "checkpoint_path", "checkpoint_sha256"}
            or type(entry.get("epoch")) is not int
            or entry.get("epoch") != expected[0]
            or type(entry.get("completed_epochs")) is not int
            or entry.get("completed_epochs") != expected[1]
            or type(entry.get("global_step")) is not int
            or entry["global_step"] <= previous_step
            or not isinstance(entry.get("checkpoint_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["checkpoint_sha256"]) is None
            or entry.get("checkpoint_path") != expected_path
        ):
            raise ValueError("epoch index has a missing or malformed checkpoint entry")
        checkpoint = safe_file(fit_dir, entry.get("checkpoint_path"))
        if file_sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"retained checkpoint hash mismatch at epoch {expected[1]}")
        previous_step = entry["global_step"]
    if entries[-1]["global_step"] != index.get("global_step"):
        raise ValueError("epoch index final optimizer step does not match its frontier")

    previous_hash = None
    for completed in range(1, EXPECTED_EPOCHS + 1):
        snapshot = safe_file(fit_dir, f"epoch_indexes/epoch-{completed:06d}.json")
        prefix, raw = read_object(snapshot)
        prefix_hash = prefix.get("index_sha256")
        if (
            set(prefix) != set(index)
            or raw != canonical_bytes(prefix) + b"\n"
            or prefix.get("completed_epochs") != completed
            or prefix.get("identity") != identity
            or prefix.get("entries") != entries[:completed]
            or prefix.get("previous_index_sha256") != previous_hash
            or prefix_hash
            != canonical_sha256(
                {key: value for key, value in prefix.items() if key != "index_sha256"}
            )
        ):
            raise ValueError(f"immutable epoch index prefix changed: epoch {completed}")
        previous_hash = prefix_hash

    terminal, _ = read_object(safe_file(fit_dir, "terminal_status.json"))
    final_model = safe_file(fit_dir, "model.pt")
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != EXPECTED_EPOCHS
        or terminal.get("identity") != identity
        or terminal.get("model_sha256") != file_sha256(final_model)
    ):
        raise ValueError("R5 fit does not have a matching completed fixed-final endpoint")
    return {
        "index": index,
        "index_sha256": index_sha,
        "index_file_sha256": hashlib.sha256(index_raw).hexdigest(),
        "identity": identity,
        "entries": entries,
        "config": config,
        "config_file_sha256": file_sha256(config_path),
        "config_recipe_sha256": canonical_sha256(normalized),
        "final_model_sha256": terminal["model_sha256"],
        "fit_dir": str(fit_dir),
    }
