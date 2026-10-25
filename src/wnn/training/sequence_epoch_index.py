from __future__ import annotations

import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from wnn.training.sequence_epoch_index_io import (
    MAX_EPOCH_INDEX_BYTES,
    atomic_replace,
    canonical_index_bytes,
    epoch_output_root,
    file_sha256,
    index_sha256,
    read_index_json,
    valid_counter,
    write_immutable,
)

_INDEX_NAME = re.compile(r"epoch-(\d{6,})\.json\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_RETAINED_EPOCHS = 10_000


def load_epoch_index(
    path: Path, expected_identity: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Read and validate one canonical immutable index or latest-index pointer."""
    output = epoch_output_root(path)
    value, encoded = read_index_json(path)
    if canonical_index_bytes(value) != encoded:
        raise ValueError("epoch index bytes are not canonical")
    _validate_index(value, output, expected_identity)
    _validate_index_location(path, value)
    return value


def publish_epoch_index(
    output: Path,
    identity: Mapping[str, object],
    epoch: int,
    completed_epochs: int,
    global_step: int,
    checkpoint_entry: Mapping[str, object],
    previous_index: Mapping[str, object] | None,
) -> dict[str, object]:
    """Publish one append-only prefix snapshot, then advance the atomic pointer."""
    if epoch != completed_epochs - 1 or completed_epochs < 1:
        raise ValueError("epoch index counters must describe one completed safe epoch")
    pointer_path = output / "epoch_index.json"
    if pointer_path.is_symlink() or (pointer_path.exists() and not pointer_path.is_file()):
        raise ValueError("latest epoch index pointer must be a regular file")
    if previous_index is None:
        if completed_epochs != 1:
            raise ValueError("the first epoch index must complete exactly one epoch")
        previous_sha: str | None = None
        entries: list[dict[str, object]] = []
    else:
        _validate_index(dict(previous_index), output, identity)
        if previous_index["completed_epochs"] != completed_epochs - 1:
            raise ValueError("epoch index publication must extend the immediately prior prefix")
        previous_sha = str(previous_index["index_sha256"])
        entries = [dict(item) for item in previous_index["entries"]]
    entry = _validate_entry(dict(checkpoint_entry), output, epoch)
    if entry["completed_epochs"] != completed_epochs or entry["global_step"] != global_step:
        raise ValueError("checkpoint entry counters do not match the new index frontier")
    entries.append(entry)
    document: dict[str, object] = {
        "schema": "wnn.sequence-epoch-index.v1",
        "identity": dict(identity),
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "previous_index_sha256": previous_sha,
        "entries": entries,
    }
    document["index_sha256"] = index_sha256(document)
    _validate_index(document, output, identity)
    encoded = canonical_index_bytes(document)
    if len(encoded) > MAX_EPOCH_INDEX_BYTES:
        raise ValueError("epoch index exceeds the bounded loader size")
    snapshot_dir = output / "epoch_indexes"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise ValueError("epoch index snapshot path must be a real directory")
    snapshot = snapshot_dir / f"epoch-{completed_epochs:06d}.json"
    write_immutable(snapshot, encoded)
    atomic_replace(output / "epoch_index.json", encoded)
    return document


def validate_epoch_index_state(
    output: Path,
    identity: Mapping[str, object],
    safe_completed_epochs: int,
    safe_global_step: int,
    history: list[dict[str, object]],
) -> dict[str, object] | None:
    """Reconcile immutable indexes and the latest pointer with resumable safe state."""
    if not valid_counter(safe_completed_epochs) or not valid_counter(safe_global_step):
        raise ValueError("safe epoch and global-step counters must be nonnegative")
    if len(history) != safe_completed_epochs:
        raise ValueError("resume history does not match the safe epoch frontier")
    cumulative_steps: list[int] = []
    total_steps = 0
    for epoch, row in enumerate(history):
        if (
            not valid_counter(row.get("epoch"))
            or row.get("epoch") != epoch
            or row.get("complete") is not True
            or not valid_counter(row.get("updates"))
        ):
            raise ValueError("resume history is malformed for epoch-index reconciliation")
        total_steps += int(row["updates"])
        cumulative_steps.append(total_steps)
    if total_steps != safe_global_step:
        raise ValueError("resume global step does not match its epoch history")

    snapshot_dir = output / "epoch_indexes"
    if snapshot_dir.is_symlink() or (snapshot_dir.exists() and not snapshot_dir.is_dir()):
        raise ValueError("epoch index snapshot path must be a real directory")
    snapshots: dict[int, Path] = {}
    if snapshot_dir.exists():
        for candidate in snapshot_dir.glob("epoch-*.json"):
            match = _INDEX_NAME.fullmatch(candidate.name)
            if match is None:
                raise ValueError("epoch index snapshot has an unexpected filename")
            number = int(match.group(1))
            if number < 1:
                raise ValueError("epoch index snapshot counter must start at one")
            if number > safe_completed_epochs:
                raise ValueError("epoch index snapshot is ahead of the resumable safe frontier")
            if number in snapshots:
                raise ValueError("duplicate epoch index snapshot")
            snapshots[number] = candidate
    if any(number not in snapshots for number in range(1, safe_completed_epochs)):
        raise ValueError("epoch index history has a gap before the safe frontier")

    previous: dict[str, object] | None = None
    latest_snapshot: dict[str, object] | None = None
    latest_count = (
        safe_completed_epochs
        if safe_completed_epochs in snapshots
        else max(0, safe_completed_epochs - 1)
    )
    for number in range(1, latest_count + 1):
        current = load_epoch_index(snapshots[number], identity)
        _validate_history_prefix(current, number, cumulative_steps)
        expected_previous = previous["index_sha256"] if previous is not None else None
        if current.get("previous_index_sha256") != expected_previous:
            raise ValueError("epoch index immutable snapshot chain is broken")
        if previous is not None and current.get("entries", [])[:-1] != previous.get("entries"):
            raise ValueError("epoch index snapshot is not an append-only prefix")
        previous = current
        latest_snapshot = current

    pointer_path = output / "epoch_index.json"
    if pointer_path.is_symlink():
        raise ValueError("latest epoch index pointer cannot be a symlink")
    pointer: dict[str, object] | None = None
    if pointer_path.exists():
        pointer = load_epoch_index(pointer_path, identity)
        pointer_count = int(pointer["completed_epochs"])
        if pointer_count > safe_completed_epochs:
            raise ValueError("latest epoch index pointer is ahead of the safe frontier")
        if pointer_count < max(0, latest_count - 1):
            raise ValueError("latest epoch index pointer is more than one snapshot behind")
        if pointer_count > 0:
            if pointer_count not in snapshots:
                raise ValueError("latest epoch index pointer has no immutable snapshot")
            matching = load_epoch_index(snapshots[pointer_count], identity)
            if pointer != matching:
                raise ValueError("latest epoch index pointer differs from its immutable snapshot")
    elif latest_count > 1 or (latest_count == 1 and safe_completed_epochs > 1):
        raise ValueError("latest epoch index pointer is missing behind multiple snapshots")

    if latest_snapshot is not None:
        latest_entry = latest_snapshot["entries"][-1]
        checkpoint_path = output / str(latest_entry["checkpoint_path"])
        if file_sha256(checkpoint_path) != latest_entry["checkpoint_sha256"]:
            raise ValueError("latest inference checkpoint hash does not match its epoch index")
        latest_count_in_pointer = int(pointer["completed_epochs"]) if pointer is not None else 0
        if latest_count_in_pointer != latest_count:
            atomic_replace(pointer_path, canonical_index_bytes(latest_snapshot))
    return latest_snapshot


def _validate_index(
    value: dict[str, object], output: Path, expected_identity: Mapping[str, object] | None
) -> None:
    required = {
        "schema",
        "identity",
        "completed_epochs",
        "global_step",
        "previous_index_sha256",
        "entries",
        "index_sha256",
    }
    if set(value) != required or value.get("schema") != "wnn.sequence-epoch-index.v1":
        raise ValueError("unsupported or malformed epoch index schema")
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("epoch index is missing its run identity")
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError("epoch index config/data/source/runtime identity does not match")
    completed, global_step = value.get("completed_epochs"), value.get("global_step")
    if (
        not valid_counter(completed)
        or completed < 1
        or completed > MAX_RETAINED_EPOCHS
        or not valid_counter(global_step)
    ):
        raise ValueError("epoch index has invalid safe-frontier counters")
    previous = value.get("previous_index_sha256")
    if (completed == 1 and previous is not None) or (
        completed > 1 and (not isinstance(previous, str) or _DIGEST.fullmatch(previous) is None)
    ):
        raise ValueError("epoch index has an invalid parent digest")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != completed:
        raise ValueError("epoch index entries do not cover the completed prefix")
    previous_step = 0
    for epoch, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError("epoch index contains a malformed checkpoint entry")
        current_entry = _validate_entry(raw_entry, output, epoch)
        if int(current_entry["global_step"]) <= previous_step:
            raise ValueError("epoch index global steps must increase at each safe boundary")
        previous_step = int(current_entry["global_step"])
    last = entries[-1]
    if last["global_step"] != global_step:
        raise ValueError("epoch index global step differs from its last checkpoint")
    supplied_sha = value.get("index_sha256")
    if not isinstance(supplied_sha, str) or _DIGEST.fullmatch(supplied_sha) is None:
        raise ValueError("epoch index is missing a valid self digest")
    if index_sha256(value) != supplied_sha:
        raise ValueError("epoch index self digest does not match its contents")


def _validate_entry(value: dict[str, object], output: Path, epoch: int) -> dict[str, object]:
    if set(value) != {
        "epoch",
        "completed_epochs",
        "global_step",
        "checkpoint_path",
        "checkpoint_sha256",
    }:
        raise ValueError("epoch index checkpoint entry has unexpected fields")
    if (
        not valid_counter(value.get("epoch"))
        or not valid_counter(value.get("completed_epochs"))
        or value.get("epoch") != epoch
        or value.get("completed_epochs") != epoch + 1
    ):
        raise ValueError("epoch index entries are not a contiguous completed prefix")
    if not valid_counter(value.get("global_step")) or value["global_step"] == 0:
        raise ValueError("epoch checkpoint entry has invalid global step")
    digest = value.get("checkpoint_sha256")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("epoch checkpoint entry has an invalid checkpoint digest")
    expected_path = f"epoch_checkpoints/epoch-{epoch + 1:06d}-{digest}.pt"
    if value.get("checkpoint_path") != expected_path:
        raise ValueError("epoch checkpoint path is not content-addressed")
    relative = PurePosixPath(expected_path)
    checkpoint_dir = output / "epoch_checkpoints"
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("epoch checkpoint directory must be a real directory")
    path = output.joinpath(*relative.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("epoch index references a missing inference checkpoint") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("epoch checkpoint reference must be a regular non-symlink file")
    return value


def _validate_history_prefix(
    value: Mapping[str, object], count: int, cumulative_steps: list[int]
) -> None:
    if value.get("completed_epochs") != count:
        raise ValueError("epoch index snapshot filename and frontier disagree")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("epoch index is missing its checkpoint prefix")
    for number, entry in enumerate(entries, start=1):
        if entry.get("global_step") != cumulative_steps[number - 1]:
            raise ValueError("epoch index global steps differ from safe resume history")


def _validate_index_location(path: Path, value: Mapping[str, object]) -> None:
    completed = int(value["completed_epochs"])
    if path.name == "epoch_index.json" and path.parent.name != "epoch_indexes":
        return
    match = _INDEX_NAME.fullmatch(path.name)
    if path.parent.name != "epoch_indexes" or match is None or int(match.group(1)) != completed:
        raise ValueError("epoch index file location does not match its completed frontier")
