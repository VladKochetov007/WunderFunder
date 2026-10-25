from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wnn.training.sequence_epoch_index import (
    load_epoch_index,
    publish_epoch_index,
    validate_epoch_index_state,
)
from wnn.training.sequence_epoch_index_io import atomic_replace, canonical_index_bytes


def _publish(
    output: Path,
    epoch: int,
    previous: dict[str, object] | None,
    global_step: int,
) -> dict[str, object]:
    checkpoint_dir = output / "epoch_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    content = f"checkpoint-{epoch}".encode()
    digest = hashlib.sha256(content).hexdigest()
    filename = f"epoch-{epoch + 1:06d}-{digest}.pt"
    (checkpoint_dir / filename).write_bytes(content)
    entry = {
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "global_step": global_step,
        "checkpoint_path": f"epoch_checkpoints/{filename}",
        "checkpoint_sha256": digest,
    }
    return publish_epoch_index(
        output,
        {"config_sha256": "a" * 64},
        epoch,
        epoch + 1,
        global_step,
        entry,
        previous,
    )


def test_epoch_indexes_are_canonical_append_only_prefixes(tmp_path: Path) -> None:
    first = _publish(tmp_path, 0, None, 3)
    second = _publish(tmp_path, 1, first, 7)

    loaded = load_epoch_index(tmp_path / "epoch_index.json", first["identity"])
    immutable = load_epoch_index(
        tmp_path / "epoch_indexes" / "epoch-000002.json", first["identity"]
    )
    assert loaded == second == immutable
    assert second["previous_index_sha256"] == first["index_sha256"]
    assert len(second["entries"]) == 2
    assert (tmp_path / "epoch_index.json").read_bytes() == canonical_index_bytes(second)


def test_resume_reconciliation_accepts_only_one_missing_tail(tmp_path: Path) -> None:
    first = _publish(tmp_path, 0, None, 3)
    history = [
        {"epoch": 0, "complete": True, "updates": 3},
        {"epoch": 1, "complete": True, "updates": 4},
    ]

    recovered_prefix = validate_epoch_index_state(tmp_path, first["identity"], 2, 7, history)
    assert recovered_prefix == first
    assert load_epoch_index(tmp_path / "epoch_index.json") == first


def test_resume_reconciliation_promotes_a_pointer_one_snapshot_behind(tmp_path: Path) -> None:
    first = _publish(tmp_path, 0, None, 3)
    second = _publish(tmp_path, 1, first, 7)
    atomic_replace(tmp_path / "epoch_index.json", canonical_index_bytes(first))

    latest = validate_epoch_index_state(
        tmp_path,
        first["identity"],
        2,
        7,
        [
            {"epoch": 0, "complete": True, "updates": 3},
            {"epoch": 1, "complete": True, "updates": 4},
        ],
    )
    assert latest == second
    assert load_epoch_index(tmp_path / "epoch_index.json") == second


def test_loader_rejects_identity_drift_and_tampered_digest(tmp_path: Path) -> None:
    index = _publish(tmp_path, 0, None, 3)
    pointer = tmp_path / "epoch_index.json"

    with pytest.raises(ValueError, match="identity"):
        load_epoch_index(pointer, {"config_sha256": "b" * 64})

    tampered = dict(index)
    tampered["identity"] = {"config_sha256": "b" * 64}
    atomic_replace(pointer, canonical_index_bytes(tampered))
    with pytest.raises(ValueError, match="digest"):
        load_epoch_index(pointer)


def test_resume_rejects_future_snapshot_and_history_gap(tmp_path: Path) -> None:
    first = _publish(tmp_path, 0, None, 3)
    with pytest.raises(ValueError, match="ahead"):
        validate_epoch_index_state(tmp_path, first["identity"], 0, 0, [])

    second = _publish(tmp_path, 1, first, 7)
    (tmp_path / "epoch_indexes" / "epoch-000001.json").unlink()
    with pytest.raises(ValueError, match="gap"):
        validate_epoch_index_state(
            tmp_path,
            second["identity"],
            2,
            7,
            [
                {"epoch": 0, "complete": True, "updates": 3},
                {"epoch": 1, "complete": True, "updates": 4},
            ],
        )
