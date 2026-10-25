"""Receiver-only checks for explicit epoch admission and immutable weight identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wnn.training.sequence_epoch_prediction import digest, epoch_admission, validate_index


def fixture_index(root: Path) -> tuple[Path, Path, dict[str, object]]:
    """Build a metadata fixture without loading or executing any model."""
    raw = b"synthetic-weight-bytes-for-hash-validation"
    sha = hashlib.sha256(raw).hexdigest()
    weight = root / "epoch_checkpoints" / f"epoch-000001-{sha}.pt"
    weight.parent.mkdir()
    weight.write_bytes(raw)
    entry = {
        "epoch": 0,
        "completed_epochs": 1,
        "global_step": 10,
        "checkpoint_path": str(weight.relative_to(root)),
        "checkpoint_sha256": sha,
    }
    index = {
        "schema": "wnn.sequence-epoch-index.v1",
        "identity": {},
        "entries": [entry],
        "completed_epochs": 1,
        "global_step": 10,
        "previous_index_sha256": None,
    }
    index["index_sha256"] = digest(index)
    path = root / "epoch_index.json"
    path.write_text(json.dumps(index))
    return weight, path, index


def test_modified_weight_cannot_be_admitted(tmp_path: Path) -> None:
    """A valid metadata receipt cannot conceal replaced weight bytes."""
    weight, path, index = fixture_index(tmp_path)
    weight.write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="bytes changed"):
        epoch_admission(weight, path, index["index_sha256"])


def test_undeclared_future_snapshot_cannot_be_admitted(tmp_path: Path) -> None:
    """An orphan snapshot outside the committed prefix cannot enter evaluation."""
    _, path, index = fixture_index(tmp_path)
    orphan = tmp_path / "epoch_checkpoints" / "epoch-000002-orphan.pt"
    orphan.write_bytes(b"not-safe-yet")
    with pytest.raises(ValueError, match="pinned safe"):
        epoch_admission(orphan, path, index["index_sha256"])


def test_index_tampering_requires_a_new_explicit_pin(tmp_path: Path) -> None:
    """Changing a frontier cannot retain its former admission hash."""
    _, _, index = fixture_index(tmp_path)
    index["completed_epochs"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_index(index, index["index_sha256"])


def test_weight_symlink_is_rejected(tmp_path: Path) -> None:
    """A convenient link does not bypass the explicit checkpoint-path contract."""
    weight, path, index = fixture_index(tmp_path)
    link = tmp_path / "linked.pt"
    link.symlink_to(weight)
    with pytest.raises(ValueError, match="escapes"):
        epoch_admission(link, path, index["index_sha256"])
