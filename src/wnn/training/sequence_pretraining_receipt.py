from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.training.sequence_checkpoint import atomic_write_json
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_normalization import moments_sha256
from wnn.training.sequence_pretraining_contract import (
    FUTURE_INNOVATION_KIND,
)
from wnn.training.sequence_provenance import canonical_sha256, file_sha256, json_value

PARENT_RECEIPT_NAME = "future_innovation_parent_receipt.json"
MAX_PARENT_CHECKPOINT_BYTES = 256_000_000
BACKBONE_SOURCE_FILES = (
    "src/wnn/nn.py",
    "src/wnn/training/sequence_data.py",
    "src/wnn/training/sequence_fit_data.py",
    "src/wnn/training/sequence_fit_utils.py",
    "src/wnn/training/sequence_fit_model.py",
    "src/wnn/training/sequence_loop.py",
    "src/wnn/training/sequence_model_helpers.py",
    "src/wnn/training/sequence_models.py",
    "src/wnn/training/sequence_normalization.py",
    "src/wnn/training/sequence_objectives.py",
    "src/wnn/training/sequence_precision.py",
    "src/wnn/training/sequence_pretraining.py",
    "src/wnn/training/sequence_pretraining_contract.py",
    "src/wnn/training/sequence_pretraining_receipt.py",
    "src/wnn/training/sequence_update.py",
)


def write_pretraining_parent_receipt(
    output: Path,
    identity: dict[str, object],
    config: dict[str, object],
    data: dict[str, object],
    normalization: dict[str, list[float]],
) -> bool:
    """Publish a parent receipt only after the pretraining model and terminal status exist."""
    if config.get("pretraining") is None or data.get("future_input_innovations") is None:
        raise ValueError("parent receipt requires completed future-input pretraining provenance")
    checkpoint_path = output / "model.pt"
    status_path = output / "terminal_status.json"
    checkpoint = load_parent_checkpoint(checkpoint_path)
    if status_path.is_symlink() or not status_path.is_file():
        raise ValueError("pretraining terminal status must be a regular, non-symlink file")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    checkpoint_provenance = checkpoint.get("provenance")
    if (
        not isinstance(status, dict)
        or status.get("status") != "completed"
        or checkpoint.get("completed_epochs") != 2
        or checkpoint.get("completed_epochs") != config.get("epochs")
        or checkpoint.get("config") != config
        or checkpoint.get("data") != json_value(data)
        or status.get("identity") != identity
        or status.get("model_sha256") != file_sha256(checkpoint_path)
        or not isinstance(checkpoint_provenance, dict)
        or checkpoint_provenance.get("identity") != identity
    ):
        raise ValueError("pretraining terminal checkpoint does not match its run identity")
    fit_members = data.get("fit_members")
    innovation = data["future_input_innovations"]
    source_files = identity.get("source_files")
    if not isinstance(fit_members, list) or not train_only_members(fit_members):
        raise ValueError("parent receipt requires train-only pretraining membership")
    if not isinstance(innovation, dict) or innovation.get("kind") != FUTURE_INNOVATION_KIND:
        raise ValueError("pretraining checkpoint is missing its future-innovation contract")
    if not isinstance(source_files, dict):
        raise ValueError("pretraining identity is missing its source file inventory")
    receipt = {
        "schema": "wnn.sequence-future-innovation-parent.v1",
        "status": "complete",
        "kind": FUTURE_INNOVATION_KIND,
        "checkpoint_file": "model.pt",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "terminal_status_sha256": file_sha256(status_path),
        "pretraining_identity_sha256": canonical_sha256(identity),
        "backbone_source_sha256": backbone_source_sha256(source_files),
        "seed": config["seed"],
        "fit_members": fit_members,
        "fit_members_sha256": membership_sha256(
            [(str(member[0]), int(member[1])) for member in fit_members]
        ),
        "split_sha256": data.get("split_sha256"),
        "input_normalization_sha256": normalization_sha256(normalization),
        "innovation_normalization_identity_sha256": innovation["normalization_identity_sha256"],
        "innovation_moments_sha256": innovation["moments_sha256"],
        "backbone": {"architecture": "gru", "hidden": 256, "layers": 2},
        "warmstart_policy": {
            "copied": ["mean", "std", "inp.*", "gru.*"],
            "excluded": ["head.*", "hint_projector.*"],
            "optimizer": "reset",
            "scheduler": "reset",
            "rng": "fresh child seed and counters",
        },
    }
    path = output / PARENT_RECEIPT_NAME
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise FileExistsError("an incompatible pretraining parent receipt already exists")
        return True
    atomic_write_json(receipt, path)
    return True


def load_parent_checkpoint(path: Path) -> dict[str, Any]:
    """Load one bounded regular sequence checkpoint for receipt creation or warmstart."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("pretraining model checkpoint must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_PARENT_CHECKPOINT_BYTES:
        raise ValueError("pretraining checkpoint exceeds its size bound")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("pretraining checkpoint has an unsupported schema")
    return payload


def train_only_members(members: object) -> bool:
    """Accept only a nonempty ordered list of train source groups."""
    return (
        isinstance(members, list)
        and bool(members)
        and all(
            isinstance(member, list)
            and len(member) == 2
            and member[0] == "train"
            and type(member[1]) is int
            and member[1] >= 0
            for member in members
        )
    )


def normalization_sha256(value: object) -> str:
    """Hash exact FP32 input moments in the same representation as sequence normalization."""
    if not isinstance(value, dict) or set(value) != {"mean", "std"}:
        raise ValueError("input normalization must contain exactly mean and std arrays")
    return moments_sha256(
        np.asarray(value["mean"], dtype=np.float32),
        np.asarray(value["std"], dtype=np.float32),
    )


def backbone_source_sha256(source_files: dict[str, object]) -> str:
    """Hash exact files defining pretext inputs, GRU weights, losses, and updates."""
    selected = {path: source_files[path] for path in BACKBONE_SOURCE_FILES if path in source_files}
    if set(selected) != set(BACKBONE_SOURCE_FILES) or any(
        not isinstance(value, str) or len(value) != 64 for value in selected.values()
    ):
        raise ValueError("source identity is missing a future-innovation backbone file")
    return canonical_sha256(selected)
