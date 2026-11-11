from __future__ import annotations

import json
from pathlib import Path

import torch

from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_pretraining_contract import (
    FUTURE_INNOVATION_HORIZONS,
    FUTURE_INNOVATION_KIND,
)
from wnn.training.sequence_pretraining_receipt import (
    backbone_source_sha256,
    load_parent_checkpoint,
    normalization_sha256,
    train_only_members,
)
from wnn.training.sequence_provenance import canonical_sha256, file_sha256

_BACKBONE_PREFIXES = ("inp.", "gru.")
_BACKBONE_BUFFERS = {"mean", "std"}


def initialize_from_parent_receipt(
    model: torch.nn.Module,
    receipt_path: Path,
    config: dict[str, object],
    data: dict[str, object],
    normalization: dict[str, list[float]],
    current_source_files: dict[str, str],
) -> dict[str, object]:
    """Copy only an exact source/data/normalization-matched GRU backbone from a pretext parent."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("initializer receipt must be a regular, non-symlink file")
    receipt_path = receipt_path.resolve()
    receipt_bytes = receipt_path.read_bytes()
    if not receipt_bytes or len(receipt_bytes) > 1_000_000:
        raise ValueError("initializer receipt exceeds its size bound")
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict) or receipt.get("schema") != (
        "wnn.sequence-future-innovation-parent.v1"
    ):
        raise ValueError("unsupported future-innovation parent receipt")
    policy = {
        "copied": ["mean", "std", "inp.*", "gru.*"],
        "excluded": ["head.*", "hint_projector.*"],
        "optimizer": "reset",
        "scheduler": "reset",
        "rng": "fresh child seed and counters",
    }
    if receipt.get("status") != "complete" or receipt.get("kind") != FUTURE_INNOVATION_KIND:
        raise ValueError("future-innovation parent receipt is incomplete")
    if receipt.get("backbone") != {"architecture": "gru", "hidden": 256, "layers": 2}:
        raise ValueError("parent receipt does not declare the approved GRU256x2 backbone")
    if receipt.get("warmstart_policy") != policy:
        raise ValueError("parent receipt does not declare model-only warmstart and fresh optimizer")
    if receipt.get("checkpoint_file") != "model.pt":
        raise ValueError("parent checkpoint path must name the adjacent model.pt artifact")
    checkpoint_path = receipt_path.parent / "model.pt"
    if checkpoint_path.is_symlink() or file_sha256(checkpoint_path) != receipt.get(
        "checkpoint_sha256"
    ):
        raise ValueError("pretraining parent checkpoint SHA-256 mismatch")
    status_path = receipt_path.parent / "terminal_status.json"
    if status_path.is_symlink() or not status_path.is_file():
        raise ValueError("pretraining terminal status must be a regular, non-symlink file")
    if file_sha256(status_path) != receipt.get("terminal_status_sha256"):
        raise ValueError("pretraining terminal-status SHA-256 mismatch")
    parent = load_parent_checkpoint(checkpoint_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    provenance = parent.get("provenance")
    identity = provenance.get("identity") if isinstance(provenance, dict) else None
    if (
        not isinstance(status, dict)
        or status.get("status") != "completed"
        or status.get("model_sha256") != receipt.get("checkpoint_sha256")
        or status.get("identity") != identity
        or parent.get("completed_epochs") != 2
    ):
        raise ValueError("pretraining terminal status does not prove a complete parent checkpoint")
    _validate_parent_identity(parent, receipt, config, data, normalization, current_source_files)
    parent_state = parent.get("model")
    if not isinstance(parent_state, dict):
        raise ValueError("pretraining parent is missing model tensors")
    target_state = model.state_dict()
    expected_keys = {key for key in target_state if _is_backbone_key(key)}
    parent_keys = {key for key in parent_state if _is_backbone_key(key)}
    if expected_keys != parent_keys:
        raise ValueError("pretraining parent GRU backbone keys differ from the child model")
    for key in expected_keys:
        source_tensor, target_tensor = parent_state[key], target_state[key]
        if (
            not isinstance(source_tensor, torch.Tensor)
            or source_tensor.dtype != torch.float32
            or source_tensor.shape != target_tensor.shape
            or not bool(torch.isfinite(source_tensor).all())
        ):
            raise ValueError(f"pretraining parent has an incompatible backbone tensor: {key}")
        target_state[key] = source_tensor
    model.load_state_dict(target_state, strict=True)
    return {
        "kind": FUTURE_INNOVATION_KIND,
        "receipt_sha256": file_sha256(receipt_path),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "parent_identity_sha256": receipt["pretraining_identity_sha256"],
        "fit_members_sha256": receipt["fit_members_sha256"],
        "input_normalization_sha256": receipt["input_normalization_sha256"],
        "optimizer_state": "reset",
        "scheduler_state": "reset",
        "rng_state": "fresh child seed and counters",
        "copied_backbone_keys": sorted(expected_keys),
    }


def _validate_parent_identity(
    parent: dict[str, object],
    receipt: dict[str, object],
    child_config: dict[str, object],
    child_data: dict[str, object],
    child_normalization: dict[str, list[float]],
    current_source_files: dict[str, str],
) -> None:
    """Check parent recipe, full fitting membership, split, source, and normalizer identity."""
    provenance = parent.get("provenance")
    parent_data, parent_config = parent.get("data"), parent.get("config")
    if not isinstance(provenance, dict) or not isinstance(parent_data, dict):
        raise ValueError("pretraining parent is missing run provenance")
    identity = provenance.get("identity")
    innovation = parent_data.get("future_input_innovations")
    if (
        not isinstance(identity, dict)
        or canonical_sha256(identity) != receipt.get("pretraining_identity_sha256")
        or parent_data.get("split_sha256") != child_data.get("split_sha256")
        or parent_data.get("fit_members") != child_data.get("fit_members")
        or not isinstance(innovation, dict)
        or innovation.get("kind") != FUTURE_INNOVATION_KIND
        or innovation.get("moments_sha256") != receipt.get("innovation_moments_sha256")
        or innovation.get("normalization_identity_sha256")
        != receipt.get("innovation_normalization_identity_sha256")
        or not isinstance(parent_config, dict)
        or parent_config.get("pretraining")
        != {"kind": FUTURE_INNOVATION_KIND, "horizons": list(FUTURE_INNOVATION_HORIZONS)}
        or parent_config.get("seed") != child_config.get("seed")
        or receipt.get("seed") != child_config.get("seed")
        or receipt.get("split_sha256") != child_data.get("split_sha256")
        or receipt.get("fit_members") != parent_data.get("fit_members")
        or child_config.get("parent_initializer") != FUTURE_INNOVATION_KIND
    ):
        raise ValueError("pretraining parent does not match this child source and fit membership")
    parent_source_files = identity.get("source_files")
    if (
        not isinstance(parent_source_files, dict)
        or backbone_source_sha256(parent_source_files) != receipt.get("backbone_source_sha256")
        or backbone_source_sha256(current_source_files) != receipt.get("backbone_source_sha256")
    ):
        raise ValueError("pretraining and child backbone/training source files differ")
    if not train_only_members(parent_data.get("fit_members", [])):
        raise ValueError("pretraining parent includes non-train fitting members")
    if not _matching_fit_sources(parent_data, child_data):
        raise ValueError("pretraining and child fitting source fingerprints differ")
    child_members = child_data.get("fit_members")
    if not isinstance(child_members, list) or receipt.get(
        "fit_members_sha256"
    ) != membership_sha256([(str(member[0]), int(member[1])) for member in child_members]):
        raise ValueError("parent receipt fitting membership hash does not match the child")
    if normalization_sha256(provenance.get("normalization", {})) != receipt.get(
        "input_normalization_sha256"
    ) or normalization_sha256(child_normalization) != receipt.get("input_normalization_sha256"):
        raise ValueError("pretraining and supervised child input normalizers differ")
    _validate_parent_model(parent_config, child_config)


def _validate_parent_model(
    parent_config: dict[str, object], child_config: dict[str, object]
) -> None:
    """Require identical GRU widths and depth while excluding task-specific output heads."""
    parent_model, child_model = parent_config.get("model"), child_config.get("model")
    if not isinstance(parent_model, dict) or not isinstance(child_model, dict):
        raise ValueError("parent/child model config is missing")
    if (
        parent_model.get("architecture") != "gru"
        or child_model.get("architecture") != "gru"
        or parent_model.get("hidden") != 256
        or child_model.get("hidden") != 256
        or parent_model.get("layers") != 2
        or child_model.get("layers") != 2
        or parent_model.get("dropout") != child_model.get("dropout")
        or parent_model.get("hint_projection") != "learned"
        or child_model.get("hint_projection") != "learned"
        or parent_model.get("hint_dim") != 336
        or child_model.get("hint_dim") != 0
    ):
        raise ValueError("pretraining parent must use the 336-output future-innovation head")


def _is_backbone_key(name: str) -> bool:
    """Include only input normalization/projection and recurrent weights."""
    return name in _BACKBONE_BUFFERS or name.startswith(_BACKBONE_PREFIXES)


def _matching_fit_sources(parent_data: dict[str, object], child_data: dict[str, object]) -> bool:
    """Compare only train parquet fingerprints named by exact source-qualified fit membership."""
    parent_sources, child_sources = parent_data.get("sources"), child_data.get("sources")
    members = parent_data.get("fit_members")
    if not isinstance(parent_sources, dict) or not isinstance(child_sources, dict):
        return False
    if not isinstance(members, list):
        return False
    paths = {f"datasets/{member[0]}.parquet" for member in members if isinstance(member, list)}
    return bool(paths) and all(
        path in parent_sources
        and path in child_sources
        and parent_sources[path] == child_sources[path]
        for path in paths
    )
