"""Strict checkpoint admission for fixed-final Signal Routes predictors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require a JSON object at a checkpoint boundary."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _sha256(value: object, name: str) -> str:
    """Validate one lowercase SHA-256 value."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_preprocessing(
    extension: Mapping[str, Any],
    split_sha256: str,
    fit_members: list[list[str | int]],
    data: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Verify the embedded train-only normalization against saved fit identity."""
    from signal_routes.bindings import RUNTIME_COMMIT, canonical
    from wnn.data import FEATURE_COLUMNS

    artifact = _mapping(extension.get("preprocessing"), "signal_routes.preprocessing")
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    artifact_sha = hashlib.sha256(encoded + b"\n").hexdigest()
    identity = _mapping(artifact.get("fit_identity"), "preprocessing.fit_identity")
    expected_fields = {
        "split_sha256",
        "fit_members_sha256",
        "normalization",
        "feature_columns",
    }
    if (
        set(artifact) != {"schema", "runtime_commit", "fit_identity", "mean", "std"}
        or artifact.get("schema") != "signal-routes.preprocessing.v1"
        or artifact.get("runtime_commit") != RUNTIME_COMMIT
        or set(identity) != expected_fields
        or artifact_sha != _sha256(extension.get("preprocessing_sha256"), "preprocessing_sha256")
    ):
        raise ValueError("embedded preprocessing schema or digest is invalid")
    mean, std = artifact.get("mean"), artifact.get("std")
    for values, label in ((mean, "mean"), (std, "std")):
        if (
            not isinstance(values, list)
            or len(values) != 112
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError(f"preprocessing {label} must contain 112 finite numbers")
    if any(value <= 0 for value in std):
        raise ValueError("preprocessing standard deviations must be positive")
    data_normalization = _mapping(data.get("normalization"), "checkpoint.data.normalization")
    saved_normalization = _mapping(provenance.get("normalization"), "provenance.normalization")
    if (
        identity.get("split_sha256") != split_sha256
        or identity.get("fit_members_sha256") != canonical(fit_members)
        or identity.get("normalization") != data_normalization
        or identity.get("feature_columns") != list(FEATURE_COLUMNS)
        or mean != saved_normalization.get("mean")
        or std != saved_normalization.get("std")
    ):
        raise ValueError("preprocessing ancestry or moments differ from checkpoint provenance")


def load_endpoint(
    checkpoint: Path,
    runtime_root: Path,
    split_sha256: str,
    fit_members: list[list[str | int]],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Strict-load one complete E4 checkpoint after source, split, and recipe admission."""
    import torch

    from signal_routes.bindings import VARIANTS, canonical, load_cell, source_identity
    from signal_routes.modeling import build_model
    from wnn.training.sequence_config import load_config

    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = _mapping(torch.load(checkpoint, map_location="cpu", weights_only=True), "checkpoint")
    if payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("unsupported canonical sequence checkpoint schema")
    config = _mapping(payload.get("config"), "checkpoint.config")
    extension = _mapping(config.get("signal_routes"), "checkpoint.config.signal_routes")
    if set(extension) != {"schema", "cell", "preprocessing", "preprocessing_sha256"}:
        raise ValueError("Signal Routes checkpoint metadata fields are incomplete")
    if extension.get("schema") != "signal-routes.model.v1":
        raise ValueError("unsupported Signal Routes model metadata")
    cell = _mapping(extension.get("cell"), "signal_routes.cell")
    variant, seed = cell.get("variant"), cell.get("seed")
    if (
        variant not in VARIANTS
        or type(seed) is not int
        or cell.get("id") != f"sr_{variant}_s{seed}"
    ):
        raise ValueError("checkpoint cell id, variant, or seed is malformed")
    packet = Path(__file__).resolve().parents[2]
    cell_path = packet / "configs" / f"{cell['id']}.json"
    declared_cell, native_path = load_cell(cell_path)
    if cell != declared_cell or config.get("seed") != cell.get("seed"):
        raise ValueError("checkpoint cell or seed differs from the frozen packet descriptor")
    native_config = load_config(native_path)
    base_config = {key: value for key, value in config.items() if key != "signal_routes"}
    if base_config != native_config:
        raise ValueError("checkpoint normalized config differs from the full native recipe")
    epochs = config.get("epochs")
    if (
        epochs != 4
        or payload.get("completed_epochs") != 4
        or payload.get("epoch") != 3
        or config.get("expected_fold") != 0
    ):
        raise ValueError("only the configured fixed E4 endpoint can be exported")
    provenance = _mapping(payload.get("provenance"), "checkpoint.provenance")
    identity = _mapping(provenance.get("identity"), "checkpoint.provenance.identity")
    current_source = source_identity(runtime_root)
    if (
        identity.get("source_sha256") != current_source["sha256"]
        or identity.get("source_files") != current_source["files"]
    ):
        raise ValueError("checkpoint source identity differs from the admitted source files")
    if identity.get("config_sha256") != canonical(config):
        raise ValueError("checkpoint config differs from its recorded source identity")
    saved_data = _mapping(payload.get("data"), "checkpoint.data")
    if (
        identity.get("data_sha256") != canonical(saved_data)
        or identity.get("data") != saved_data
        or saved_data.get("split_sha256") != split_sha256
        or saved_data.get("fit_members") != fit_members
        or saved_data.get("signal_routes_preprocessing_sha256")
        != extension.get("preprocessing_sha256")
        or len(fit_members) != 10_325
    ):
        raise ValueError("checkpoint fit ancestry differs from requested official data")
    _validate_preprocessing(extension, split_sha256, fit_members, saved_data, provenance)
    state = _mapping(payload.get("model"), "checkpoint.model")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or (
            tensor.is_floating_point()
            and (tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError(f"checkpoint contains an invalid model tensor: {name}")
    model = build_model(config, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
