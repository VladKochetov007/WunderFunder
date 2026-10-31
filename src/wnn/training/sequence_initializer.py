"""Source-bound fresh-optimizer initialization from a completed sequence fit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from wnn.training.sequence_cache_ensemble_provenance import (
    canonical_sha256,
    common_provenance,
    validate_precision_contract,
)
from wnn.training.sequence_provenance import file_sha256

_MEMBERSHIP_FIELDS = (
    "split_sha256",
    "primary_split_sha256",
    "fit_members",
    "evaluation_members",
    "evaluation_role",
    "evaluation_source",
    "evaluation_mask",
    "training_mask",
    "mask",
    "input_schema",
    "normalization",
    "sources",
)


def load_fitting_initializer(
    path: Path,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Load parent weights without optimizer state and bind exact auxiliary ancestry."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("initializer must be a regular checkpoint file")
    checkpoint_sha256 = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("initializer must be a canonical fixed-final sequence checkpoint")
    parent_config = payload.get("config")
    parent_data = payload.get("data")
    provenance = payload.get("provenance")
    state = payload.get("model")
    if not all(
        isinstance(value, dict) for value in (parent_config, parent_data, provenance, state)
    ):
        raise ValueError("initializer checkpoint is missing config, data, provenance, or model")
    epochs = parent_config.get("epochs")
    if (
        type(epochs) is not int
        or epochs < 1
        or payload.get("completed_epochs") != epochs
        or payload.get("epoch") != epochs - 1
    ):
        raise ValueError("initializer must be the parent's configured fixed-final endpoint")
    if parent_config.get("precision_mode") != "fp32" or config.get("precision_mode") != "fp32":
        raise ValueError("fresh initialization supports strict-FP32 parent and child recipes only")
    if parent_config.get("model") != config.get("model"):
        raise ValueError("initializer and child model architectures/configurations must match")
    parent_model = parent_config.get("model")
    if (
        not isinstance(parent_model, Mapping)
        or type(parent_config.get("kd_weight")) not in (int, float)
        or parent_config["kd_weight"] != 0
        or type(parent_config.get("hint_weight")) not in (int, float)
        or parent_config["hint_weight"] != 0
        or parent_model.get("hint_dim") != 0
    ):
        raise ValueError("initializer parent must be strict-FP32 label-only with hint_dim=0")
    if parent_data.get("teacher") is not None or parent_data.get("initializer") is not None:
        raise ValueError(
            "initializer parent must be direct and have no teacher or initializer ancestry"
        )

    identity = provenance.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("initializer parent provenance is missing its run identity")
    source_files = identity.get("source_files")
    if (
        identity.get("config_sha256") != canonical_sha256(parent_config)
        or identity.get("data_sha256") != canonical_sha256(parent_data)
        or identity.get("data") != parent_data
        or not isinstance(source_files, dict)
        or identity.get("source_sha256") != canonical_sha256(source_files)
    ):
        raise ValueError("initializer parent run identity does not bind its checkpoint metadata")
    runtime = identity.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("autocast") is not False
        or runtime.get("amp_scaler") is not False
    ):
        raise ValueError("initializer parent runtime was not strict FP32")
    if (
        not isinstance(provenance.get("precision"), Mapping)
        or not isinstance(runtime.get("precision"), Mapping)
        or not isinstance(runtime.get("compiler"), Mapping)
    ):
        raise ValueError("initializer parent must use current precision and compiler identities")
    try:
        validate_precision_contract(parent_config, provenance, runtime)
    except ValueError as error:
        raise ValueError("initializer parent runtime was not strict FP32") from error
    parent_producer: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": parent_data.get("split_sha256"),
        "fit_members": parent_data.get("fit_members"),
        "teacher": parent_data.get("teacher"),
        "config": parent_config,
        "provenance": provenance,
    }
    common_provenance(parent_producer, provenance)
    if not isinstance(provenance.get("normalization"), Mapping):
        raise ValueError("initializer parent is missing fitted normalization values")
    if provenance["normalization"] != dict(normalization):
        raise ValueError("initializer normalization differs from this exact fitting membership")
    for field in _MEMBERSHIP_FIELDS:
        if parent_data.get(field) != data_provenance.get(field):
            raise ValueError(f"initializer auxiliary ancestry differs in {field}")
    if (
        parent_data.get("evaluation_role") != "control"
        or parent_data.get("evaluation_source") != "train"
    ):
        raise ValueError(
            "initializer parent must have been fitted and evaluated on auxiliary train"
        )

    if any(
        not isinstance(name, str)
        or not isinstance(value, torch.Tensor)
        or (
            value.is_floating_point()
            and (value.dtype != torch.float32 or not torch.isfinite(value).all())
        )
        for name, value in state.items()
    ):
        raise ValueError("initializer model state must contain finite strict-FP32 tensors")
    parent_mean, parent_std = state.get("mean"), state.get("std")
    if not isinstance(parent_mean, torch.Tensor) or not isinstance(parent_std, torch.Tensor):
        raise ValueError("initializer model must contain fitting-only mean and std buffers")
    child_mean = torch.as_tensor(normalization.get("mean"), dtype=torch.float32)
    child_std = torch.as_tensor(normalization.get("std"), dtype=torch.float32)
    if not torch.equal(parent_mean, child_mean) or not torch.equal(parent_std, child_std):
        raise ValueError("initializer model normalization buffers differ from child fit moments")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError("initializer model tensor shapes do not match the child model") from error
    if file_sha256(path) != checkpoint_sha256:
        raise ValueError("initializer checkpoint changed while it was being loaded")

    fit_members = parent_data.get("fit_members")
    evaluation_members = parent_data.get("evaluation_members")
    return {
        "schema": "wnn.sequence-initializer.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "parent_identity_sha256": canonical_sha256(identity),
        "parent_config_sha256": canonical_sha256(parent_config),
        "parent_split_sha256": parent_data["split_sha256"],
        "primary_split_sha256": parent_data["primary_split_sha256"],
        "fit_members_sha256": canonical_sha256(fit_members),
        "evaluation_members_sha256": canonical_sha256(evaluation_members),
        "evaluation_role": parent_data["evaluation_role"],
        "evaluation_source": parent_data["evaluation_source"],
        "training_mask": parent_data["training_mask"],
        "mask": parent_data["mask"],
        "parent_producer_sha256": canonical_sha256(parent_producer),
        "parent_producer": parent_producer,
    }
