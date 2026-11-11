"""Source-qualified fixed-final checkpoint admission for Impact81 modules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_preprocessing(
    impact: Mapping[str, Any],
    *,
    card: str,
    split_sha256: str,
    fit_members: list[list[str | int]],
    data: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    from wnn.data import FEATURE_COLUMNS

    from .bindings import canonical
    from .runtime_family import family_for_card, runtime_commit_for_card

    preprocessing = _mapping(impact.get("preprocessing"), "impact81.preprocessing")
    encoded = json.dumps(preprocessing, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded + b"\n").hexdigest()
    family = family_for_card(card)
    expected_schema = (
        "impact81.preprocessing.v1" if family == "transformer" else "impact81.gru-preprocessing.v1"
    )
    if (
        preprocessing.get("schema") != expected_schema
        or preprocessing.get("runtime_commit") != runtime_commit_for_card(card)
        or digest != _sha256(impact.get("preprocessing_sha256"), "preprocessing_sha256")
    ):
        raise ValueError("unsupported or malformed embedded preprocessing artifact")
    if family == "transformer":
        knots = preprocessing.get("knots")
        if (
            not isinstance(knots, list)
            or len(knots) != 112
            or any(not isinstance(row, list) or len(row) != 16 for row in knots)
        ):
            raise ValueError("Transformer preprocessing knot dimensions are invalid")
    else:
        if set(preprocessing) != {
            "schema",
            "runtime_commit",
            "fit_identity",
            "mean",
            "std",
            "clip_event_feasibility",
        }:
            raise ValueError("GRU preprocessing fields differ from the frozen artifact schema")
        for name in ("mean", "std"):
            values = preprocessing.get(name)
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
                raise ValueError(f"GRU preprocessing {name} must contain 112 finite moments")
        if any(value <= 0 for value in preprocessing["std"]):
            raise ValueError("GRU preprocessing scales must be positive")
        from .preparation_later import validate_feasibility

        validate_feasibility(
            _mapping(
                preprocessing.get("clip_event_feasibility"),
                "preprocessing.clip_event_feasibility",
            ),
            require_feasible=card == "10",
        )
    fit_identity = _mapping(preprocessing.get("fit_identity"), "preprocessing.fit_identity")
    if set(fit_identity) != {
        "split_sha256",
        "fit_members_sha256",
        "normalization",
        "feature_columns",
    }:
        raise ValueError("preprocessing fit identity fields differ from the frozen schema")
    expected_members_sha256 = canonical(fit_members)
    if (
        fit_identity.get("split_sha256") != split_sha256
        or fit_identity.get("fit_members_sha256") != expected_members_sha256
        or fit_identity.get("normalization") != data.get("normalization")
        or fit_identity.get("feature_columns") != list(FEATURE_COLUMNS)
    ):
        raise ValueError("preprocessing ancestry differs from this checkpoint's official fit")
    saved_normalization = _mapping(provenance.get("normalization"), "provenance.normalization")
    if preprocessing.get("mean") != saved_normalization.get("mean") or preprocessing.get(
        "std"
    ) != saved_normalization.get("std"):
        raise ValueError("preprocessing moments differ from the checkpoint normalization")


def load_endpoint(
    checkpoint: Path,
    runtime_root: Path,
    split_sha256: str,
    fit_members: list[list[str | int]],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct a fixed-final endpoint only after exact source, split and artifact admission."""
    import torch

    from .bindings import canonical, source_identity
    from .modeling import build_model
    from .runtime_family import epochs_for_card
    from .variants import variant_for

    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = _mapping(payload, "checkpoint")
    if payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("unsupported checkpoint schema")
    config = _mapping(payload.get("config"), "checkpoint.config")
    impact = _mapping(config.get("impact81"), "checkpoint.config.impact81")
    if set(impact) != {"schema", "cell", "preprocessing", "preprocessing_sha256"}:
        raise ValueError("Impact81 checkpoint metadata fields are incomplete or unexpected")
    if impact.get("schema") != "impact81.model.v1":
        raise ValueError("unsupported Impact81 model metadata")
    cell = _mapping(impact.get("cell"), "impact81.cell")
    if cell.get("schema") != "impact81.cell.v1":
        raise ValueError("unsupported Impact81 cell descriptor")
    card, arm, variant, seed = (cell.get(key) for key in ("card", "arm", "variant", "seed"))
    if (
        variant != variant_for(card, arm)
        or seed not in (204101, 204102)
        or cell.get("id") != f"ip{card}_{arm}_s{seed}"
    ):
        raise ValueError("checkpoint cell is outside the declared paired panel")
    endpoint_epochs = epochs_for_card(card)
    if (
        config.get("epochs") != endpoint_epochs
        or payload.get("completed_epochs") != endpoint_epochs
        or payload.get("epoch") != endpoint_epochs - 1
    ):
        raise ValueError("export requires the family-configured fixed-final endpoint")
    provenance = _mapping(payload.get("provenance"), "checkpoint.provenance")
    identity = _mapping(provenance.get("identity"), "provenance.identity")
    current_source = source_identity(runtime_root)
    if (
        identity.get("source_sha256") != current_source["sha256"]
        or identity.get("source_files") != current_source["files"]
    ):
        raise ValueError(
            "checkpoint source identity differs from the current runtime/Impact81 files"
        )
    if identity.get("config_sha256") != canonical(config):
        raise ValueError("checkpoint config differs from its recorded run identity")
    saved_data = _mapping(payload.get("data"), "checkpoint.data")
    if (
        identity.get("data_sha256") != canonical(saved_data)
        or identity.get("data") != saved_data
        or saved_data.get("split_sha256") != split_sha256
        or saved_data.get("fit_members") != fit_members
        or saved_data.get("impact81_preprocessing_sha256") != impact.get("preprocessing_sha256")
    ):
        raise ValueError("checkpoint data ancestry differs from the requested split/membership")
    _validate_preprocessing(
        impact,
        card=card,
        split_sha256=split_sha256,
        fit_members=fit_members,
        data=saved_data,
        provenance=provenance,
    )
    state = _mapping(payload.get("model"), "checkpoint.model")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or (
            tensor.is_floating_point()
            and (tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError(f"invalid checkpoint model tensor: {name}")
    model = build_model(config, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if variant == "deep_supervision":
        from .depth_models import drop_auxiliary_heads

        drop_auxiliary_heads(model)
    elif variant == "future_heads":
        from .future_model import drop_future_heads

        drop_future_heads(model)
    elif variant in ("perturbation_control", "state_perturbation", "nll_transfer", "clip_event"):
        from .gru_models import drop_gru_auxiliary_heads

        drop_gru_auxiliary_heads(model)
    return model, payload
