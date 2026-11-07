"""Strict options for the bounded fresh compact architecture screen."""

from __future__ import annotations

from collections.abc import Mapping

COMPACT_DEFAULTS: dict[str, dict[str, int]] = {
    "gru_local": {"local_hidden": 128, "local_layers": 2},
    "mamba2": {
        "ssm_state_size": 16,
        "ssm_head_dim": 32,
        "ssm_expand": 2,
        "ssm_conv_width": 4,
        "ssm_chunk_size": 256,
    },
    "ema_tabm_style": {
        "ema_projection": 32,
        "ensemble_members": 4,
        "numerical_embedding_dim": 2,
    },
}
COMPACT_ARCHITECTURES = tuple(COMPACT_DEFAULTS)


def compact_model_fields(model: Mapping[str, object]) -> set[str]:
    """Return only the additional settings belonging to this declared adapter."""
    return set(COMPACT_DEFAULTS.get(str(model.get("architecture")), {}))


def compact_model_settings(config: Mapping[str, object]) -> dict[str, object]:
    """Validate compact dimensions without importing optional CUDA dependencies."""
    architecture = str(config["architecture"])
    if architecture not in COMPACT_ARCHITECTURES:
        raise ValueError("unsupported compact sequence architecture")
    model: dict[str, object] = COMPACT_DEFAULTS[architecture] | dict(config)
    for key in ("hidden", "layers", "heads", "context_rows", "ff_multiplier"):
        if type(model[key]) is not int or int(model[key]) < 1:
            raise ValueError(f"compact model {key} must be a positive integer")
    for key in COMPACT_DEFAULTS[architecture]:
        if type(model[key]) is not int or int(model[key]) < 1:
            raise ValueError(f"compact model {key} must be a positive integer")
    if (
        model.get("precision_mode", "fp32") != "fp32"
        or model.get("input_feature_mode", "raw112") != "raw112"
        or model.get("attention_backend", "math") != "math"
        or model.get("compile_blocks", False) is not False
        or model.get("dropout", 0.0) != 0.0
        or model.get("hint_dim", 0) != 0
    ):
        raise ValueError("compact screen requires raw112/fp32, math, no compile/dropout/hints")
    if type(model.get("checkpoint_blocks", False)) is not bool:
        raise ValueError("compact checkpoint_blocks must be boolean")
    if architecture == "gru_local":
        if int(model["local_hidden"]) % (2 * int(model["heads"])):
            raise ValueError("local attention requires even integral RoPE head widths")
    else:
        if model.get("checkpoint_blocks", False):
            raise ValueError("Mamba-2/EMA adapters do not implement activation checkpointing")
        if any(model[key] != 1 for key in ("heads", "context_rows", "ff_multiplier")):
            raise ValueError(
                "non-attention compact adapters require heads/context/ff placeholders=1"
            )
    if architecture == "mamba2":
        if int(model["hidden"]) * int(model["ssm_expand"]) % int(model["ssm_head_dim"]):
            raise ValueError("Mamba-2 hidden*expand must divide into complete SSD heads")
        size = int(model["ssm_chunk_size"])
        if size & (size - 1):
            raise ValueError("Mamba-2 scan chunk size must be a power of two")
    return model


def validate_compact_recipe(config: dict[str, object]) -> None:
    """Reject implicit objective/precision/optimizer changes in the compact cohort."""
    model = config["model"]
    if not isinstance(model, dict) or model.get("architecture") not in COMPACT_ARCHITECTURES:
        return
    if (
        config["precision_mode"] != "fp32"
        or config["optimizer"] != "adamw"
        or config["kd_weight"] != 0
        or config["hint_weight"] != 0
    ):
        raise ValueError("compact screen requires fresh fp32 label-only AdamW fits")
    config["model"] = compact_model_settings(model)
