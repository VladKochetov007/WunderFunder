from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from wnn.training.sequence_compact_config import (
    COMPACT_ARCHITECTURES,
    compact_model_fields,
    validate_compact_recipe,
)
from wnn.training.sequence_config_validation import (
    is_galam_lstm64_muon,
    validate_input_prefetch_batches,
    validate_monitoring_config,
    validate_sequence_objective_config,
)
from wnn.training.sequence_config_validation import (
    validate_expected_fold as validate_expected_fold,
)
from wnn.training.sequence_config_validation import (
    validate_runtime_requirements as validate_runtime_requirements,
)
from wnn.training.sequence_precision import PRECISION_MODES, validate_process_environment

# JSON is heterogeneous at this boundary; validated fields are consumed by the runtime.
DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "epochs": 3,
    "batch_size": 1,
    "accumulate_chunks": 4,
    "chunk_rows": 256,
    "seed": 1001,
    "lr": 0.0001,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_ns_eps": 1e-7,
    "adamw_beta1": 0.9,
    "adamw_beta2": 0.999,
    "adamw_eps": 1e-8,
    "grad_clip": 1.0,
    "warmup_fraction": 0.05,
    "schedule": "legacy_warmup_cosine",
    "min_lr_ratio": 0.0,
    "kd_weight": 0.0,
    "hint_weight": 0.0,
    "hint_mode": "aligned",
    "normalization_sequences": 32,
    "training_mask": "is_scored",
    "normalization_scope": "prefix",
    "required_world_size": None,
    "required_device": None,
    "expected_fold": None,
    "precision_mode": "fp32",
}
MODEL_DEFAULTS: dict[str, Any] = {
    "architecture": "transformer",
    "hidden": 768,
    "layers": 24,
    "heads": 12,
    "ff_multiplier": 4,
    "context_rows": 512,
    "dropout": 0.0,
    "checkpoint_blocks": True,
    "hint_dim": 0,
    "hint_projection": "learned",
    "attention_backend": "math",
    "compile_blocks": False,
}


def _validate_input_feature_mode(model: dict[str, Any]) -> str:
    """Keep expanded inputs limited to the existing causal GRU and Transformer adapters."""
    mode = model.get("input_feature_mode", "raw112")
    if mode not in ("raw112", "event211"):
        raise ValueError("model.input_feature_mode must be raw112 or event211")
    if mode == "event211":
        architecture = model["architecture"]
        if architecture == "gru" and (model["hidden"], model["layers"]) != (256, 2):
            raise ValueError("event211 is scoped to the standalone GRU256x2 launch")
        if architecture not in ("gru", "transformer"):
            raise ValueError("event211 inputs are supported only by GRU and Transformer adapters")
    return str(mode)


def load_config(path: Path) -> dict[str, Any]:
    """Read a bounded precision-explicit recipe; reject silently ignored settings."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) - (
        set(DEFAULTS)
        | {
            "model",
            "checkpoint_retention",
            "dropout_policy",
            "monitoring",
            "input_prefetch_batches",
            "objective_mode",
        }
    ):
        raise ValueError("unknown sequence recipe field")
    if "input_prefetch_batches" in raw:
        validate_input_prefetch_batches(raw["input_prefetch_batches"])
    if "monitoring" in raw:
        validate_monitoring_config(
            raw["monitoring"],
            raw.get("model", {}),
            raw.get("epochs", DEFAULTS["epochs"]),
            raw.get("checkpoint_retention", "final_only"),
        )
    if "dropout_policy" in raw and raw["dropout_policy"] != "explicit_regularization_v1":
        raise ValueError("unsupported explicit dropout policy")
    # Omission preserves historical config hashes and fixed-final profile identities.
    if "checkpoint_retention" in raw and raw["checkpoint_retention"] not in (
        "final_only",
        "all_epochs",
    ):
        raise ValueError("checkpoint_retention must be final_only or all_epochs")
    config = DEFAULTS | raw
    # Preserve legacy AdamW's .999 while defaulting the Muon-side AdamW groups to .95.
    if config["optimizer"] == "muon" and "adamw_beta2" not in raw:
        config["adamw_beta2"] = 0.95
    raw_model = raw.get("model", {})
    if not isinstance(raw_model, dict):
        raise ValueError("model must be a mapping")
    allowed_model_fields = (
        set(MODEL_DEFAULTS) | {"input_feature_mode"} | compact_model_fields(raw_model)
    )
    if raw_model.get("architecture") == "galam":
        from wnn.training.galam_adapter import galam_model_settings

        galam_input_mode = raw_model.get("input_feature_mode", "raw112")
        model = galam_model_settings(
            {name: value for name, value in raw_model.items() if name != "input_feature_mode"}
        )
        if "input_feature_mode" in raw_model:
            model["input_feature_mode"] = galam_input_mode
        allowed_model_fields.update(("galam", "precision_mode"))
        galam_muon = is_galam_lstm64_muon(model)
        if config["precision_mode"] != "fp32" or (
            config["optimizer"] != "adamw" and not galam_muon
        ):
            raise ValueError(
                "unprofiled GALAM requires fp32 and AdamW except for the LSTM64 hybrid"
            )
        if type(config["batch_size"]) is not int or config["batch_size"] < 2:
            raise ValueError("GALAM per-time BatchNorm requires batch_size >= 2 per rank")
    else:
        model = MODEL_DEFAULTS | raw_model
    if set(model) - allowed_model_fields:
        raise ValueError("unknown sequence model field")
    config["model"] = model
    for key in (
        "schema_version",
        "epochs",
        "batch_size",
        "accumulate_chunks",
        "chunk_rows",
        "normalization_sequences",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["schema_version"] != 1 or config["chunk_rows"] > 20_000:
        raise ValueError("unsupported schema or chunk length")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    for key in (
        "lr",
        "weight_decay",
        "grad_clip",
        "warmup_fraction",
        "kd_weight",
        "hint_weight",
        "muon_ns_eps",
        "adamw_eps",
    ):
        if not isinstance(config[key], (float, int)) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if config["lr"] == 0 or config["grad_clip"] == 0 or config["warmup_fraction"] >= 1:
        raise ValueError("positive learning rate/clipping and warmup_fraction < 1 required")
    if type(config["muon_ns_steps"]) is not int or config["muon_ns_steps"] <= 0:
        raise ValueError("muon_ns_steps must be a positive integer")
    for key in ("muon_ns_eps", "adamw_eps"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0:
            raise ValueError(f"{key} must be a positive number")
    for key in ("muon_momentum", "adamw_beta1", "adamw_beta2"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key} must be finite and in [0, 1)")
    if config["optimizer"] not in ("adamw", "muon"):
        raise ValueError("optimizer must be adamw or muon")
    if config["precision_mode"] not in PRECISION_MODES:
        raise ValueError("precision_mode must be fp32, tf32, bf16, fp32_bf16x9, or bf16_bf16x9")
    validate_process_environment(config["precision_mode"])
    if config["schedule"] not in ("legacy_warmup_cosine", "warmup_cosine_v2"):
        raise ValueError("unsupported sequence learning-rate schedule")
    ratio = config["min_lr_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (float, int)):
        raise ValueError("min_lr_ratio must be numeric")
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")
    if config["schedule"] == "legacy_warmup_cosine" and ratio != 0:
        raise ValueError("legacy_warmup_cosine requires min_lr_ratio=0")
    for key in ("hidden", "layers", "heads", "ff_multiplier", "context_rows"):
        if key == "ff_multiplier" and model["architecture"] == "galam":
            continue  # Explicit positive integer feed_forward_width is checked in GALAMConfig.
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if type(model["hint_dim"]) is not int or model["hint_dim"] < 0:
        raise ValueError("hint_dim must be a nonnegative integer")
    if model["architecture"] not in ("transformer", "gru", "lstm", "galam", *COMPACT_ARCHITECTURES):
        raise ValueError("unsupported sequence architecture")
    input_feature_mode = _validate_input_feature_mode(model)
    if model["attention_backend"] not in ("math", "efficient", "flex"):
        raise ValueError("model.attention_backend must be math, efficient, or flex")
    if model["architecture"] in ("gru", "lstm") and model["attention_backend"] != "math":
        raise ValueError("attention backends are supported only by Transformer models")
    if type(model["compile_blocks"]) is not bool:
        raise ValueError("model.compile_blocks must be boolean")
    if model["architecture"] in ("gru", "lstm") and model["compile_blocks"]:
        raise ValueError("model.compile_blocks requires a Transformer model")
    if config["optimizer"] == "muon":
        supported_muon = (
            model["architecture"] == "transformer"
            or (
                model["architecture"] == "gru"
                and input_feature_mode == "event211"
                and (model["hidden"], model["layers"]) == (256, 2)
            )
            or is_galam_lstm64_muon(model)
        )
        if not supported_muon:
            raise ValueError(
                "sequence Muon supports Transformer models only by default; "
                "approved GRU/GALAM adapters are explicit exceptions"
            )
    if model["hint_projection"] not in ("fixed", "learned"):
        raise ValueError("hint_projection must be fixed or learned")
    dropout = model["dropout"]
    explicit_dropout = raw.get("dropout_policy") == "explicit_regularization_v1"
    if explicit_dropout and (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise ValueError("explicit dropout must be finite numeric in [0, 1)")
    if (not explicit_dropout and dropout != 0) or type(model["checkpoint_blocks"]) is not bool:
        raise ValueError("this paired runtime freezes dropout=0 and boolean checkpoint_blocks")
    if model["architecture"] == "transformer" and model["hidden"] % (2 * model["heads"]):
        raise ValueError("RoPE requires an even integral head dimension")
    if config["hint_weight"] > 0 and (
        model["hint_dim"] == 0 or model["hint_projection"] != "learned"
    ):
        raise ValueError("hint distillation requires a learned student projection")
    if config["hint_mode"] not in ("aligned", "component_permuted"):
        raise ValueError("unsupported hint control")
    if config["hint_mode"] != "aligned" and config["hint_weight"] <= 0:
        raise ValueError("permuted hints require a positive hint loss")
    if config["training_mask"] not in ("is_scored", "all_after_warmup"):
        raise ValueError("training_mask must be is_scored or all_after_warmup")
    if config["normalization_scope"] not in ("prefix", "all_fit"):
        raise ValueError("normalization_scope must be prefix or all_fit")
    required_world_size = config["required_world_size"]
    if required_world_size is not None and (
        type(required_world_size) is not int or required_world_size < 1
    ):
        raise ValueError("required_world_size must be null or a positive integer")
    required_device = config["required_device"]
    if required_device not in (None, "B300", "L4"):
        raise ValueError("required_device must be null, B300, or L4")
    expected_fold = config["expected_fold"]
    if expected_fold is not None and (type(expected_fold) is not int or expected_fold < 0):
        raise ValueError("expected_fold must be null or a nonnegative integer")
    # This runtime-only field lets fit and inference build the same precision-aware model.
    model["precision_mode"] = config["precision_mode"]
    config["model"] = model
    if "objective_mode" in raw:
        validate_sequence_objective_config(config)
    validate_compact_recipe(config)
    return config
