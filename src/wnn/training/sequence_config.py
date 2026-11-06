from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from wnn.training.sequence_blend import validate_blend_config
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


def load_config(path: Path) -> dict[str, Any]:
    """Read a bounded precision-explicit recipe; reject silently ignored settings."""
    raw = json.loads(path.read_text())
    optional = {"model", "checkpoint_retention", "blend_objective_alpha"}
    if not isinstance(raw, dict) or set(raw) - (set(DEFAULTS) | optional):
        raise ValueError("unknown sequence recipe field")
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
    model = MODEL_DEFAULTS | raw.get("model", {})
    if set(model) - set(MODEL_DEFAULTS):
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
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if type(model["hint_dim"]) is not int or model["hint_dim"] < 0:
        raise ValueError("hint_dim must be a nonnegative integer")
    if model["architecture"] not in ("transformer", "gru"):
        raise ValueError("architecture must be transformer or gru")
    if model["attention_backend"] not in ("math", "efficient", "flex"):
        raise ValueError("model.attention_backend must be math, efficient, or flex")
    if model["architecture"] == "gru" and model["attention_backend"] != "math":
        raise ValueError("attention backends are supported only by Transformer models")
    if type(model["compile_blocks"]) is not bool:
        raise ValueError("model.compile_blocks must be boolean")
    if model["architecture"] == "gru" and model["compile_blocks"]:
        raise ValueError("model.compile_blocks requires a Transformer model")
    if config["optimizer"] == "muon" and model["architecture"] != "transformer":
        raise ValueError("sequence Muon currently supports Transformer models only")
    if model["hint_projection"] not in ("fixed", "learned"):
        raise ValueError("hint_projection must be fixed or learned")
    if model["dropout"] != 0 or type(model["checkpoint_blocks"]) is not bool:
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
    if required_device not in (None, "B300"):
        raise ValueError("required_device must be null or B300")
    expected_fold = config["expected_fold"]
    if expected_fold is not None and (type(expected_fold) is not int or expected_fold < 0):
        raise ValueError("expected_fold must be null or a nonnegative integer")
    # This runtime-only field lets fit and inference build the same precision-aware model.
    model["precision_mode"] = config["precision_mode"]
    config["model"] = model
    validate_blend_config(config)
    return config


def validate_runtime_requirements(
    config: dict[str, Any], world_size: int, device_name: str
) -> bool:
    """Check the recipe's optional world-size and device requirements on this rank."""
    required_world_size = config["required_world_size"]
    if required_world_size is not None and world_size != required_world_size:
        raise ValueError(f"recipe requires world size {required_world_size}, received {world_size}")
    if config["required_device"] == "B300" and "B300" not in device_name.upper():
        raise ValueError(f"recipe requires an NVIDIA B300 device, received {device_name!r}")
    return True


def validate_expected_fold(config: dict[str, Any], actual_fold: int) -> bool:
    """Check the loaded split's fold against the recipe's optional fold requirement."""
    expected_fold = config["expected_fold"]
    if expected_fold is not None and actual_fold != expected_fold:
        raise ValueError(f"recipe requires fold {expected_fold}, received fold {actual_fold}")
    return True
