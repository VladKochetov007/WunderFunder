"""Frozen R4 stage choices and complete sequence-runtime config validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from campaign_io import PACKET_ROOT, R3_ROOT, REPO_ROOT, read_object, require

RECIPE_PATH = PACKET_ROOT / "recipe.json"
STUDENT_TEMPLATE_PATH = PACKET_ROOT / "templates/student_gru512.json"
R3_BASE_PATHS = {
    0: R3_ROOT / "large_teacher_r3_muon30/configs/t1792_f0.json",
    2: R3_ROOT / "large_teacher_r3_muon30/configs/t1792_f2.json",
}
TOP_LEVEL_CONFIG_KEYS = {
    "schema_version",
    "epochs",
    "batch_size",
    "accumulate_chunks",
    "chunk_rows",
    "seed",
    "lr",
    "weight_decay",
    "optimizer",
    "muon_momentum",
    "muon_ns_steps",
    "muon_ns_eps",
    "adamw_beta1",
    "adamw_beta2",
    "adamw_eps",
    "grad_clip",
    "warmup_fraction",
    "schedule",
    "min_lr_ratio",
    "kd_weight",
    "hint_weight",
    "hint_mode",
    "normalization_sequences",
    "training_mask",
    "normalization_scope",
    "required_world_size",
    "required_device",
    "expected_fold",
    "precision_mode",
    "model",
}
MODEL_CONFIG_KEYS = {
    "architecture",
    "hidden",
    "layers",
    "heads",
    "ff_multiplier",
    "context_rows",
    "dropout",
    "checkpoint_blocks",
    "hint_dim",
    "hint_projection",
    "attention_backend",
    "compile_blocks",
}

R4_PRECISION_MODE = "bf16_bf16x9"

# Training configs are heterogeneous JSON maps; Any is confined to this boundary.


def recipe() -> dict[str, Any]:
    """Load the static campaign recipe used by the materializer."""
    value = read_object(RECIPE_PATH)
    require(value.get("schema") == "wnn.high-compute-r4-recipe.v1", "unsupported R4 recipe")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Require a complete, strict sequence runtime config with no ignored fields."""
    require(
        set(config) in (TOP_LEVEL_CONFIG_KEYS, TOP_LEVEL_CONFIG_KEYS | {"checkpoint_retention"}),
        "sequence config is incomplete or has unknown fields",
    )
    require(
        config.get("checkpoint_retention", "final_only") in ("final_only", "all_epochs"),
        "unsupported checkpoint retention policy",
    )
    model = config.get("model")
    require(
        isinstance(model, dict) and set(model) == MODEL_CONFIG_KEYS, "model config is incomplete"
    )
    require(config["schema_version"] == 1, "unsupported sequence config version")
    for name in (
        "epochs",
        "batch_size",
        "accumulate_chunks",
        "chunk_rows",
        "normalization_sequences",
    ):
        require(
            type(config[name]) is int and config[name] > 0, f"invalid sequence config field {name}"
        )
    require(type(config["seed"]) is int and 0 <= config["seed"] < 2**32, "invalid config seed")
    for name in ("lr", "weight_decay", "grad_clip", "warmup_fraction", "kd_weight", "hint_weight"):
        value = config[name]
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value),
            f"invalid numeric field {name}",
        )
        require(value >= 0, f"negative config field {name}")
    require(
        config["lr"] > 0 and config["grad_clip"] > 0 and config["warmup_fraction"] < 1,
        "invalid learning schedule",
    )
    require(config["optimizer"] in ("muon", "adamw"), "unsupported optimizer")
    require(
        config["schedule"] == "warmup_cosine_v2" and 0 <= config["min_lr_ratio"] <= 1,
        "unsupported schedule",
    )
    require(
        config["training_mask"] == "all_after_warmup"
        and config["normalization_scope"] == "all_fit",
        "unsupported fit/normalization scope",
    )
    require(
        config["required_world_size"] in (1, 8) and config["required_device"] == "B300",
        "unsupported execution target",
    )
    require(config["expected_fold"] in (0, 2), "invalid expected fold")
    require(
        config["precision_mode"] == R4_PRECISION_MODE,
        "R4 requires the explicit BF16_BF16x9 precision mode",
    )
    require(model["architecture"] in ("transformer", "gru"), "unsupported architecture")
    for name in ("hidden", "layers", "heads", "ff_multiplier", "context_rows"):
        require(type(model[name]) is int and model[name] > 0, f"invalid model field {name}")
    require(
        model["hint_dim"] == 0 and config["hint_weight"] == 0,
        "R4 freezes hidden-hint distillation off",
    )
    require(
        model["dropout"] == 0 and type(model["checkpoint_blocks"]) is bool,
        "unsupported model execution policy",
    )
    if model["architecture"] == "transformer":
        require(
            model["attention_backend"] == "efficient"
            and model["compile_blocks"] is True
            and model["checkpoint_blocks"] is False,
            "R4 teachers require efficient attention, compiled blocks, and no checkpointing",
        )
    else:
        require(
            model["attention_backend"] == "math" and model["compile_blocks"] is False,
            "R4 GRU students require the math attention default and no block compilation",
        )
    require(
        config["optimizer"] != "muon" or model["architecture"] == "transformer",
        "Muon is teacher-only",
    )
    require(
        model["architecture"] != "transformer" or model["hidden"] % (2 * model["heads"]) == 0,
        "Transformer hidden width must divide into even heads",
    )
    require(config["kd_weight"] in (0, 1), "R4 freezes KD to labels-only or unit output KD")


def teacher_base(fold: int, seed: int) -> dict[str, Any]:
    """Copy the R3 teacher recipe and add the frozen R4 precision/runtime policy."""
    require(fold in R3_BASE_PATHS, "R4 supports outer folds F0 and F2")
    config = read_object(R3_BASE_PATHS[fold])
    config["expected_fold"] = fold
    config["seed"] = seed
    config["precision_mode"] = R4_PRECISION_MODE
    config["model"] = config["model"].copy()
    config["model"].update(
        {"attention_backend": "efficient", "compile_blocks": True, "checkpoint_blocks": False}
    )
    validate_config(config)
    return config


def student_base(fold: int, seed: int, kd_weight: float) -> dict[str, Any]:
    """Build the frozen raw112 GRU512x2 student recipe from its complete template."""
    config = read_object(STUDENT_TEMPLATE_PATH)
    config["expected_fold"] = fold
    config["seed"] = seed
    config["kd_weight"] = kd_weight
    validate_config(config)
    return config


def normalized_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Mirror load_config's runtime-only nested precision field without changing raw JSON."""
    validate_config(config)
    result = config.copy()
    model = config["model"].copy()
    model["precision_mode"] = config["precision_mode"]
    result["model"] = model
    return result


def profile_one_config(config: dict[str, Any]) -> dict[str, Any]:
    """Clone a config for one-rank engineering profiling."""
    result = config.copy()
    result["model"] = config["model"].copy()
    result["required_world_size"] = 1
    validate_config(result)
    return result


def profile_peak_config(config: dict[str, Any]) -> dict[str, Any]:
    """Clone a config for peak-LR stress profiling without warmup."""
    result = config.copy()
    result["model"] = config["model"].copy()
    result["warmup_fraction"] = 0.0
    validate_config(result)
    return result


def r3_reference_pair_equal(candidate: dict[str, Any], fold: int, seed: int) -> bool:
    """Compare a candidate to R3 after excluding only fold and seed identities."""
    reference = teacher_base(fold, seed)
    left, right = candidate.copy(), reference.copy()
    left.pop("expected_fold")
    right.pop("expected_fold")
    left.pop("seed")
    right.pop("seed")
    return left == right


def source_path(path: Path) -> str:
    """Return a stable repository-relative reference for a frozen packet input."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()
