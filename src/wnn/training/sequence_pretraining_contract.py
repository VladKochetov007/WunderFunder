from __future__ import annotations

from collections.abc import Mapping

FUTURE_INNOVATION_KIND = "future_input_innovations_v1"
FUTURE_INNOVATION_HORIZONS = (16, 64, 256)
FUTURE_INNOVATION_WIDTH = 112 * len(FUTURE_INNOVATION_HORIZONS)
PARENT_INITIALIZER_KIND = FUTURE_INNOVATION_KIND

FUTURE_INNOVATION_CONTRACT: dict[str, object] = {
    "kind": FUTURE_INNOVATION_KIND,
    "horizons": list(FUTURE_INNOVATION_HORIZONS),
    "target_transform": "raw_input[t+h]-raw_input[t]",
    "standardization": "independent float64 population moments per horizon and input feature; fit members only",
    "training_rows": "row_index>=99 and row_index<sequence_length-256",
    "feature_width_per_horizon": 112,
}


def validate_future_training_config(config: Mapping[str, object]) -> None:
    """Admit only the bounded future-innovation pretrain and its four-epoch child."""
    if "pretraining" in config and config.get("pretraining") is None:
        raise ValueError("pretraining cannot be null")
    if "parent_initializer" in config and config.get("parent_initializer") is None:
        raise ValueError("parent_initializer cannot be null")
    pretraining = config.get("pretraining")
    initializer = config.get("parent_initializer")
    if pretraining is not None:
        _validate_pretraining(config, pretraining, initializer)
    elif initializer is not None:
        _validate_child_finetune(config, initializer)


def _validate_pretraining(
    config: Mapping[str, object], pretraining: object, initializer: object
) -> None:
    """Require the two-epoch label-free stage and its exact horizon geometry."""
    if (
        not isinstance(pretraining, dict)
        or set(pretraining) != {"kind", "horizons"}
        or pretraining.get("kind") != FUTURE_INNOVATION_KIND
        or pretraining.get("horizons") != list(FUTURE_INNOVATION_HORIZONS)
        or initializer is not None
    ):
        raise ValueError("pretraining requires the fixed future-input-innovation contract")
    model = config["model"]
    if (
        config["epochs"] != 2
        or config["seed"] not in (203201, 203202)
        or config.get("objective_mode") is not None
        or config.get("supervised_target") is not None
        or config["kd_weight"] != 0
        or config["hint_weight"] != 1
        or config.get("blend_objective_alpha", 0) != 0
        or "correlation_prediction_std_floor" in config
        or "monitoring" in config
        or not _model_matches(model, hint_dim=FUTURE_INNOVATION_WIDTH)
    ):
        raise ValueError(
            "future-input pretraining requires label-free GRU256x2 and 336 learned hints"
        )
    _validate_common_recipe(config)
    _validate_l4_runtime(config)


def _validate_child_finetune(config: Mapping[str, object], initializer: object) -> None:
    """Require shared two-target supervised fine-tuning from the declared pretext parent."""
    if initializer != PARENT_INITIALIZER_KIND:
        raise ValueError("parent_initializer must name the future-input-innovation parent")
    model = config["model"]
    if (
        config["epochs"] != 4
        or config["seed"] not in (203201, 203202)
        or config["objective_mode"] != "standalone_labels_wmse_v2"
        or config.get("supervised_target") is not None
        or config["kd_weight"] != 0
        or config["hint_weight"] != 0
        or config.get("blend_objective_alpha", 0) != 0
        or not _model_matches(model, hint_dim=0)
    ):
        raise ValueError(
            "pretrained child requires four-epoch shared two-target GRU256x2 supervision"
        )
    _validate_common_recipe(config)
    _validate_l4_runtime(config)


def _validate_common_recipe(config: Mapping[str, object]) -> None:
    """Keep paired stages on the same bounded optimizer and sequence geometry."""
    expected = {
        "batch_size": 32,
        "accumulate_chunks": 1,
        "chunk_rows": 1000,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.1,
        "normalization_scope": "all_fit",
        "normalization_sequences": 10325,
        "training_mask": "all_after_warmup",
        "precision_mode": "fp32",
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "future-innovation stages require the frozen common optimizer and data recipe"
        )


def _model_matches(model: object, *, hint_dim: int) -> bool:
    """Check the fixed shared GRU backbone and requested pretext-head width."""
    return (
        isinstance(model, Mapping)
        and model.get("architecture") == "gru"
        and model.get("hidden") == 256
        and model.get("layers") == 2
        and model.get("hint_dim") == hint_dim
        and model.get("hint_projection") == "learned"
        and model.get("dropout") == 0
    )


def _validate_l4_runtime(config: Mapping[str, object]) -> None:
    """Bind the two-stage screen to one L4 worker per stage."""
    if (
        config.get("required_device") != "L4"
        or config.get("required_world_size") != 1
        or config.get("profile_reload") is not True
    ):
        raise ValueError("future-innovation stages require one L4 and profile_reload=true")
