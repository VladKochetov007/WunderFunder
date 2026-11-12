from __future__ import annotations

import math
from collections.abc import Mapping

from wnn.training.sequence_ensemble_config import STANDALONE_LABELS_MODE


def validate_model_config(
    config: Mapping[str, object],
    model: Mapping[str, object],
    input_stem_kernel: int,
    supervised_target: int | None,
    auxiliary_heads: tuple[str, ...],
) -> None:
    """Validate the model architecture and reject incompatible training objectives."""
    for key in ("hidden", "layers", "heads", "ff_multiplier", "context_rows"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if type(model["hint_dim"]) is not int or model["hint_dim"] < 0:
        raise ValueError("hint_dim must be a nonnegative integer")
    if model["architecture"] not in ("transformer", "gru"):
        raise ValueError("architecture must be transformer or gru")
    if input_stem_kernel and model["architecture"] != "gru":
        raise ValueError("input_stem_kernel is currently supported only for GRU models")
    if model["attention_backend"] not in ("math", "efficient", "flex"):
        raise ValueError("model.attention_backend must be math, efficient, or flex")
    if model["architecture"] == "gru" and model["attention_backend"] != "math":
        raise ValueError("attention backends are supported only by Transformer models")
    if type(model["compile_blocks"]) is not bool:
        raise ValueError("model.compile_blocks must be boolean")
    if model["architecture"] == "gru" and model["compile_blocks"]:
        raise ValueError("model.compile_blocks requires a Transformer model")
    if config["optimizer"] == "muon" and model["architecture"] not in (
        "transformer",
        "gru",
    ):
        raise ValueError("sequence Muon requires a Transformer or GRU model")
    if model["hint_projection"] not in ("fixed", "learned"):
        raise ValueError("hint_projection must be fixed or learned")
    dropout = model["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0 <= dropout < 1
    ):
        raise ValueError("model.dropout must be finite and in [0, 1)")
    if type(model["checkpoint_blocks"]) is not bool:
        raise ValueError("checkpoint_blocks must be boolean")
    if model["architecture"] != "gru" and dropout != 0:
        raise ValueError("this paired runtime freezes non-GRU dropout=0")
    if model["architecture"] == "gru" and dropout not in (0, 0.1):
        raise ValueError("this successor runtime supports GRU dropout=0 or 0.1")
    if model["architecture"] == "gru" and dropout > 0 and model["layers"] < 2:
        raise ValueError("GRU dropout requires at least two recurrent layers")
    if model["architecture"] == "transformer" and model["hidden"] % (2 * model["heads"]):
        raise ValueError("RoPE requires an even integral head dimension")
    if config["hint_weight"] > 0 and (
        model["hint_dim"] == 0 or model["hint_projection"] != "learned"
    ):
        raise ValueError("hint distillation requires a learned student projection")
    if supervised_target is not None and (
        config["kd_weight"] > 0
        or config["hint_weight"] > 0
        or model["hint_dim"] != 0
        or config.get("blend_objective_alpha", 0) != 0
        or config.get("objective_mode") != STANDALONE_LABELS_MODE
        or "correlation_prediction_std_floor" in config
    ):
        raise ValueError(
            "single-target supervision requires two-head label-only training without mix or auxiliaries"
        )
    if config["since_weight"] > 0 and (
        model["architecture"] != "gru"
        or config["kd_weight"] != 0
        or config["hint_weight"] != 0
        or model["hint_dim"] != 0
        or config.get("objective_mode") not in (None, STANDALONE_LABELS_MODE)
        or config.get("blend_objective_alpha", 0) != 0
        or supervised_target is not None
        or config.get("pretraining") is not None
        or config.get("parent_initializer") is not None
    ):
        raise ValueError(
            "since weighting currently requires cold, two-head labels-only GRU training"
        )
    auxiliary_weights = {
        "structure": float(config["aux_structure_weight"]),
        "next_change": float(config["aux_next_change_weight"]),
    }
    for head, weight in auxiliary_weights.items():
        if (head in auxiliary_heads) != (weight > 0):
            raise ValueError(
                f"{head} auxiliary head and positive coefficient must be enabled together"
            )
    if auxiliary_heads and (
        model["architecture"] != "gru"
        or config["kd_weight"] != 0
        or config["hint_weight"] != 0
        or model["hint_dim"] != 0
        or config.get("blend_objective_alpha", 0) != 0
        or config.get("objective_mode") not in (None, STANDALONE_LABELS_MODE)
        or config.get("correlation_prediction_std_floor") is not None
        or supervised_target is not None
        or config["since_weight"] != 0
        or config.get("pretraining") is not None
        or config.get("parent_initializer") is not None
    ):
        raise ValueError(
            "auxiliary heads require a fresh GRU with standalone labels and no other auxiliaries"
        )
