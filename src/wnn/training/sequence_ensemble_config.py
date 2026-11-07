from __future__ import annotations

from collections.abc import Mapping

LABELS_MODE = "labels_wmse_v1"
MIXTURE_MODE = "mixture_wmse_pearson_v1"


def validate_ensemble_objective(config: Mapping[str, object]) -> None:
    """Admit only the explicit four-fit package; omission preserves historical objectives."""
    mode = config.get("objective_mode")
    if mode is None:
        if "correlation_prediction_std_floor" in config:
            raise ValueError("correlation floor requires the explicit treatment objective")
        return
    if mode not in (LABELS_MODE, MIXTURE_MODE):
        raise ValueError("unsupported explicit ensemble objective mode")
    expected = {
        "epochs": 4,
        "batch_size": 128,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "required_world_size": 2,
        "required_device": "B300",
        "precision_mode": "bf16_bf16x9",
        "optimizer": "muon",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "normalization_scope": "all_fit",
        "training_mask": "all_after_warmup",
    }
    if any(config.get(key) != value for key, value in expected.items()) or config["seed"] not in (
        101101,
        101102,
    ):
        raise ValueError("ensemble objective requires the exact declared four-fit numerical recipe")
    model = config["model"]
    fixed_model = {
        "architecture": "transformer",
        "hidden": 768,
        "layers": 24,
        "heads": 12,
        "context_rows": 1024,
        "attention_backend": "efficient",
        "compile_blocks": True,
        "dropout": 0.0,
        "hint_dim": 0,
        "checkpoint_blocks": False,
    }
    if any(model.get(key) != value for key, value in fixed_model.items()) or config.get(
        "muon_batch_by_shape", False
    ):
        raise ValueError("ensemble objective model/optimizer packing differs from the contract")
    if config.get("blend_objective_alpha", 0.0) != (0.1 if mode == MIXTURE_MODE else 0.0):
        raise ValueError("label control omits reference; treatment requires fixed alpha0.1")
    if (
        mode == MIXTURE_MODE
        and config.get("correlation_prediction_std_floor") != 0.01
        or mode == LABELS_MODE
        and "correlation_prediction_std_floor" in config
    ):
        raise ValueError("only the treatment declares the frozen0.01-std numerical policy")
    if "monitoring" not in config or config.get("checkpoint_retention") != "final_only":
        raise ValueError(
            "ensemble objective requires report-only full valid and fixed-final retention"
        )
