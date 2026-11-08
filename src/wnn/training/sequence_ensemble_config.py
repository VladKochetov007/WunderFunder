from __future__ import annotations

from collections.abc import Mapping

LABELS_MODE = "labels_wmse_v1"
MIXTURE_MODE = "mixture_wmse_pearson_v1"
STANDALONE_LABELS_MODE = "standalone_labels_wmse_v2"
P5_CONFIG_FIELDS = frozenset(
    (
        "gradient_horizon_rows",
        "optimizer_update_diagnostics_every",
        "lr_role_factors",
        "lr_head_factor",
        "lr_body_factor",
    )
)


def objective_contract(config: Mapping[str, object]) -> dict[str, object] | None:
    """Describe the new standalone-label objective before it enters run identity."""
    if config.get("objective_mode") != STANDALONE_LABELS_MODE:
        return None
    return {
        "kind": "standalone_label_weighted_mse",
        "mode": STANDALONE_LABELS_MODE,
        "target_clip": [-2.0, 2.0],
        "target_weight": "abs(clipped_target)",
        "aggregation": "global_ddp_weighted_wmse_per_actual_optimizer_window",
        "pearson_term": "none",
        "reference_training_ancestry": "none",
    }


def validate_ensemble_objective(config: Mapping[str, object]) -> None:
    """Admit only the explicit four-fit package; omission preserves historical objectives."""
    mode = config.get("objective_mode")
    if mode is None:
        if "correlation_prediction_std_floor" in config:
            raise ValueError("correlation floor requires the explicit treatment objective")
        return
    if mode == STANDALONE_LABELS_MODE:
        if (
            config["kd_weight"] != 0
            or config["hint_weight"] != 0
            or config["model"]["hint_dim"] != 0
            or config.get("blend_objective_alpha", 0) != 0
            or "correlation_prediction_std_floor" in config
        ):
            raise ValueError(
                "standalone labels require no KD, hints, blend, or correlation objective"
            )
        if (
            config["normalization_scope"] != "all_fit"
            or config["training_mask"] != "all_after_warmup"
        ):
            raise ValueError("standalone labels require all-fit normalization and all-after-warmup")
        if not isinstance(config.get("monitoring"), Mapping):
            raise ValueError("standalone labels require full official-valid report-only monitoring")
        return
    if mode not in (LABELS_MODE, MIXTURE_MODE):
        raise ValueError("unsupported explicit ensemble objective mode")
    if P5_CONFIG_FIELDS.intersection(config):
        raise ValueError("frozen four-fit objectives reject new runtime and optimizer controls")
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
