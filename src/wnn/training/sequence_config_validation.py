"""Focused validators for prospective sequence monitoring and optimizer variants."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

STANDALONE_LABELS_OBJECTIVE_MODE = "standalone_labels_wmse_v2"


def validate_sequence_objective_config(value: Mapping[str, object]) -> None:
    """Fail closed for the explicit reference-free weighted-label objective."""
    mode = value.get("objective_mode")
    if mode != STANDALONE_LABELS_OBJECTIVE_MODE:
        raise ValueError(f"unsupported sequence objective mode: {mode!r}")
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("standalone labels objective requires a validated model mapping")
    for key in ("kd_weight", "hint_weight"):
        weight = value.get(key)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight != 0
        ):
            raise ValueError("standalone labels objective requires kd_weight=0 and hint_weight=0")
    if model.get("hint_dim") != 0 or value.get("hint_mode") != "aligned":
        raise ValueError("standalone labels objective forbids hint targets and hint permutation")
    if value.get("training_mask") != "all_after_warmup":
        raise ValueError("standalone labels objective requires training_mask=all_after_warmup")
    if value.get("normalization_scope") != "all_fit":
        raise ValueError("standalone labels objective requires normalization_scope=all_fit")


def validate_runtime_requirements(
    config: dict[str, Any], world_size: int, device_name: str
) -> bool:
    """Check the recipe's optional world-size and device requirements on this rank."""
    required_world_size = config["required_world_size"]
    if required_world_size is not None and world_size != required_world_size:
        raise ValueError(f"recipe requires world size {required_world_size}, received {world_size}")
    required_device = config["required_device"]
    if required_device is not None and required_device not in device_name.upper():
        raise ValueError(f"recipe requires NVIDIA {required_device}, received {device_name!r}")
    return True


def validate_expected_fold(config: dict[str, Any], actual_fold: int) -> bool:
    """Check the loaded split's fold against the recipe's optional fold requirement."""
    expected_fold = config["expected_fold"]
    if expected_fold is not None and actual_fold != expected_fold:
        raise ValueError(f"recipe requires fold {expected_fold}, received fold {actual_fold}")
    return True


def validate_input_prefetch_batches(value: object) -> None:
    """Bound opt-in pinned sequence prefetch to one in-flight batch."""
    if type(value) is not int or value not in (0, 1):
        raise ValueError("input_prefetch_batches must be 0 or 1")


def validate_monitoring_config(
    value: object, raw_model: object, epochs: object, checkpoint_retention: object
) -> None:
    """Validate monitored recipes for the canonical sequence model families."""
    base_fields = {"train_eval_sequences", "diagnostics_every_steps"}
    extended_fields = {
        "train_eval_mask",
        "eval_batch_size",
        "training_telemetry_every_steps",
        "early_stopping",
    }
    selection_fields = {"train_eval_selection"}
    if (
        not isinstance(value, dict)
        or not base_fields.issubset(value)
        or set(value) - (base_fields | extended_fields | selection_fields)
    ):
        raise ValueError("monitoring has unsupported fields or omits its required cohort controls")
    if value.keys() & extended_fields and "train_eval_mask" not in value:
        raise ValueError("extended monitoring controls require train_eval_mask")
    if not isinstance(raw_model, dict) or raw_model.get("architecture") not in (
        "galam",
        "gru",
        "lstm",
        "transformer",
        "gru_local",
        "mamba2",
        "ema_tabm_style",
    ):
        raise ValueError("online monitoring requires a supported sequence model")
    if type(epochs) is not int or not 1 <= epochs <= 10_000:
        raise ValueError("monitored fits require epochs in [1, 10000]")
    for key in ("train_eval_sequences", "diagnostics_every_steps"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"monitoring.{key} must be a positive integer")
    if value["train_eval_sequences"] > 256:
        raise ValueError("monitoring.train_eval_sequences cannot exceed 256")
    if value.get("train_eval_selection", "prefix") not in ("prefix", "hash_rank_v1"):
        raise ValueError("monitoring.train_eval_selection must be prefix or hash_rank_v1")
    if "train_eval_mask" in value and value["train_eval_mask"] != "all_after_warmup":
        raise ValueError("monitoring.train_eval_mask must be all_after_warmup")
    extended_runtime = {"eval_batch_size", "training_telemetry_every_steps"}
    if not value.keys() & extended_runtime and "early_stopping" not in value:
        return
    if not extended_runtime.issubset(value) or (
        checkpoint_retention != "all_epochs"
        and (checkpoint_retention != "final_only" or "early_stopping" in value)
    ):
        raise ValueError("extended monitoring requires checkpoint_retention=all_epochs")
    for key in extended_runtime:
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"monitoring.{key} must be a positive integer")
    if "early_stopping" not in value:
        return
    early_stopping = value["early_stopping"]
    required_policy = {"min_epoch", "patience", "min_delta"}
    if (
        not isinstance(early_stopping, dict)
        or not required_policy.issubset(early_stopping)
        or set(early_stopping) - (required_policy | {"action"})
    ):
        raise ValueError(
            "monitoring.early_stopping requires min_epoch, patience, min_delta, and optional action"
        )
    action = early_stopping.get("action", "stop")
    if action not in ("stop", "report_only"):
        raise ValueError("monitoring.early_stopping.action must be stop or report_only")
    if (
        type(early_stopping["min_epoch"]) is not int
        or not 1 <= early_stopping["min_epoch"] <= epochs
    ):
        raise ValueError("monitoring.early_stopping.min_epoch must be within configured epochs")
    if type(early_stopping["patience"]) is not int or early_stopping["patience"] < 1:
        raise ValueError("monitoring.early_stopping.patience must be positive")
    min_delta = early_stopping["min_delta"]
    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, (int, float))
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("monitoring.early_stopping.min_delta must be finite and nonnegative")
    if action == "stop" and (
        early_stopping["min_epoch"],
        early_stopping["patience"],
        min_delta,
    ) != (5, 4, 0.0005):
        raise ValueError("historical stop action uses the fixed 5/4/0.0005 policy")


def is_galam_lstm64_muon(model: Mapping[str, object]) -> bool:
    """Identify the one approved GALAM hybrid optimizer recipe."""
    galam = model.get("galam")
    return (
        model.get("architecture") == "galam"
        and isinstance(galam, Mapping)
        and galam.get("variant") == "schematic_lstm_full"
        and galam.get("width") == 64
        and galam.get("lstm_layers") == 1
    )
