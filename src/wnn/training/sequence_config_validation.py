"""Focused validators for prospective sequence monitoring and optimizer variants."""

from __future__ import annotations

import math
from collections.abc import Mapping


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
    if (
        not isinstance(value, dict)
        or not base_fields.issubset(value)
        or set(value) - (base_fields | extended_fields)
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
    if "train_eval_mask" in value and value["train_eval_mask"] != "all_after_warmup":
        raise ValueError("monitoring.train_eval_mask must be all_after_warmup")
    extended_runtime = {"eval_batch_size", "training_telemetry_every_steps"}
    if not value.keys() & extended_runtime and "early_stopping" not in value:
        return
    if not extended_runtime.issubset(value) or checkpoint_retention != "all_epochs":
        raise ValueError("extended monitoring requires checkpoint_retention=all_epochs")
    for key in extended_runtime:
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"monitoring.{key} must be a positive integer")
    if "early_stopping" not in value:
        return
    early_stopping = value["early_stopping"]
    if not isinstance(early_stopping, dict) or set(early_stopping) != {
        "min_epoch",
        "patience",
        "min_delta",
    }:
        raise ValueError("monitoring.early_stopping requires min_epoch, patience, and min_delta")
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
    if (early_stopping["min_epoch"], early_stopping["patience"], min_delta) != (5, 4, 0.0005):
        raise ValueError("extended monitoring uses the approved 5/4/0.0005 early-stop policy")


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
