from __future__ import annotations

import math
import re
from collections.abc import Mapping

_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _groups(payload: Mapping[str, object]) -> Mapping[str, object]:
    groups = payload.get("groups", {})
    if not isinstance(groups, Mapping):
        raise ValueError("optimizer update groups must be a named mapping")
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in groups):
        raise ValueError("optimizer update group names must be stable path-safe names")
    return groups


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def flatten_optimizer_update_diagnostics(payload: object) -> dict[str, float]:
    """Expose actual sampled updates without importing Torch or inventing absent values."""
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer update diagnostics must be a mapping")
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        number = _number(value)
        if isinstance(key, str) and _NAME.fullmatch(key) and number is not None:
            metrics[f"optimizer_update/{key}"] = number
    rates = payload.get("applied_learning_rates", {})
    if not isinstance(rates, Mapping):
        raise ValueError("applied learning rates must be a named mapping")
    for name, value in rates.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("applied learning-rate group name is unsafe")
        number = _number(value)
        if number is not None:
            metrics[f"optimizer_update/applied_lr/{name}"] = number
    for name, values in _groups(payload).items():
        if not isinstance(values, Mapping):
            raise ValueError("optimizer update group metrics must be a mapping")
        floored = _number(values.get("zero_weight_denominator_count"))
        for key, value in values.items():
            # Relative change is undefined for zero/tiny pre-weights; absolute
            # deltas and the floor flag remain visible, raw JSON remains intact.
            if key == "delta_weight_over_pre_weight_norm" and floored is not None and floored > 0:
                continue
            if key in ("momentum_norm", "exp_avg_norm"):
                state_name = key.removesuffix("_norm")
                if values.get(f"{state_name}_parameter_tensor_count") == 0:
                    continue
            number = _number(value)
            if isinstance(key, str) and _NAME.fullmatch(key) and number is not None:
                metrics[f"optimizer_update/group/{name}/{key}"] = number
    return metrics


def optimizer_update_group_context(payload: object) -> dict[str, dict[str, str]]:
    """Preserve semantic roles and partitions separately from numeric chart series."""
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer update diagnostics must be a mapping")
    context: dict[str, dict[str, str]] = {}
    for name, values in _groups(payload).items():
        if not isinstance(values, Mapping):
            raise ValueError("optimizer update group metrics must be a mapping")
        context[name] = {
            key: str(values[key])
            for key in ("role", "parameter_partition")
            if isinstance(values.get(key), str)
        }
    return context
