from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Protocol


class SuggestingTrial(Protocol):
    """Small Optuna trial surface used by the conditional stage spaces."""

    def suggest_categorical(self, name: str, choices: list[object]) -> object: ...

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False, step: float | None = None
    ) -> float: ...

    def suggest_int(
        self, name: str, low: int, high: int, *, log: bool = False, step: int = 1
    ) -> int: ...


def suggest_stage1(
    trial: SuggestingTrial, families: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    """Draw one model family and only its reviewed conditional parameters."""
    family = trial.suggest_categorical("family", sorted(families))
    if not isinstance(family, str) or family not in families:
        raise ValueError("Optuna returned an unadmitted model family")
    family_spec = families[family]
    if not isinstance(family_spec, Mapping):
        raise ValueError(f"family {family} recipe must be an object")
    search_space = family_spec.get("search_space", {})
    if not isinstance(search_space, Mapping):
        raise ValueError(f"family {family} search_space must be an object")
    params: dict[str, object] = {}
    pending = dict(search_space)
    while pending:
        ready = [
            name
            for name, raw_spec in pending.items()
            if not isinstance(raw_spec, Mapping)
            or not isinstance(raw_spec.get("conditional_on"), Mapping)
            or raw_spec["conditional_on"].get("dimension") in params
        ]
        if not ready:
            raise ValueError(
                f"conditional stage1 dimensions have a dependency cycle: {sorted(pending)}"
            )
        for name in sorted(ready):
            raw_spec = pending.pop(name)
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"search dimension {family}.{name} must be an object")
            condition = raw_spec.get("conditional_on")
            if isinstance(condition, Mapping):
                parent, allowed = condition.get("dimension"), condition.get("choices")
                if not isinstance(parent, str) or not isinstance(allowed, list):
                    raise ValueError(f"conditional search dimension {family}.{name} is malformed")
                if params[parent] not in allowed:
                    if "otherwise" not in condition:
                        raise ValueError(f"conditional dimension {family}.{name} has no fallback")
                    params[name] = condition["otherwise"]
                    continue
            _suggest_stage1_dimension(trial, family, name, raw_spec, params)
    return family, params


def _suggest_stage1_dimension(
    trial: SuggestingTrial,
    family: str,
    name: str,
    raw_spec: Mapping[str, object],
    params: dict[str, object],
) -> None:
    """Suggest one ready conditional dimension using its frozen Optuna kind."""
    key, kind = f"{family}.{name}", raw_spec.get("kind", raw_spec.get("type"))
    if kind == "categorical":
        choices = raw_spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search dimension {key} needs choices")
        value = trial.suggest_categorical(key, choices)
    elif kind == "float":
        low, high = raw_spec.get("low"), raw_spec.get("high")
        if not _finite_number(low) or not _finite_number(high) or low >= high:
            raise ValueError(f"search dimension {key} needs finite ordered bounds")
        log, step = raw_spec.get("log", False), raw_spec.get("step")
        if type(log) is not bool or (step is not None and not _finite_number(step)):
            raise ValueError(f"search dimension {key} has invalid float options")
        value = trial.suggest_float(key, float(low), float(high), log=log, step=step)
    elif kind == "int":
        low, high = raw_spec.get("low"), raw_spec.get("high")
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError(f"search dimension {key} needs integer ordered bounds")
        log, step = raw_spec.get("log", False), raw_spec.get("step", 1)
        if type(log) is not bool or type(step) is not int or step < 1:
            raise ValueError(f"search dimension {key} has invalid integer options")
        value = trial.suggest_int(key, low, high, log=log, step=step)
    else:
        raise ValueError(f"search dimension {key} has unsupported kind")
    params[name] = value


def apply_stage1_config(
    base_config: Mapping[str, object],
    family: str,
    params: Mapping[str, object],
    family_spec: Mapping[str, object],
) -> dict[str, object]:
    """Apply reviewed fixed and sampled paths to a fresh recipe copy."""
    config = json.loads(json.dumps(base_config, allow_nan=False))
    overrides, dimensions = (
        family_spec.get("fixed_overrides", {}),
        family_spec.get("search_space", {}),
    )
    if not isinstance(overrides, Mapping) or not isinstance(dimensions, Mapping):
        raise ValueError("family overrides and search space must be mappings")
    for path, value in overrides.items():
        if not isinstance(path, str):
            raise ValueError("config override paths must be strings")
        _set_existing_path(config, path, value)
    assigned_paths: set[str] = set()
    for name, value in params.items():
        raw_spec = dimensions.get(name)
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"search dimension {family}.{name} is malformed")
        config_path, config_paths = raw_spec.get("config_path"), raw_spec.get("config_paths")
        if isinstance(config_path, str) and config_paths is None:
            paths = [config_path]
        elif config_path is None and isinstance(config_paths, list) and config_paths:
            if not all(isinstance(path, str) for path in config_paths):
                raise ValueError(f"search dimension {family}.{name} config_paths must be strings")
            paths = config_paths
        else:
            raise ValueError(
                f"search dimension {family}.{name} needs one config_path or config_paths"
            )
        if len(set(paths)) != len(paths):
            raise ValueError(f"search dimension {family}.{name} repeats a config path")
        if assigned_paths.intersection(paths):
            raise ValueError("two sampled dimensions assign the same config path")
        assigned_paths.update(paths)
        for path in paths:
            _set_existing_path(config, path, value)
    if config.get("precision_mode") != "fp32":
        raise ValueError("stage1 config changed the strict-FP32 policy")
    if config.get("training_mask") != "all_after_warmup":
        raise ValueError("stage1 config must use the all-after-warmup train proxy")
    if config.get("checkpoint_retention") != "all_epochs":
        raise ValueError("stage1 profiles must retain every safe epoch checkpoint")
    monitoring = config.get("monitoring")
    if not isinstance(monitoring, dict):
        raise ValueError("stage1 configs require producer-native monitoring")
    if monitoring.get("training_telemetry_every_steps", 1) != 1:
        raise ValueError("exact-profile verification requires per-update training telemetry")
    return config


def _set_existing_path(config: object, dotted_path: str, value: object) -> None:
    """Set one existing JSON leaf without permitting accidental schema expansion."""
    parts, node = dotted_path.split("."), config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"config path does not exist: {dotted_path}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"config path does not exist: {dotted_path}")
    node[parts[-1]] = value


def _finite_number(value: object) -> bool:
    """Return true for a finite JSON number excluding Boolean values."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
