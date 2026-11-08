"""Final opt-in recipe validators for the sequence configuration loader."""

from __future__ import annotations

from typing import Any

from wnn.training.sequence_compact_config import validate_compact_recipe
from wnn.training.sequence_config_validation import (
    validate_sequence_objective_config,
    validate_sequence_update_config,
)


def finalize_loaded_config(
    config: dict[str, Any], raw: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Attach runtime precision and run fail-closed opt-in validators."""
    model["precision_mode"] = config["precision_mode"]
    config["model"] = model
    if "objective_mode" in raw:
        validate_sequence_objective_config(config)
    validate_sequence_update_config(config, raw)
    validate_compact_recipe(config)
    return config
