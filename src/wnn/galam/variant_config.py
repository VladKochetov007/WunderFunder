from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from wnn.galam.config import GALAMConfig

GALAMVariant = Literal[
    "schematic_lstm_full",
    "schematic_gru_full",
    "schematic_lstm_local_only",
]
AdaptedGALAMVariant = Literal["schematic_gru_full", "schematic_lstm_local_only"]
DEFAULT_GALAM_VARIANT: GALAMVariant = "schematic_lstm_full"
_GALAM_VARIANTS = frozenset(
    {"schematic_lstm_full", "schematic_gru_full", "schematic_lstm_local_only"}
)


def galam_variant_from_mapping(values: Mapping[str, object]) -> GALAMVariant:
    """Resolve the nested variant selector while preserving the legacy omission default."""
    value = values.get("variant", DEFAULT_GALAM_VARIANT)
    if not isinstance(value, str) or value not in _GALAM_VARIANTS:
        raise ValueError(f"GALAM variant must be one of {sorted(_GALAM_VARIANTS)}")
    return cast(GALAMVariant, value)


def variant_channel_order(config: GALAMConfig, variant: GALAMVariant) -> tuple[str, ...]:
    """Derive the active order from the declared order instead of ignoring its permutation."""
    excluded = {
        "schematic_lstm_full": None,
        "schematic_gru_full": "cell_projection",
        "schematic_lstm_local_only": "global_memory",
    }[variant]
    return tuple(name for name in config.fusion_channel_order if name != excluded)


def validate_variant_config(config: GALAMConfig, variant: GALAMVariant) -> None:
    """Reject tuning unused local-only fields; their canonical placeholders are explicit."""
    if variant != "schematic_lstm_local_only":
        return
    placeholders = {
        "global_blocks": 1,
        "conv_hidden_width": 2 * config.width,
        "conv_kernel_size": 5,
    }
    for name, expected in placeholders.items():
        if getattr(config, name) != expected:
            raise ValueError(
                f"local-only inactive field {name} requires canonical placeholder {expected}; "
                "this field does not configure an active module"
            )


def describe_variant(config: GALAMConfig, variant: GALAMVariant) -> dict[str, object]:
    """Report effective topology and inactive placeholders for recipe/dashboard inspection."""
    validate_variant_config(config, variant)
    inactive = (
        ["global_blocks", "conv_hidden_width", "conv_kernel_size"]
        if variant == "schematic_lstm_local_only"
        else []
    )
    channels = variant_channel_order(config, variant)
    return {
        "variant": variant,
        "recurrent_cell": "gru" if variant == "schematic_gru_full" else "lstm",
        "has_global": variant != "schematic_lstm_local_only",
        "has_cell": variant != "schematic_gru_full",
        "active_fusion_channel_order": list(channels),
        "decoder_width": len(channels) * config.width,
        "inactive_fields": inactive,
    }
