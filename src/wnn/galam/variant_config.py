from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

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
