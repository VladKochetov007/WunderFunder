from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from wnn.galam.variant_config import GALAMVariant

GALAMExecutionBackend = Literal["step", "block"]


def galam_execution_backend(
    values: Mapping[str, object], variant: GALAMVariant
) -> GALAMExecutionBackend:
    """Resolve the explicit execution selector without changing omitted legacy configs."""
    backend = values.get("execution_backend", "step")
    if not isinstance(backend, str) or backend not in ("step", "block"):
        raise ValueError("GALAM execution_backend must be step or block")
    if backend == "block" and variant != "schematic_gru_full":
        raise ValueError("block execution currently supports only schematic_gru_full")
    return cast(GALAMExecutionBackend, backend)
