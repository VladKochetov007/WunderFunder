"""Identity-bound sequence, attention, TBPTT and optimizer-window context."""

from __future__ import annotations

import math
from collections.abc import Mapping

from wnn.data import SEQUENCE_LENGTH

P5_RUNTIME_KEYS = frozenset(
    (
        "gradient_horizon_rows",
        "optimizer_update_diagnostics_every",
        "lr_role_factors",
        "lr_head_factor",
        "lr_body_factor",
    )
)


def runtime_context_contract(config: Mapping[str, object]) -> dict[str, object]:
    """Return declared and actual sequence context without conflating its clocks."""
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("runtime context contract requires a model mapping")
    chunk = config.get("chunk_rows")
    accumulation = config.get("accumulate_chunks")
    if type(chunk) is not int or chunk < 1 or type(accumulation) is not int or accumulation < 1:
        raise ValueError("runtime context requires positive chunk_rows and accumulate_chunks")
    declared = config.get("gradient_horizon_rows")
    if declared is not None:
        if type(declared) is not int or declared < 1:
            raise ValueError("gradient_horizon_rows must be a positive integer")
        if declared != chunk:
            raise ValueError(
                "gradient_horizon_rows must equal chunk_rows; state is detached at each chunk"
            )
    chunk_count = math.ceil(SEQUENCE_LENGTH / chunk)
    window_count = math.ceil(chunk_count / accumulation)
    tail_chunks = chunk_count - (window_count - 1) * accumulation
    tail_chunk_rows = SEQUENCE_LENGTH - (chunk_count - 1) * chunk
    final_window_rows = (tail_chunks - 1) * chunk + tail_chunk_rows
    architecture = str(model.get("architecture", "transformer"))
    if architecture not in ("transformer", "gru"):
        raise ValueError("runtime context supports only Transformer or GRU state")
    transformer = architecture == "transformer"
    return {
        "sequence_rows": SEQUENCE_LENGTH,
        "model_context": {
            "architecture": architecture,
            "forward_state_kind": (
                "per-layer Transformer key/value cache"
                if transformer
                else "continuous GRU recurrent hidden state"
            ),
            "attention_window_rows_per_layer": (
                int(model["context_rows"]) if transformer else None
            ),
            "max_cached_rows_per_layer": (int(model["context_rows"]) - 1 if transformer else None),
            "context_scope": (
                "causal attention window per Transformer layer"
                if transformer
                else "recurrent state"
            ),
        },
        "tbptt": {
            "detach_chunk_rows": chunk,
            "gradient_horizon_rows": chunk,
            "declared_gradient_horizon_rows": declared,
            "optimizer_window_chunks": accumulation,
            "nominal_optimizer_window_rows": chunk * accumulation,
            "optimizer_windows_per_sequence": window_count,
            "final_optimizer_window_rows": final_window_rows,
            "state_detached_at_chunk_boundary": True,
            "state_carried_across_optimizer_updates": True,
            "carried_state_semantics": "detached stale approximation from earlier parameter values",
        },
    }
