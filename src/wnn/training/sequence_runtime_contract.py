"""Small identity-bound descriptions of chunked sequence training semantics."""

from __future__ import annotations

from collections.abc import Mapping

_UPDATE_TELEMETRY_FIELDS = {
    "optimizer_update_diagnostics_every",
    "lr_role_factors",
    "lr_head_factor",
    "lr_body_factor",
    "gradient_horizon_rows",
}


def record_sequence_runtime_contract(
    config: Mapping[str, object], provenance: dict[str, object]
) -> None:
    """Bind opt-in update telemetry recipes to their actual chunked gradient horizon."""
    if not _UPDATE_TELEMETRY_FIELDS.intersection(config):
        return
    chunk_rows = config.get("chunk_rows")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("sequence runtime contract requires positive chunk_rows")
    declared_horizon = config.get("gradient_horizon_rows", chunk_rows)
    if type(declared_horizon) is not int or declared_horizon != chunk_rows:
        raise ValueError("gradient_horizon_rows must equal the actual chunk_rows")
    model = config.get("model")
    architecture = model.get("architecture") if isinstance(model, Mapping) else None
    if not isinstance(architecture, str):
        raise ValueError("sequence runtime contract requires a model architecture")
    attention_architectures = {"transformer", "gru_local", "galam"}
    state_kinds = {
        "gru": "persistent_recurrent_state",
        "lstm": "persistent_recurrent_state",
        "transformer": "detached_transformer_kv_cache",
        "gru_local": "persistent_recurrent_and_detached_kv_cache",
        "mamba2": "detached_selective_state_space_cache",
        "galam": "detached_hybrid_lstm_attention_convolution_state",
        "ema_tabm_style": "detached_multiscale_ema_feature_state",
    }
    state_updates = {
        "gru": "detached hidden state spans optimizer updates; carried values reflect earlier weights",
        "lstm": "detached hidden and cell state span optimizer updates; carried values reflect earlier weights",
        "transformer": "detached key/value cache spans optimizer updates; prior chunks use earlier weights, so cache is approximate",
        "gru_local": "detached GRU state and attention cache span optimizer updates; carried values reflect earlier weights",
        "mamba2": "detached selective state and convolution cache span optimizer updates; carried values reflect earlier weights",
        "galam": "detached LSTM, attention, and convolution state span optimizer updates; carried values reflect earlier weights",
        "ema_tabm_style": "detached multiscale EMA feature state spans optimizer updates; carried values reflect earlier weights",
    }
    if architecture not in state_kinds:
        raise ValueError(f"sequence runtime contract has no state description for {architecture}")
    attention_rows = model.get("context_rows") if architecture in attention_architectures else None
    if attention_rows is not None and (type(attention_rows) is not int or attention_rows < 1):
        raise ValueError("Transformer runtime contract requires positive model.context_rows")
    provenance["sequence_runtime_contract"] = {
        "schema": "wnn.sequence-runtime-contract.v1",
        "architecture": architecture,
        "sequence_rows": 20_000,
        "forward_state_kind": state_kinds[architecture],
        "state_carry": "detached across chunks and optimizer updates within a sequence batch",
        "state_reset": "new sequence minibatch",
        "state_detach_boundary": "after every chunk",
        "chunk_rows": chunk_rows,
        "tbptt_gradient_horizon_rows": chunk_rows,
        "declared_gradient_horizon_rows": declared_horizon,
        "accumulate_chunks": config.get("accumulate_chunks"),
        "accumulation_changes_gradient_horizon": False,
        "attention_window_rows_per_layer": attention_rows,
        "attention_max_cached_rows": (
            None if attention_rows is None else min(attention_rows - 1, 20_000)
        ),
        "state_update_semantics": state_updates[architecture],
        "context_rows_semantics": _context_rows_semantics(architecture),
    }


def _context_rows_semantics(architecture: str) -> str:
    """Avoid assigning Transformer window meaning to recurrent and state-space models."""
    if architecture in {"gru", "lstm"}:
        return "not a row-window or gradient-horizon control"
    if architecture in {"transformer", "gru_local", "galam"}:
        return "attention cache/window; does not change chunk-level TBPTT horizon"
    if architecture in {"mamba2", "ema_tabm_style"}:
        return "not used by this architecture's carried sequence state"
    return "no attention-window claim"
