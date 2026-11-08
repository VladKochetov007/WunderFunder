from __future__ import annotations

import pytest

from wnn.training.sequence_config_validation import validate_sequence_update_config
from wnn.training.sequence_runtime_contract import record_sequence_runtime_contract


def _config(**overrides: object) -> dict[str, object]:
    return {
        "chunk_rows": 1000,
        "accumulate_chunks": 1,
        "model": {"architecture": "gru", "context_rows": 1},
        "optimizer_update_diagnostics_every": 50,
        **overrides,
    }


def test_gru_runtime_contract_uses_chunk_rows_and_describes_carried_state() -> None:
    config = _config(gradient_horizon_rows=1000)
    provenance: dict[str, object] = {}

    validate_sequence_update_config(config, config)
    record_sequence_runtime_contract(config, provenance)

    contract = provenance["sequence_runtime_contract"]
    assert isinstance(contract, dict)
    assert contract["forward_state_kind"] == "persistent_recurrent_state"
    assert contract["chunk_rows"] == contract["tbptt_gradient_horizon_rows"] == 1000
    assert contract["accumulate_chunks"] == 1
    assert contract["accumulation_changes_gradient_horizon"] is False
    assert contract["state_reset"] == "new sequence minibatch"
    assert contract["attention_window_rows_per_layer"] is None
    assert "not a row-window" in contract["context_rows_semantics"]


def test_declared_gradient_horizon_must_match_chunk_rows() -> None:
    config = _config(gradient_horizon_rows=2000)

    with pytest.raises(ValueError, match="must equal chunk_rows"):
        validate_sequence_update_config(config, config)
    with pytest.raises(ValueError, match="must equal the actual chunk_rows"):
        record_sequence_runtime_contract(config, {})


@pytest.mark.parametrize(
    "overrides",
    [
        {"optimizer_update_diagnostics_every": 0},
        {"optimizer_update_diagnostics_every": None},
        {"lr_role_factors": {"adamw": True}},
        {"lr_role_factors": {"adamw": 0.0}},
        {"lr_role_factors": None},
        {"lr_head_factor": float("nan")},
        {"lr_body_factor": False},
        {"gradient_horizon_rows": None},
    ],
)
def test_update_policy_rejects_invalid_sampling_and_factors(
    overrides: dict[str, object],
) -> None:
    config = _config(**overrides)

    with pytest.raises((ValueError, TypeError)):
        validate_sequence_update_config(config, config)


def test_legacy_config_records_no_new_runtime_identity() -> None:
    provenance: dict[str, object] = {"existing": "unchanged"}

    record_sequence_runtime_contract(
        {"chunk_rows": 256, "model": {"architecture": "gru"}}, provenance
    )

    assert provenance == {"existing": "unchanged"}


def test_lstm_runtime_does_not_claim_attention_context() -> None:
    provenance: dict[str, object] = {}
    config = _config(model={"architecture": "lstm", "context_rows": 512})

    record_sequence_runtime_contract(config, provenance)

    contract = provenance["sequence_runtime_contract"]
    assert isinstance(contract, dict)
    assert contract["forward_state_kind"] == "persistent_recurrent_state"
    assert contract["attention_window_rows_per_layer"] is None
    assert contract["attention_max_cached_rows"] is None


def test_transformer_runtime_reports_per_layer_window_separately_from_tbptt() -> None:
    provenance: dict[str, object] = {}
    config = _config(model={"architecture": "transformer", "context_rows": 1024})

    record_sequence_runtime_contract(config, provenance)

    contract = provenance["sequence_runtime_contract"]
    assert isinstance(contract, dict)
    assert contract["forward_state_kind"] == "detached_transformer_kv_cache"
    assert contract["attention_window_rows_per_layer"] == 1024
    assert contract["attention_max_cached_rows"] == 1023
    assert contract["tbptt_gradient_horizon_rows"] == 1000
    assert contract["context_rows_semantics"] == (
        "attention cache/window; does not change chunk-level TBPTT horizon"
    )
