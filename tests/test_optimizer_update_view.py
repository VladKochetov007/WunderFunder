from __future__ import annotations

import pytest

from wnn.optimizer_update_view import (
    flatten_optimizer_update_diagnostics,
    optimizer_update_group_context,
)


def test_actual_updates_keep_zero_rates_and_role_context() -> None:
    """Keep measured zeroes and actual rates without confusing roles with numbers."""
    payload = {
        "sampled_update_step": 1,
        "sampling_interval": 50,
        "preclip_gradient_norm": 3.0,
        "postclip_gradient_norm": 1.0,
        "applied_learning_rates": {"adamw.head": 0.0},
        "groups": {
            "adamw.head": {
                "role": "adamw_matrix",
                "parameter_partition": "head",
                "delta_weight_norm": 0.0,
                "delta_weight_over_pre_weight_norm": 0.0,
                "zero_weight_denominator_count": 1,
                "applied_lr": 0.0,
                "momentum_norm": 0.0,
                "momentum_parameter_tensor_count": 0,
                "unavailable": None,
                "invalid": float("nan"),
            }
        },
    }
    metrics = flatten_optimizer_update_diagnostics(payload)
    assert metrics["optimizer_update/applied_lr/adamw.head"] == 0.0
    assert metrics["optimizer_update/group/adamw.head/delta_weight_norm"] == 0.0
    assert metrics["optimizer_update/group/adamw.head/zero_weight_denominator_count"] == 1
    assert "optimizer_update/group/adamw.head/delta_weight_over_pre_weight_norm" not in metrics
    assert "optimizer_update/group/adamw.head/momentum_norm" not in metrics
    assert not any(key.endswith(("/role", "/unavailable", "/invalid")) for key in metrics)
    assert optimizer_update_group_context(payload) == {
        "adamw.head": {"role": "adamw_matrix", "parameter_partition": "head"}
    }
    assert flatten_optimizer_update_diagnostics(None) == {}


@pytest.mark.parametrize("payload", [{"groups": []}, {"groups": {"bad/name": {}}}])
def test_actual_update_view_rejects_ambiguous_groups(payload: object) -> None:
    """Fail before silently merging producer groups or dropping a malformed schema."""
    with pytest.raises(ValueError):
        flatten_optimizer_update_diagnostics(payload)
