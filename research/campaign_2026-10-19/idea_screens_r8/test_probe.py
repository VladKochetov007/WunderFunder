"""Synthetic invariants for the fixed R8 constant-variance control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from idea_screens_r2.views import build_view_for
from idea_screens_r7.model_math import epoch_permutation, predict_rows
from idea_screens_r7.model_training import train_arm

from idea_screens_r8 import scoring
from idea_screens_r8.model import (
    ConstantVarianceMLP,
    initialize_constant_variance_model,
)
from idea_screens_r8.reporting import reference_contrasts
from idea_screens_r8.run_probe import (
    _source_path,
    validate_runtime_versions,
)
from wnn.gmetric import GlobalAccumulator

EXPECTED_INITIAL = "649b11f31a011e43aac99f4e420c596fb191abb68f5daa502e4693764fce1db3"


def test_r8_initial_state_matches_stored_r7_fingerprint() -> None:
    """The R8 subclass starts from the exact full seed-19031 R7 state."""
    model, fingerprints = initialize_constant_variance_model(EXPECTED_INITIAL)
    assert isinstance(model, ConstantVarianceMLP)
    assert fingerprints["initial"] == EXPECTED_INITIAL
    assert fingerprints["candidate_initial"] == EXPECTED_INITIAL
    assert torch.count_nonzero(model.logvar_head.weight).item() == 0


def test_logvariance_is_constant_and_contributes_no_trunk_or_weight_gradient() -> None:
    """Zeroing the head input retains only constant bias gradients."""
    model, _ = initialize_constant_variance_model(EXPECTED_INITIAL)
    features = torch.linspace(-2.0, 2.0, 5 * 224).reshape(5, 224)
    _, logvariance = model.forward_all(features)
    with torch.no_grad():
        model.logvar_head.bias.copy_(torch.tensor([0.25, -0.75]))
    _, logvariance = model.forward_all(features)
    assert torch.equal(logvariance, logvariance[0].expand_as(logvariance))
    assert torch.equal(logvariance[0], model.logvar_head.bias)

    model.zero_grad(set_to_none=True)
    _, logvariance = model.forward_all(features)
    logvariance.sum().backward()
    assert model.trunk.weight.grad is not None
    assert torch.count_nonzero(model.trunk.weight.grad).item() == 0
    assert model.mean_head.weight.grad is None
    assert model.logvar_head.weight.grad is not None
    assert torch.count_nonzero(model.logvar_head.weight.grad).item() == 0
    assert torch.equal(model.logvar_head.bias.grad, torch.full((2,), 5.0))


def test_frozen_r7_noncontrol_trainer_keeps_constant_weights_on_tiny_synthetic_rows() -> None:
    """R8 reuses the R7 non-control schedule; only synthetic rows enter this test."""
    model, _ = initialize_constant_variance_model(EXPECTED_INITIAL)
    features = np.random.default_rng(801).normal(size=(8, 224)).astype(np.float32)
    targets = np.array(
        [
            [-1.5, -0.7],
            [-1.0, 0.2],
            [-0.5, 0.8],
            [0.1, 1.2],
            [0.6, -1.3],
            [1.1, -0.4],
            [1.5, 0.6],
            [0.9, 1.7],
        ],
        dtype=np.float32,
    )
    epochs: list[dict[str, object]] = []
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    fit = train_arm(
        model,
        features,
        targets,
        "heteroskedastic",
        epochs.append,
        lambda stage: {"stage": stage, "cgroup_memory_current_bytes": 0},
        counters,
    )
    assert fit["arm"] == "heteroskedastic"
    assert fit["epochs_completed"] == 12
    assert fit["attempted_updates"] == fit["successful_updates"] == 12
    assert counters == {"attempted_updates": 12, "successful_updates": 12, "nonfinite_count": 0}
    assert torch.count_nonzero(model.logvar_head.weight).item() == 0
    assert model.logvar_head.weight.grad is not None
    assert torch.count_nonzero(model.logvar_head.weight.grad).item() == 0
    with torch.no_grad():
        raw_logvar = model.forward_all(torch.from_numpy(features))[1]
    assert torch.equal(raw_logvar, raw_logvar[0].expand_as(raw_logvar))
    assert len(epochs) == 12


def test_r8_epoch_order_is_frozen_r7_order() -> None:
    """The fixed seed-plus-epoch row order comes directly from R7."""
    assert torch.equal(epoch_permutation(19, 0), epoch_permutation(19, 0))
    assert not torch.equal(epoch_permutation(19, 0), epoch_permutation(19, 1))


def test_single_endpoint_scorer_matches_canonical_pool_and_keeps_drop_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reduced synthetic fixture exercises six-sum and pooled scorer wiring."""
    monkeypatch.setattr(scoring, "SEQUENCE_ROWS", 4)
    monkeypatch.setattr(scoring, "EVAL_COMPONENTS", 2)
    monkeypatch.setattr(scoring, "EVAL_SCORED_ROWS", 3)
    model, _ = initialize_constant_variance_model(EXPECTED_INITIAL)
    sequences = [
        {
            "component": 301,
            "x": np.arange(4 * 112, dtype=np.float32).reshape(4, 112) / 100.0,
            "y": np.array([[0.3, -0.7], [0.9, 0.5], [2.4, -1.2], [0.1, 1.0]], dtype=np.float32),
            "scored": np.array([True, True, False, False]),
        },
        {
            "component": 302,
            "x": np.ones((4, 112), dtype=np.float32),
            "y": np.array([[-0.8, 1.4], [0.2, 0.1], [0.4, -0.5], [1.0, 0.2]], dtype=np.float32),
            "scored": np.array([True, False, False, False]),
        },
    ]
    mean = np.zeros(224, dtype=np.float32)
    scale = np.ones(224, dtype=np.float32)
    score = scoring.score_endpoint(model, sequences, mean, scale)
    expected = GlobalAccumulator()
    for item in sequences:
        view = build_view_for(item["x"], scoring.VIEW_NAME)
        x = (view[item["scored"]] - mean) / scale
        prediction, _ = predict_rows(model, x)
        expected.add(item["y"][item["scored"]], prediction, np.ones(len(prediction), dtype=bool))
    assert score["official_metric"] == expected.result()
    assert score["rows"] == 3
    assert set(score["component_rows"]) == {"301", "302"}
    assert len(score["component_drop_one_scores"]) == 2
    assert all(
        len(sums) == 2 and len(sums[0]) == 6
        for sums in score["raw_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp"].values()
    )


def test_contrast_signs_use_stored_r7_scalars_and_all_drop_one_components() -> None:
    """Primary/secondary delta directions are fixed by the R8 design."""
    score = {
        "official_metric": {"wp": 0.4},
        "component_drop_one_scores": {"1": {"wp": 0.3}, "2": {"wp": 0.5}},
    }
    reference = {
        "result_sha256": "frozen",
        "heteroskedastic_wp": 0.6,
        "control_wp": 0.2,
        "heteroskedastic_drop_one": {"1": {"wp": 0.7}, "2": {"wp": 0.1}},
        "control_drop_one": {"1": {"wp": 0.1}, "2": {"wp": 0.2}},
    }
    contrasts = reference_contrasts(score, reference)
    assert contrasts["primary_stored_r7_heteroskedastic_minus_r8_constant_wp"] == pytest.approx(0.2)
    assert contrasts["secondary_r8_constant_minus_stored_r7_control_wp"] == pytest.approx(0.2)
    assert contrasts["drop_one_contrasts"]["1"] == {
        "primary_stored_r7_heteroskedastic_minus_r8_constant_wp": pytest.approx(0.4),
        "secondary_r8_constant_minus_stored_r7_control_wp": pytest.approx(0.2),
    }
    assert set(contrasts["drop_one_contrasts"]) == {"1", "2"}


def test_b_runtime_gate_and_manifest_path_safety() -> None:
    """Version and source-key checks reject unbound runtime or traversal paths."""
    validate_runtime_versions("2.2.6", "2.14.0+cu130")
    with pytest.raises(RuntimeError):
        validate_runtime_versions("2.2.6", "2.14.0")
    with pytest.raises(ValueError):
        _source_path("../outside.py", Path("/tmp/support"), Path("/tmp/base/source"))
    with pytest.raises(ValueError):
        _source_path("canonical:../outside.py", Path("/tmp/support"), Path("/tmp/base/source"))


def test_compact_r7_result_fixture_is_hash_bound() -> None:
    """The exact reviewed compact result fixture remains available as evidence."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "idea_screens_r7/records/probe_v1/terminal_capture/RESULT.json"
    )
    payload = fixture.read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "dd36e82a51672a9b9e2ab694dd92fcd827a2445d607927548200d333d1f4b9e0"
    )
    result = json.loads(payload)
    assert result["schema"] == "nf19.r7_mean_variance.result.v1"
    assert result["status"] == "completed"
    assert set(result["final_evaluation"]["arms"]) == {"control", "heteroskedastic"}
