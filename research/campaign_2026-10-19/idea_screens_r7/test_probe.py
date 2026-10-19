"""Small synthetic invariants for the fixed R7 pair and panel helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from idea_screens_r2.views import build_view_for

from wnn.gmetric import GlobalAccumulator
from wnn.nn import weighted_mse_loss

from .model_math import (
    SEED,
    apply_normalizer,
    bounded_gaussian_loss,
    epoch_permutation,
    fit_normalizer,
    initialize_pair,
    state_fingerprint,
)
from .model_training import (
    BATCH_SIZE,
    _train_epoch,
)
from .result_payload import build_result
from .runtime_support import _read_cgroup_memory


def test_zero_logvariance_matches_canonical_loss_and_mean_gradient() -> None:
    """At s=0, candidate loss and mean gradients equal canonical WMSE."""
    target = torch.tensor([[3.0, -0.5], [0.2, -4.0], [0.0, 1.2]])
    initial = torch.tensor([[0.4, 0.3], [-0.1, 0.7], [0.2, -0.1]])
    valid = torch.tensor([True, True, True])
    canonical_mean = initial.clone().requires_grad_(True)
    candidate_mean = initial.clone().requires_grad_(True)
    zero_logvar = torch.zeros_like(initial, requires_grad=True)
    canonical_loss = weighted_mse_loss(canonical_mean, target, valid)
    candidate_loss = bounded_gaussian_loss(candidate_mean, zero_logvar, target)
    canonical_gradient = torch.autograd.grad(canonical_loss, canonical_mean)[0]
    candidate_gradient = torch.autograd.grad(candidate_loss, candidate_mean)[0]
    assert torch.allclose(candidate_loss, canonical_loss, atol=1e-7, rtol=1e-7)
    assert torch.allclose(candidate_gradient, canonical_gradient, atol=1e-7, rtol=1e-7)


def test_control_variance_head_is_not_evaluated_or_updated() -> None:
    """The control objective never calls the variance head or changes its weights."""
    model, _, _ = initialize_pair()

    def reject_variance_forward(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]
    ) -> None:
        raise AssertionError("control path evaluated the unused variance head")

    model.logvar_head.register_forward_pre_hook(reject_variance_forward)
    features = torch.zeros((8, 224))
    targets = torch.linspace(-1, 1, 16).reshape(8, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    before_weight = model.logvar_head.weight.detach().clone()
    before_bias = model.logvar_head.bias.detach().clone()
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    _train_epoch(model, optimizer, features, targets, "control", 0, counters)
    assert counters == {"attempted_updates": 1, "successful_updates": 1, "nonfinite_count": 0}
    assert torch.equal(model.logvar_head.weight, before_weight)
    assert torch.equal(model.logvar_head.bias, before_bias)
    assert model.logvar_head.weight.grad is None
    assert model.logvar_head.bias.grad is None


def test_nonfinite_adam_state_fails_before_counting_success() -> None:
    """A bad optimizer moment increments failure count and not successful updates."""
    model, _, _ = initialize_pair()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    original_step = optimizer.step

    def corrupt_step() -> None:
        original_step()
        for state in optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].fill_(float("nan"))
                return
        raise AssertionError("AdamW did not create its first-moment state")

    optimizer.step = corrupt_step
    features = torch.zeros((8, 224))
    targets = torch.linspace(-1, 1, 16).reshape(8, 2)
    counters = {"attempted_updates": 0, "successful_updates": 0, "nonfinite_count": 0}
    with pytest.raises(FloatingPointError, match="nonfinite optimizer state"):
        _train_epoch(model, optimizer, features, targets, "control", 0, counters)
    assert counters == {"attempted_updates": 1, "successful_updates": 0, "nonfinite_count": 1}


def test_pair_initialization_and_epoch_permutations_are_shared() -> None:
    """Both arms share the complete zero-variance initialization and order."""
    control, candidate, fingerprints = initialize_pair()
    assert fingerprints["initial"] == fingerprints["control_initial"]
    assert fingerprints["initial"] == fingerprints["candidate_initial"]
    assert torch.count_nonzero(control.logvar_head.weight).item() == 0
    assert torch.count_nonzero(control.logvar_head.bias).item() == 0
    assert torch.equal(epoch_permutation(31808, 4), epoch_permutation(31808, 4))
    assert not torch.equal(epoch_permutation(31808, 4), epoch_permutation(31808, 5))
    assert epoch_permutation(31808, 4).numel() == 31808
    assert (31808 + BATCH_SIZE - 1) // BATCH_SIZE == 63
    assert SEED == 19031
    assert state_fingerprint(control) == state_fingerprint(candidate)


def test_fit_normalizer_uses_only_passed_rows_and_floors_constant_columns() -> None:
    """Fit normalization uses its explicit training input and maps constant scale to one."""
    fit = np.zeros((4, 224), dtype=np.float32)
    fit[:, 0] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mean, scale = fit_normalizer(fit)
    assert mean[0] == 2.5
    assert np.isclose(scale[0], np.std(fit[:, 0], dtype=np.float64))
    assert scale[1] == 1.0
    evaluation_only_value = np.full((2, 224), 10_000.0, dtype=np.float32)
    normalized = apply_normalizer(evaluation_only_value, mean, scale)
    assert mean[0] == 2.5
    assert normalized[0, 0] > 1_000


def test_logvariance_clamp_keeps_extreme_finite_parameters_bounded() -> None:
    """The predeclared logvariance clamp produces finite objective and gradients."""
    mean = torch.tensor([[0.5, -0.2]], requires_grad=True)
    logvar = torch.tensor([[-100.0, 100.0]], requires_grad=True)
    target = torch.tensor([[0.7, 0.4]])
    loss = bounded_gaussian_loss(mean, logvar, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(logvar.grad).all()


def test_signed_ema_view_is_causal_across_a_changed_suffix() -> None:
    """Changing later rows cannot alter an already computed causal feature prefix."""
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(192, 112)).astype(np.float32)
    changed = raw.copy()
    changed[96:] += 10.0
    first = build_view_for(raw, "raw_signed_delta_ema_hl32")
    second = build_view_for(changed, "raw_signed_delta_ema_hl32")
    assert first.shape == (192, 224)
    assert np.array_equal(first[:96], second[:96])


def test_canonical_pool_and_component_drop_one_use_same_scorer() -> None:
    """Pooled additions and component removal retain canonical WP definitions."""
    targets = np.array([[0.2, -0.4], [0.8, 0.1], [-0.7, 0.3], [0.5, 0.9]], dtype=np.float32)
    predictions = np.array([[0.1, -0.2], [0.6, 0.0], [-0.2, 0.4], [0.7, 0.3]], dtype=np.float32)
    masks = np.ones(4, dtype=bool)
    pooled = GlobalAccumulator()
    parts: dict[int, GlobalAccumulator] = {}
    for component, selection in ((11, slice(0, 2)), (22, slice(2, 4))):
        part = GlobalAccumulator()
        part.add(targets[selection], predictions[selection], masks[selection])
        parts[component] = part
        pooled.add(targets[selection], predictions[selection], masks[selection], group=component)
    direct = GlobalAccumulator()
    direct.add(targets, predictions, masks)
    assert np.isclose(pooled.result()["wp"], direct.result()["wp"], atol=1e-12)
    assert set(pooled.groups) == {11, 22}
    assert all(part.selected_rows == 2 for part in parts.values())
    assert torch.isfinite(
        bounded_gaussian_loss(
            torch.tensor(predictions),
            torch.zeros_like(torch.tensor(predictions)),
            torch.tensor(targets),
        )
    )


def test_memory_guard_reads_only_the_exact_self_cgroup(tmp_path) -> None:
    """A root counter cannot stand in for a missing process-specific counter."""
    (tmp_path / "memory.current").write_text("10\n")
    own_cgroup = tmp_path / "job" / "unit"
    own_cgroup.mkdir(parents=True)
    own_memory = own_cgroup / "memory.current"
    own_memory.write_text("25\n")
    assert _read_cgroup_memory(tmp_path, "0::/job/unit\n") == 25
    own_memory.unlink()
    with pytest.raises(FileNotFoundError):
        _read_cgroup_memory(tmp_path, "0::/job/unit\n")


def test_r3_producer_metadata_fixture_has_the_fixed_panel_bindings() -> None:
    """The actual R3 metadata fixture preserves one bound sequence per component."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
    )
    metadata = json.loads(fixture.read_bytes())
    assert metadata["schema"] == "idea_screens_r3.sample.v2"
    records = metadata["records"]
    fit = [record for record in records if record["role"] == "fit"]
    evaluation = [record for record in records if record["role"] == "eval"]
    assert len(fit) == 64
    assert len(evaluation) == 32
    assert (
        metadata["split_sha256"]
        == "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    )
    assert metadata["selection"]["seeds"] == {
        "fit_components": 19021,
        "fit_groups": 19022,
        "eval_components": 19023,
        "eval_groups": 19024,
    }
    fit_components = {record["component"] for record in fit}
    eval_components = {record["component"] for record in evaluation}
    assert len(fit_components) == len(fit)
    assert len(eval_components) == len(evaluation)
    assert fit_components.isdisjoint(eval_components)
    source_groups = [(record["source"], record["group"]) for record in records]
    assert len(set(source_groups)) == len(records)
    for record in records:
        assert record["role"] in ("fit", "eval")
        assert set(record["files"]) == {"x", "y", "scored"}
        for binding in record["files"].values():
            assert set(binding) == {"name", "sha256", "bytes"}
            assert Path(binding["name"]).name == binding["name"]
            assert binding["name"].endswith(".npy")
    assert {record["files"]["x"]["bytes"] for record in records} == {8_960_128}
    assert {record["files"]["y"]["bytes"] for record in records} == {160_128}
    assert {record["files"]["scored"]["bytes"] for record in records} == {20_128}


def test_result_builder_keeps_the_fixed_pair_and_scope_metadata() -> None:
    """The synthetic result contains the validated head flag and bounded scope."""
    arm = {
        "variance_head_unchanged": True,
        "epochs": [],
        "wall_seconds": 0.1,
        "cpu_seconds": 0.1,
        "successful_updates": 756,
        "attempted_updates": 756,
    }
    candidate = {**arm, "variance_head_unchanged": False}
    result = build_result(
        identities={"sample_sha256": "sample"},
        preparation_sha256="preparation",
        initialization_sha256="initialization",
        control_fit=arm,
        candidate_fit=candidate,
        score={"metric_id": "synthetic"},
        archive={"sha256": "archive"},
        runtime={"device": "cpu"},
        source_phase_seconds={"collection": 0.2},
        evaluation_seconds=0.1,
        started_wall=0.0,
        started_cpu=0.0,
    )
    assert result["control_variance_head_unchanged"] is True
    assert result["total_successful_updates"] == 1512
    assert (
        result["budget_interpretation"]
        == "equal examples and optimizer updates, not exact FLOPs or wall time"
    )
    assert result["predictions_in_json_or_archive"] is False
    json.dumps(result, allow_nan=False)
