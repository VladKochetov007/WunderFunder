"""Causal, parameter-matched, and gradient-path checks for E425/H106."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import torch
from wnn.specialist_diagnostics import SoftRouteDiagnostics

from wnn.specialist import PREFIX_ROWS, SpecialistGRUModel, causal_prefix_mean

REPO_ROOT = Path(__file__).resolve().parents[5]


def _production_model(route: str) -> SpecialistGRUModel:
    return SpecialistGRUModel(
        n_in=211,
        hidden=96,
        expert_hidden=32,
        experts=4,
        layers=1,
        n_out=23,
        route=route,
        prefix_rows=99,
        hash_seed=1729,
    )


def _root_specialist_module() -> ModuleType:
    """Load the frozen root implementation without changing the WNN overlay."""
    source = REPO_ROOT / "src/wnn/specialist.py"
    spec = importlib.util.spec_from_file_location("e425_root_specialist_reference", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the root specialist reference from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soft_route_preserves_exact_production_parameter_budget() -> None:
    models = [_production_model(route) for route in ("learned", "soft", "hash", "universal")]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [154_343] * 4
    state_keys = tuple(models[0].state_dict())
    assert all(tuple(model.state_dict()) == state_keys for model in models)


def test_learned_hard_forward_matches_frozen_root_with_identical_parameters() -> None:
    torch.manual_seed(424)
    root_module = _root_specialist_module()
    root_model = root_module.SpecialistGRUModel(
        n_in=211,
        hidden=96,
        expert_hidden=32,
        experts=4,
        layers=1,
        n_out=23,
        route="learned",
        prefix_rows=99,
        hash_seed=1729,
    ).eval()
    staged_model = _production_model("learned").eval()
    assert sum(parameter.numel() for parameter in root_model.parameters()) == 154_343
    assert sum(parameter.numel() for parameter in staged_model.parameters()) == 154_343
    staged_model.load_state_dict(root_model.state_dict(), strict=True)
    assert tuple(root_model.state_dict()) == tuple(staged_model.state_dict())
    for root_value, staged_value in zip(
        root_model.state_dict().values(), staged_model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(root_value, staged_value, rtol=0, atol=0)

    values = torch.randn(2, PREFIX_ROWS + 23, 211)
    with torch.no_grad():
        root_output, root_state = root_model(values)
        staged_output, staged_state = staged_model(values)
    torch.testing.assert_close(staged_output, root_output, rtol=0, atol=0)
    for staged_value, root_value in zip(staged_state, root_state, strict=True):
        torch.testing.assert_close(staged_value, root_value, rtol=0, atol=0)


def test_soft_prefix_is_causal_fixed_after_warmup_and_chunk_exact() -> None:
    torch.manual_seed(425)
    model = _production_model("soft").eval()
    values = torch.randn(3, PREFIX_ROWS + 41, 211)

    with torch.no_grad():
        full, full_state = model(values)
        full_probabilities = model.last_route_probabilities
        first, state = model(values[:, :41])
        second, state = model(values[:, 41:PREFIX_ROWS], state)
        third, state = model(values[:, PREFIX_ROWS:], state)

    chunked = torch.cat((first, second, third), dim=1)
    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)
    assert len(state) == model.num_experts + 3
    assert all(isinstance(component, torch.Tensor) for component in state)
    for chunk_state, full_component in zip(state, full_state, strict=True):
        torch.testing.assert_close(chunk_state, full_component, atol=2e-6, rtol=2e-6)
    assert state[-1].tolist() == [PREFIX_ROWS] * 3
    assert full_probabilities is not None
    assert full_probabilities.shape == (3, len(values[0]), model.num_experts)
    torch.testing.assert_close(
        full_probabilities[:, PREFIX_ROWS - 1 :],
        full_probabilities[:, PREFIX_ROWS - 1 : PREFIX_ROWS].expand(
            -1, values.shape[1] - PREFIX_ROWS + 1, -1
        ),
        rtol=0,
        atol=0,
    )

    changed_suffix = values.clone()
    changed_suffix[:, PREFIX_ROWS:] += 100
    with torch.no_grad():
        changed, _ = model(changed_suffix)
        changed_probabilities = model.last_route_probabilities
    torch.testing.assert_close(
        changed[:, :PREFIX_ROWS], full[:, :PREFIX_ROWS], atol=2e-6, rtol=2e-6
    )
    assert changed_probabilities is not None
    torch.testing.assert_close(changed_probabilities, full_probabilities, rtol=0, atol=0)


def test_soft_route_streaming_batch_and_reset_match_independent_sequences() -> None:
    torch.manual_seed(4251)
    model = _production_model("soft").eval()
    values = torch.randn(3, PREFIX_ROWS + 7, 211)
    normalized = (values - model.mean) / model.std

    with torch.no_grad():
        batched, batch_state = model(values)
        separate = [model(values[index : index + 1])[0] for index in range(len(values))]
        separate_output = torch.cat(separate, dim=0)
        torch.testing.assert_close(batched, separate_output, atol=2e-6, rtol=2e-6)

        streamed_parts = []
        state = None
        for row in values.split(1, dim=1):
            output, state = model(row, state)
            streamed_parts.append(output)
        streamed = torch.cat(streamed_parts, dim=1)
        torch.testing.assert_close(streamed, batched, atol=2e-6, rtol=2e-6)

        # GRU states and causal prefix means are observable model behavior. The
        # internal accumulated prefix sum can differ by FP32 reduction order.
        for streamed_state, whole_state in zip(state[:-2], batch_state[:-2], strict=True):
            torch.testing.assert_close(streamed_state, whole_state, atol=2e-6, rtol=2e-6)
        batch_prefix, (batch_sum, batch_count) = causal_prefix_mean(normalized)
        streamed_prefixes = []
        prefix_state = None
        for row in normalized.split(1, dim=1):
            prefix, prefix_state = causal_prefix_mean(row, prefix_state)
            streamed_prefixes.append(prefix)
        streamed_prefix = torch.cat(streamed_prefixes, dim=1)
        streamed_sum, streamed_count = prefix_state
        torch.testing.assert_close(batch_prefix, streamed_prefix, atol=2e-6, rtol=2e-6)
        assert torch.equal(batch_count, streamed_count)
        torch.testing.assert_close(
            batch_sum / batch_count[:, None],
            streamed_sum / streamed_count[:, None],
            atol=2e-6,
            rtol=2e-6,
        )

        prefix_length = min(PREFIX_ROWS, normalized.shape[1])
        counts64 = torch.arange(1, prefix_length + 1, dtype=torch.float64)
        fp64_prefix = normalized[:, :prefix_length].double().cumsum(dim=1) / counts64[None, :, None]
        if prefix_length < normalized.shape[1]:
            frozen = fp64_prefix[:, -1:].expand(-1, normalized.shape[1] - prefix_length, -1)
            fp64_prefix = torch.cat((fp64_prefix, frozen), dim=1)
        torch.testing.assert_close(batch_prefix.double(), fp64_prefix, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(streamed_prefix.double(), fp64_prefix, atol=2e-6, rtol=2e-6)

        # Each FP32 accumulation is bounded by gamma_n * sum(abs(x)); check
        # both accumulators against an independent FP64 sum and bound their
        # difference by the sum of those two absolute-error bounds.
        fp32_epsilon = torch.finfo(torch.float32).eps
        gamma = (prefix_length * fp32_epsilon) / (1 - prefix_length * fp32_epsilon)
        absolute_mass = normalized[:, :prefix_length].double().abs().sum(dim=1)
        exact_prefix_sum = normalized[:, :prefix_length].double().sum(dim=1)
        roundoff_bound = gamma * absolute_mass + 1e-7
        batch_sum_error = (batch_sum.double() - exact_prefix_sum).abs()
        streamed_sum_error = (streamed_sum.double() - exact_prefix_sum).abs()
        assert torch.all(batch_sum_error <= roundoff_bound)
        assert torch.all(streamed_sum_error <= roundoff_bound)
        assert torch.all(batch_sum_error + streamed_sum_error <= 2 * roundoff_bound)
        assert torch.all((batch_sum - streamed_sum).double().abs() <= 2 * roundoff_bound)
        assert torch.equal(state[-1], batch_state[-1])

        model(values[:1])  # Leave a nonempty prior state in the module's last forward.
        reset_first, _ = model(values[1:2])
        model(values[:1])
        reset_second, _ = model(values[1:2])
        torch.testing.assert_close(reset_first, reset_second, rtol=0, atol=0)


def test_soft_route_gives_all_experts_prediction_gradients_and_updates() -> None:
    torch.manual_seed(426)
    model = _production_model("soft")
    values = torch.randn(2, PREFIX_ROWS + 5, 211)
    output, _ = model(values)
    probabilities = model.last_route_probabilities
    assert probabilities is not None
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    assert torch.isfinite(entropy)
    assert 0 <= entropy.item() <= math.log(model.num_experts) + 1e-6

    loss = output.square().mean()
    loss.backward()
    assert model.router[-1].weight.grad is not None
    assert torch.isfinite(model.router[-1].weight.grad).all()
    assert model.router[-1].weight.grad.norm().item() > 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    diagnostics = SoftRouteDiagnostics(model)
    snapshots = diagnostics.before_optimizer_step(model)
    optimizer.step()
    diagnostics.after_optimizer_step(model, snapshots)
    summary = diagnostics.finish()
    assert summary["optimizer_attempts"] == 1
    assert summary["route_rows"] == values.shape[0] * values.shape[1]
    assert len(summary["route_probability_mean"]) == model.num_experts
    assert math.isclose(sum(summary["route_probability_mean"]), 1.0, abs_tol=1e-6)
    assert 0 <= summary["route_entropy_mean_nats"] <= math.log(model.num_experts) + 1e-6
    assert summary["expert_prediction_gradient_nonzero_steps"] == [1] * model.num_experts
    assert summary["expert_parameter_nonzero_update_steps"] == [1] * model.num_experts
    for index, (expert, head) in enumerate(zip(model.experts, model.expert_heads, strict=True)):
        assert expert.weight_ih_l0.grad is not None, f"expert {index} GRU received no gradient"
        assert head.weight.grad is not None, f"expert {index} head received no gradient"
        assert summary["expert_prediction_gradient_l2_mean"][index] > 0
        assert summary["expert_parameter_update_l2_mean"][index] > 0


def test_uniform_soft_weights_equal_universal_expert_average() -> None:
    torch.manual_seed(427)
    soft = _production_model("soft").eval()
    universal = _production_model("universal").eval()
    universal.load_state_dict(soft.state_dict())
    with torch.no_grad():
        soft.router[-1].weight.zero_()
        soft.router[-1].bias.zero_()

    values = torch.randn(2, 17, 211)
    with torch.no_grad():
        soft_output, soft_state = soft(values)
        universal_output, universal_state = universal(values)

    torch.testing.assert_close(soft_output, universal_output, atol=0, rtol=0)
    for soft_component, universal_component in zip(soft_state, universal_state, strict=True):
        torch.testing.assert_close(soft_component, universal_component, atol=0, rtol=0)
