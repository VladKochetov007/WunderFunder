from __future__ import annotations

import torch

from wnn.specialist import (
    PREFIX_ROWS,
    PrefixAdapterGRUModel,
    SpecialistGRUModel,
    causal_prefix_mean,
)


def test_prefix_mean_is_causal_frozen_and_chunk_exact() -> None:
    torch.manual_seed(13)
    values = torch.randn(2, PREFIX_ROWS + 7, 5)
    full, full_state = causal_prefix_mean(values)
    first, state = causal_prefix_mean(values[:, :31])
    second, state = causal_prefix_mean(values[:, 31:84], state)
    third, state = causal_prefix_mean(values[:, 84:], state)
    torch.testing.assert_close(torch.cat([first, second, third], dim=1), full)
    torch.testing.assert_close(state[0], full_state[0])
    torch.testing.assert_close(state[1], full_state[1])
    assert state[1].tolist() == [PREFIX_ROWS, PREFIX_ROWS]

    changed_future = values.clone()
    changed_future[:, 1] += 20
    changed, _ = causal_prefix_mean(changed_future)
    torch.testing.assert_close(changed[:, :1], full[:, :1])
    assert not torch.allclose(changed[:, 1], full[:, 1])
    changed_suffix = values.clone()
    changed_suffix[:, PREFIX_ROWS:] += 100
    changed, _ = causal_prefix_mean(changed_suffix)
    torch.testing.assert_close(changed[:, PREFIX_ROWS - 1 :], full[:, PREFIX_ROWS - 1 :])


def test_specialist_router_arms_share_budget_and_all_experts_update() -> None:
    torch.manual_seed(17)
    models = [
        SpecialistGRUModel(7, hidden=12, expert_hidden=4, experts=3, layers=1, route=route)
        for route in ("learned", "hash", "universal")
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [counts[0]] * len(counts)

    x = torch.randn(2, PREFIX_ROWS + 3, 7)
    prefix, _ = causal_prefix_mean(x)
    for model in models[:2]:
        routes = model.route_indices(prefix)
        assert torch.equal(
            routes[:, PREFIX_ROWS - 1 :],
            routes[:, PREFIX_ROWS - 1 : PREFIX_ROWS].expand(-1, x.shape[1] - PREFIX_ROWS + 1),
        )
        altered = x.clone()
        altered[:, PREFIX_ROWS:] += 10
        altered_prefix, _ = causal_prefix_mean(altered)
        assert torch.equal(
            routes[:, PREFIX_ROWS - 1 :], model.route_indices(altered_prefix)[:, PREFIX_ROWS - 1 :]
        )
    target = torch.randn(x.shape[0], x.shape[1], 2)
    model = models[0]
    observed_rows: list[int] = []
    handles = [
        expert.register_forward_hook(
            lambda _module, arguments, _output: observed_rows.append(arguments[0].shape[1])
        )
        for expert in model.experts
    ]
    output, state = model(x)
    for handle in handles:
        handle.remove()
    assert output.shape == target.shape
    assert len(state) == model.num_experts + 3
    assert observed_rows == [x.shape[1]] * model.num_experts
    ((output - target) ** 2).mean().backward()
    assert any(parameter.grad is not None for parameter in model.router.parameters())

    model.eval()
    with torch.no_grad():
        first, first_state = models[1](x[:, :4])
        rest, _ = models[1](x[:, 4:], first_state)
        together, _ = models[1](x)
    torch.testing.assert_close(torch.cat([first, rest], dim=1), together, atol=2e-6, rtol=2e-6)


def test_prefix_adapter_controls_have_equal_parameters_and_causal_chunking() -> None:
    torch.manual_seed(19)
    models = [
        PrefixAdapterGRUModel(6, hidden=10, layers=1, rank=3, mode=mode)
        for mode in PrefixAdapterGRUModel.MODES
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [counts[0]] * len(counts)
    normalized = torch.randn(2, 4, 6)
    prefix = torch.randn(2, 4, 6)
    changed_prefix = prefix + 1
    assert not torch.allclose(
        models[0].encode(normalized, prefix), models[0].encode(normalized, changed_prefix)
    )
    torch.testing.assert_close(
        models[1].encode(normalized, prefix), models[1].encode(normalized, changed_prefix)
    )
    assert not torch.allclose(
        models[2].encode(normalized, prefix), models[2].encode(normalized, changed_prefix)
    )
    x = torch.randn(2, PREFIX_ROWS + 5, 6)

    for model in models:
        full, full_state = model(x)
        left, state = model(x[:, :51])
        right, state = model(x[:, 51:101], state)
        tail, state = model(x[:, 101:], state)
        chunked = torch.cat([left, right, tail], dim=1)
        torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(state[1], full_state[1])
        torch.testing.assert_close(state[2], full_state[2])

        altered = x.clone()
        altered[:, PREFIX_ROWS:] += 50
        with torch.no_grad():
            changed, _ = model(altered)
        # The input rows still affect their own predictions; the test instead
        # isolates the prefix path through adapter modulation.
        original_prefix, _ = causal_prefix_mean(x)
        changed_prefix, _ = causal_prefix_mean(altered)
        torch.testing.assert_close(
            original_prefix[:, :PREFIX_ROWS], changed_prefix[:, :PREFIX_ROWS]
        )
