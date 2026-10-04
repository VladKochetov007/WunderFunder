from __future__ import annotations

import torch

from wnn.specialist import PREFIX_ROWS, SpecialistGRUModel


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


def test_soft_route_preserves_exact_production_parameter_budget() -> None:
    models = [_production_model(route) for route in ("learned", "soft", "hash", "universal")]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [154_343] * 4


def test_soft_route_prefix_is_causal_and_chunk_continuation_matches() -> None:
    torch.manual_seed(425)
    model = _production_model("soft").eval()
    values = torch.randn(2, PREFIX_ROWS + 41, 211)

    with torch.no_grad():
        full, full_state = model(values)
        first, state = model(values[:, :41])
        second, state = model(values[:, 41:PREFIX_ROWS], state)
        third, state = model(values[:, PREFIX_ROWS:], state)

    chunked = torch.cat((first, second, third), dim=1)
    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)
    assert len(state) == model.num_experts + 3
    for chunk_state, full_component in zip(state, full_state, strict=True):
        torch.testing.assert_close(chunk_state, full_component, atol=2e-6, rtol=2e-6)
    assert state[-1].tolist() == [PREFIX_ROWS, PREFIX_ROWS]

    changed_suffix = values.clone()
    changed_suffix[:, PREFIX_ROWS:] += 100
    with torch.no_grad():
        changed, _ = model(changed_suffix)
    torch.testing.assert_close(
        changed[:, :PREFIX_ROWS], full[:, :PREFIX_ROWS], atol=2e-6, rtol=2e-6
    )


def test_soft_route_gives_every_expert_prediction_gradient() -> None:
    torch.manual_seed(426)
    model = _production_model("soft")
    values = torch.randn(2, PREFIX_ROWS + 3, 211)
    output, _ = model(values)
    loss = output.square().mean()
    loss.backward()

    assert model.router[-1].weight.grad is not None
    assert torch.isfinite(model.router[-1].weight.grad).all()
    assert model.router[-1].weight.grad.norm().item() > 0
    for index, (expert, head) in enumerate(zip(model.experts, model.expert_heads, strict=True)):
        head_gradient = head.weight.grad
        recurrent_gradient = expert.weight_ih_l0.grad
        assert head_gradient is not None, f"expert {index} head received no gradient"
        assert recurrent_gradient is not None, f"expert {index} GRU received no gradient"
        assert head_gradient.norm().item() > 0, f"expert {index} head gradient is zero"
        assert recurrent_gradient.norm().item() > 0, f"expert {index} GRU gradient is zero"


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
