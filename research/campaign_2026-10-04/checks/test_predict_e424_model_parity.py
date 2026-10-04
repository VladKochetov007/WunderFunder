"""CPU model parity tests for the sealed E424 prediction wrapper."""

from __future__ import annotations

import torch

from wnn.specialist import PrefixAdapterGRUModel, SpecialistGRUModel


def _assert_chunk_and_reset_parity(model: torch.nn.Module) -> None:
    """Check strict state reload, uneven chunk carry, and fresh-call reset."""
    model.eval()
    x = torch.randn(2, 17, 7)
    with torch.no_grad():
        whole, _ = model(x, None)
        state = None
        chunks = []
        for start, stop in ((0, 1), (1, 8), (8, 13), (13, 17)):
            prediction, state = model(x[:, start:stop], state)
            chunks.append(prediction)
        torch.testing.assert_close(torch.cat(chunks, dim=1), whole, rtol=2e-5, atol=2e-6)

        one_sequence = x[:1]
        first, _ = model(one_sequence, None)
        model(torch.randn(1, 5, 7), None)
        after_independent_call, _ = model(one_sequence, None)
        torch.testing.assert_close(after_independent_call, first, rtol=0, atol=0)

    clone = type(model)
    if isinstance(model, SpecialistGRUModel):
        reloaded = clone(
            n_in=7,
            hidden=16,
            expert_hidden=5,
            experts=3,
            layers=1,
            n_out=23,
            route="learned",
            prefix_rows=9,
        )
    else:
        reloaded = clone(
            n_in=7,
            hidden=16,
            layers=2,
            dropout=0.0,
            n_out=23,
            rank=4,
            mode="conditional",
            prefix_rows=9,
        )
    reloaded.load_state_dict(model.state_dict(), strict=True)
    reloaded.eval()
    with torch.no_grad():
        reloaded_prediction, _ = reloaded(x, None)
        torch.testing.assert_close(reloaded_prediction, whole, rtol=0, atol=0)


def test_specialist_prediction_model_cpu_chunk_reset_and_reload_parity() -> None:
    torch.manual_seed(17)
    model = SpecialistGRUModel(
        n_in=7,
        hidden=16,
        expert_hidden=5,
        experts=3,
        layers=1,
        n_out=23,
        route="learned",
        prefix_rows=9,
    )
    _assert_chunk_and_reset_parity(model)


def test_prefix_adapter_prediction_model_cpu_chunk_reset_and_reload_parity() -> None:
    torch.manual_seed(23)
    model = PrefixAdapterGRUModel(
        n_in=7,
        hidden=16,
        layers=2,
        dropout=0.0,
        n_out=23,
        rank=4,
        mode="conditional",
        prefix_rows=9,
    )
    _assert_chunk_and_reset_parity(model)
