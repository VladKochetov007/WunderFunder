from __future__ import annotations

import torch

from wnn.training.sequence_model_helpers import detach_state
from wnn.training.sequence_models import build_sequence_model


def test_lstm_sequence_adapter_preserves_state_across_chunks() -> None:
    model = build_sequence_model(
        {
            "architecture": "lstm",
            "hidden": 12,
            "layers": 2,
            "heads": 1,
            "ff_multiplier": 4,
            "context_rows": 32,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "hint_projection": "learned",
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
            "input_feature_mode": "raw112",
        }
    ).eval()
    values = torch.randn(2, 7, 112, generator=torch.Generator().manual_seed(7))

    with torch.inference_mode():
        whole, _, _ = model(values)
        prefix, _, state = model(values[:, :3])
        suffix, _, next_state = model(values[:, 3:], state)

    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), whole)
    assert len(next_state) == 2
    assert all(value.shape == (2, 2, 12) for value in next_state)


def test_lstm_state_detaches_both_hidden_and_cell_tensors() -> None:
    hidden = torch.ones(2, requires_grad=True)
    cell = torch.ones(2, requires_grad=True)

    detached = detach_state((hidden, cell))

    assert isinstance(detached, tuple)
    assert all(not value.requires_grad for value in detached)
