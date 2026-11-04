from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_predict import predict_sequence, predict_sequence_batch


class _CumulativeCausalModel(nn.Module):
    """Tiny stateful model whose result exposes cross-sequence state leakage."""

    precision_mode = "fp32"

    def forward(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous = values.new_zeros(values.shape[0], 1) if state is None else state
        cumulative = values[:, :, :1].cumsum(dim=1) + previous.unsqueeze(1)
        prediction = torch.cat((cumulative, -cumulative), dim=-1)
        hints = values.new_zeros(values.shape[0], values.shape[1], 0)
        return prediction, hints, cumulative[:, -1]


def test_batched_sequence_predictions_match_serial_causal_resets() -> None:
    model = _CumulativeCausalModel().eval()
    inputs = torch.randn(5, 31, 112, dtype=torch.float32)

    batched = predict_sequence_batch(model, inputs, 7, torch.device("cpu"))
    serial = np.stack(
        [predict_sequence(model, sequence, 7, torch.device("cpu")) for sequence in inputs]
    )

    np.testing.assert_array_equal(batched, serial)


def test_canonical_recurrent_adapters_match_serial_and_batched_predictions() -> None:
    inputs = torch.randn(3, 31, 112, dtype=torch.float32)
    for architecture in ("gru", "lstm"):
        model = build_sequence_model(
            {
                "architecture": architecture,
                "hidden": 8,
                "layers": 2,
                "heads": 1,
                "context_rows": 8,
                "hint_dim": 2,
                "dropout": 0.0,
                "precision_mode": "fp32",
                "attention_backend": "math",
            }
        ).eval()

        batched = predict_sequence_batch(model, inputs, 7, torch.device("cpu"))
        serial = np.stack(
            [predict_sequence(model, sequence, 7, torch.device("cpu")) for sequence in inputs]
        )

        np.testing.assert_allclose(batched, serial, rtol=1e-6, atol=1e-7)
