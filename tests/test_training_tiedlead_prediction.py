from __future__ import annotations

import pytest
import torch

from wnn.data import N_FEATURES
from wnn.tiedlead import TIEDLEAD_INPUT_WIDTH
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.evaluation import predict
from wnn.training.inputs import build_inputs
from wnn.training.models import make_model
from wnn.training.prediction_inputs import configure_prediction_inputs


def test_checkpoint_selected_tiedlead_prediction_uses_263_columns() -> None:
    checkpoint_args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "tiedlead",
        "hidden": 8,
        "layers": 2,
        "dropout": 0.0,
    }
    environment = TrainingEnvironment(inputs=InputConfiguration())
    configure_prediction_inputs(environment, checkpoint_args)
    model = make_model(checkpoint_args, TIEDLEAD_INPUT_WIDTH, 2).eval()
    raw = torch.randn(2, 13, N_FEATURES)

    assembled = build_inputs(raw, environment=environment)
    predictions = predict(
        model, raw, chunk=5, device=torch.device("cpu"), environment=environment
    )

    assert assembled.shape == (2, 13, TIEDLEAD_INPUT_WIDTH)
    assert predictions.shape == (2, 13, 2)
    assert torch.isfinite(torch.from_numpy(predictions)).all()


def test_non_tiedlead_checkpoint_resets_tiedlead_input_mode() -> None:
    environment = TrainingEnvironment(inputs=InputConfiguration(tiedlead=True))
    configure_prediction_inputs(environment, {"enc": "linear"})

    assert not environment.inputs.tiedlead
    assert build_inputs(torch.randn(1, 7, N_FEATURES), environment=environment).shape[-1] == 211


def test_prediction_input_configuration_requires_checkpoint_mapping() -> None:
    environment = TrainingEnvironment()
    with pytest.raises(TypeError, match="mapping"):
        configure_prediction_inputs(environment, [])
