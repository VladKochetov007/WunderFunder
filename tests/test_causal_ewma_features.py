from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.features import StreamingEventFeatures
from wnn.research_features import (
    CAUSAL_EWMA_HALF_LIFE,
    CAUSAL_EWMA_MODE,
    StreamingResearchFeatures,
    research_feature_control_width,
    research_feature_names,
    research_features_numpy,
    research_features_torch,
)
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.inputs import build_inputs
from wnn.training.models import make_model
from wnn.training.source_identity import training_source_identity


def _raw_rows(rows: int, seed: int = 1417) -> np.ndarray:
    """Create deterministic float32 rows for representation-only checks."""
    return np.random.default_rng(seed).normal(size=(rows, 112)).astype(np.float32)


@pytest.mark.parametrize("control", ("none", "raw_zero"))
def test_causal_ewma_control_and_candidate_have_same_width(control: str) -> None:
    raw = torch.from_numpy(_raw_rows(13)[None])
    features = research_features_torch(raw, CAUSAL_EWMA_MODE, control=control)

    assert features.shape == (1, 13, 224)
    assert len(research_feature_names(CAUSAL_EWMA_MODE)) == 224
    assert research_feature_control_width(CAUSAL_EWMA_MODE, "raw_zero") == 112
    if control == "raw_zero":
        torch.testing.assert_close(features[..., :112], raw)
        assert torch.count_nonzero(features[..., 112:]) == 0
        return

    decay_value = 2.0 ** (-1.0 / CAUSAL_EWMA_HALF_LIFE)
    decay = raw.new_tensor(decay_value)
    complement = raw.new_tensor(1.0 - decay_value)
    expected = torch.empty_like(raw)
    expected[:, 0] = raw[:, 0]
    for row in range(1, raw.shape[1]):
        expected[:, row] = decay * expected[:, row - 1] + complement * raw[:, row]
    torch.testing.assert_close(features[..., :112], expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(features[..., 112:], raw - expected, rtol=2e-5, atol=2e-6)


def test_causal_ewma_is_prefix_invariant_and_matches_numpy_streaming() -> None:
    rows = np.stack((_raw_rows(29, 1417), _raw_rows(29, 2718)))
    raw = torch.from_numpy(rows)
    actual = research_features_torch(raw, CAUSAL_EWMA_MODE)
    altered = raw.clone()
    altered[:, 12:] += 100.0
    future_changed = research_features_torch(altered, CAUSAL_EWMA_MODE)
    torch.testing.assert_close(actual[:, :12], future_changed[:, :12])

    expected_numpy = research_features_numpy(rows, CAUSAL_EWMA_MODE)
    np.testing.assert_allclose(actual.numpy(), expected_numpy, rtol=2e-5, atol=2e-6)
    streamed = []
    features = StreamingResearchFeatures(CAUSAL_EWMA_MODE)
    for sequence in rows:
        streamed.append(np.stack([features.update(row) for row in sequence]))
        features.reset()
    np.testing.assert_allclose(actual.numpy(), np.stack(streamed), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("control", ("none", "raw_zero"))
def test_checkpoint_rebuild_and_chunked_prediction_use_saved_feature_mode(
    tmp_path: Path, control: str
) -> None:
    rows = _raw_rows(23, 931)
    mode_args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "research_features": CAUSAL_EWMA_MODE,
        "research_feature_control": control,
    }
    environment = TrainingEnvironment(
        inputs=InputConfiguration(
            research_features=mode_args["research_features"],
            research_feature_control=mode_args["research_feature_control"],
        )
    )
    raw = torch.from_numpy(rows[None])
    full_inputs = build_inputs(raw, environment=environment)
    source_model = make_model(mode_args, full_inputs.shape[-1], 2)
    checkpoint_path = tmp_path / "ewma-checkpoint.pt"
    torch.save(
        {"args": mode_args, "n_in": full_inputs.shape[-1], "model": source_model.state_dict()},
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_environment = TrainingEnvironment(
        inputs=InputConfiguration(
            research_features=checkpoint["args"]["research_features"],
            research_feature_control=checkpoint["args"]["research_feature_control"],
        )
    )
    restored_inputs = build_inputs(raw, environment=restored_environment)
    restored_model = make_model(checkpoint["args"], checkpoint["n_in"], 2)
    restored_model.load_state_dict(checkpoint["model"])
    restored_model.eval()

    event_stream = StreamingEventFeatures()
    research_stream = StreamingResearchFeatures(
        checkpoint["args"]["research_features"],
        control=checkpoint["args"]["research_feature_control"],
    )
    stream_rows = np.stack(
        [
            np.concatenate((row, event_stream.update(row), research_stream.update(row)))
            for row in rows
        ]
    )
    streaming_inputs = torch.from_numpy(stream_rows[None])
    with torch.inference_mode():
        full_prediction, _ = restored_model(restored_inputs)
        state = None
        chunks = []
        for start in range(0, len(rows), 5):
            prediction, state = restored_model(
                streaming_inputs[:, start : start + 5], state
            )
            chunks.append(prediction)
    torch.testing.assert_close(
        torch.cat(chunks, dim=1), full_prediction, rtol=3e-5, atol=3e-6
    )


def test_training_source_identity_binds_feature_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    identity = training_source_identity(TrainingEnvironment(source_root=root))
    assert "src/wnn/research_features.py" in identity["files"]
    assert "src/wnn/causal_research_features.py" in identity["files"]
