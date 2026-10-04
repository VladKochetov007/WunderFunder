from __future__ import annotations

import numpy as np
import pytest

from wnn.research_features import (
    ENDPOINT,
    RESEARCH_FEATURE_MODES,
    StreamingResearchFeatures,
    research_feature_names,
    research_features_numpy,
    research_features_torch,
)


def fixture(rows: int = 120) -> np.ndarray:
    """Include consecutive resets, unchanged volumes and a long quiet suffix."""
    rng = np.random.default_rng(301)
    raw = rng.normal(size=(rows, 112)).astype(np.float32)
    raw[:, 110] = 1
    raw[[0, 1, 17, 18, 63], 110] = ENDPOINT
    raw[80:, 22:44] = raw[79, 22:44]
    return raw


@pytest.mark.parametrize("mode", RESEARCH_FEATURE_MODES)
def test_batch_stream_reset_prefix_and_torch(mode: str) -> None:
    import torch

    raw = fixture()
    expected = research_features_numpy(raw, mode)
    assert expected.shape == (len(raw), len(research_feature_names(mode)))
    stream = StreamingResearchFeatures(mode)
    actual = np.stack([stream.update(row) for row in raw])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-6)
    stream.reset()
    np.testing.assert_array_equal(stream.update(raw[0]), expected[0])
    tensor = torch.from_numpy(np.stack((raw, raw)))
    np.testing.assert_allclose(
        research_features_torch(tensor, mode).numpy()[0], expected, atol=2e-5, rtol=2e-6
    )
    changed = raw.copy()
    changed[64:] *= 20
    np.testing.assert_array_equal(research_features_numpy(changed, mode)[:64], expected[:64])
    np.testing.assert_array_equal(research_features_numpy(raw[:64], mode), expected[:64])


def test_run_start_excluded_and_origin_latched() -> None:
    raw = np.zeros((4, 112), dtype=np.float32)
    raw[:, 110] = [0, 0, ENDPOINT, 0]
    raw[:, 22] = [2, 5, 50, 48]
    result = research_features_numpy(raw, "volume28")
    np.testing.assert_array_equal(result[:, 0], [0, 3, 0, -2])
    np.testing.assert_array_equal(result[:, 22], [0, 3, 0, 2])
    np.testing.assert_array_equal(result[:, 24], [0, 1, 0, 1])
    np.testing.assert_array_equal(result[:, 26], [0, 3, 0, 2])


def test_model_batch_streaming_predictions() -> None:
    import torch

    from wnn.features import StreamingEventFeatures, event_features_torch
    from wnn.nn import RNNModel

    torch.manual_seed(17)
    torch.set_num_threads(1)
    raw = fixture()
    model = RNNModel(239, hidden=16, layers=2).eval()
    tensor = torch.from_numpy(raw[None])
    inputs = torch.cat(
        (tensor, event_features_torch(tensor), research_features_torch(tensor, "volume28")), -1
    )
    state = None
    event, volume = StreamingEventFeatures(), StreamingResearchFeatures("volume28")
    predictions = []
    with torch.no_grad():
        expected, _ = model(inputs)
        for row in raw:
            features = np.concatenate((row, event.update(row), volume.update(row)))
            output, state = model(torch.from_numpy(features[None, None]), state)
            predictions.append(output[0, 0].numpy())
    np.testing.assert_allclose(np.stack(predictions), expected[0].numpy(), atol=2e-6, rtol=2e-5)


def test_rejects_autocast_endpoint_source() -> None:
    import torch

    with pytest.raises(ValueError, match="float32"):
        research_features_torch(torch.from_numpy(fixture()[None]).half(), "volume28")
