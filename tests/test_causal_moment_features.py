from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn  # noqa: E402

from wnn.data import SequenceFile  # noqa: E402
from wnn.features import event_features_torch  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402
from wnn.research_features import (  # noqa: E402
    MOMENT_FEATURE_MODES,
    StreamingResearchFeatures,
    fit_causal_moment_prior,
    research_feature_control_width,
    research_feature_names,
    research_features_numpy,
    research_features_torch,
)


def _raw(rows: int = 220, seed: int = 17) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, 112)).astype(np.float32)


@pytest.mark.parametrize("mode", MOMENT_FEATURE_MODES)
def test_causal_moments_match_batch_stream_and_ignore_future(mode: str) -> None:
    fit_rows = _raw(256, 31)
    prior = fit_causal_moment_prior([fit_rows])
    rows = _raw()
    changed_future = rows.copy()
    changed_future[100:] += np.float32(5.0)

    want = research_features_numpy(rows, mode, prior)
    changed = research_features_numpy(changed_future, mode, prior)
    np.testing.assert_array_equal(want[:100], changed[:100])
    assert np.isfinite(want).all()

    batched = research_features_torch(torch.from_numpy(rows[None]), mode, prior)[0].numpy()
    np.testing.assert_allclose(batched, want, rtol=3e-4, atol=5e-5)

    stream = StreamingResearchFeatures(mode, prior)
    streamed = np.stack([stream.update(row) for row in rows])
    np.testing.assert_allclose(streamed, want, rtol=3e-4, atol=5e-5)
    assert streamed.shape[1] == len(research_feature_names(mode))
    # The transform consumes every row during the 99-row serving warm-up.
    np.testing.assert_allclose(streamed[98:100], want[98:100], rtol=3e-4, atol=5e-5)
    stream.reset()
    np.testing.assert_array_equal(stream.update(rows[0]), streamed[0])


@pytest.mark.parametrize(
    ("mode", "zero_width"),
    [("causal_moments", 64), ("causal_innovations", 16)],
)
def test_newest_control_keeps_width_and_matches_batch_stream(mode: str, zero_width: int) -> None:
    prior = fit_causal_moment_prior([_raw(256, 39)])
    rows = _raw(120, 40)
    candidate = research_features_numpy(rows, mode, prior)
    controlled = research_features_numpy(rows, mode, prior, control="newest")

    assert research_feature_control_width(mode, "newest") == zero_width
    assert controlled.shape == candidate.shape == (len(rows), len(research_feature_names(mode)))
    if zero_width < controlled.shape[-1]:
        np.testing.assert_array_equal(controlled[..., :-zero_width], candidate[..., :-zero_width])
    np.testing.assert_array_equal(controlled[..., -zero_width:], 0.0)

    torch_controlled = research_features_torch(
        torch.from_numpy(rows[None]), mode, prior, control="newest"
    )[0].numpy()
    np.testing.assert_allclose(torch_controlled, controlled, rtol=3e-4, atol=5e-5)
    stream = StreamingResearchFeatures(mode, prior, control="newest")
    streamed = np.stack([stream.update(row) for row in rows])
    np.testing.assert_allclose(streamed, controlled, rtol=3e-4, atol=5e-5)

    if mode == "causal_innovations":
        with pytest.raises(ValueError, match="requires a causal moment"):
            research_feature_control_width("none", "newest")


def test_research_feature_extension_zero_initialization_preserves_parent() -> None:
    parent = RNNModel(211, hidden=8, layers=1).eval()
    child = RNNModel(211 + 64, hidden=8, layers=1).eval()
    with torch.no_grad():
        parent.mean.normal_()
        parent.std.uniform_(0.5, 2.0)
    expanded = train_rnn.expand_research_feature_state(
        parent.state_dict(),
        child.state_dict(),
        "none",
        "causal_moments",
        211,
    )
    child.load_state_dict(expanded)
    assert train_rnn.can_extend_research_features("causal_moments", "causal_innovations")
    assert not train_rnn.can_extend_research_features("volume28", "causal_moments")

    base = torch.randn(2, 40, 211)
    moments = torch.randn(2, 40, 64)
    with torch.no_grad():
        parent_output, _ = parent(base)
        child_output, _ = child(torch.cat((base, moments), dim=-1))
    torch.testing.assert_close(child_output, parent_output, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", MOMENT_FEATURE_MODES)
def test_trainer_input_builder_appends_checkpointed_features(mode: str, monkeypatch) -> None:
    prior = fit_causal_moment_prior([_raw(256, 51)])
    raw = torch.from_numpy(_raw(140, 52)[None])
    monkeypatch.setattr(train_rnn, "RESEARCH_FEATURES", mode)
    monkeypatch.setattr(train_rnn, "RESEARCH_FEATURE_PRIOR", prior)
    monkeypatch.setattr(train_rnn, "ONLINE_NORM", False)
    monkeypatch.setattr(train_rnn, "HEURISTICS", False)
    monkeypatch.setattr(train_rnn, "RAW_ONLY", False)
    monkeypatch.setattr(train_rnn, "PRIVILEGED_MODE", "none")
    monkeypatch.setattr(train_rnn, "INPUT_PROPENSITY", None)
    monkeypatch.setattr(train_rnn, "SENTINEL_PREPROCESSOR", None)

    inputs = train_rnn.build_inputs(raw)
    expected = torch.cat(
        (
            raw,
            event_features_torch(raw),
            research_features_torch(raw, mode, prior),
        ),
        dim=-1,
    )
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)


def test_trainer_input_builder_applies_newest_matched_control(monkeypatch) -> None:
    mode = "causal_innovations"
    prior = fit_causal_moment_prior([_raw(256, 61)])
    raw = torch.from_numpy(_raw(140, 62)[None])
    monkeypatch.setattr(train_rnn, "RESEARCH_FEATURES", mode)
    monkeypatch.setattr(train_rnn, "RESEARCH_FEATURE_PRIOR", prior)
    monkeypatch.setattr(train_rnn, "RESEARCH_FEATURE_CONTROL", "newest")
    monkeypatch.setattr(train_rnn, "ONLINE_NORM", False)
    monkeypatch.setattr(train_rnn, "HEURISTICS", False)
    monkeypatch.setattr(train_rnn, "RAW_ONLY", False)
    monkeypatch.setattr(train_rnn, "PRIVILEGED_MODE", "none")
    monkeypatch.setattr(train_rnn, "INPUT_PROPENSITY", None)
    monkeypatch.setattr(train_rnn, "SENTINEL_PREPROCESSOR", None)

    inputs = train_rnn.build_inputs(raw)
    expected_features = research_features_torch(raw, mode, prior, control="newest")
    assert inputs.shape[-1] == 112 + 99 + 80
    torch.testing.assert_close(inputs[..., -16:], torch.zeros_like(inputs[..., -16:]))
    torch.testing.assert_close(inputs[..., -80:], expected_features, rtol=0, atol=0)


@pytest.mark.data
def test_full_real_sequence_scan_matches_streaming() -> None:
    """Exercise forty scan blocks and real nearly constant encoded inputs."""
    source = Path(__file__).resolve().parents[1] / "datasets/train.parquet"
    if not source.exists():
        pytest.skip("requires the supplied training data")
    reader = SequenceFile(source)
    prior = fit_causal_moment_prior([reader.read(0, targets=False, mask=False).features])
    raw = reader.read(1, targets=False, mask=False).features
    assert len(raw) == 20_000
    expected = research_features_numpy(raw, "causal_innovations", prior)
    actual = research_features_torch(torch.from_numpy(raw[None]), "causal_innovations", prior)[
        0
    ].numpy()
    stream = StreamingResearchFeatures("causal_innovations", prior)
    streamed = np.stack([stream.update(row) for row in raw])
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=5e-5)
    np.testing.assert_allclose(actual, streamed, rtol=3e-4, atol=5e-5)
