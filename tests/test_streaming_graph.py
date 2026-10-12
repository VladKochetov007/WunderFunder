from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wnn.features import event_features  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402
from wnn.research_features import (  # noqa: E402
    fit_causal_moment_prior,
    research_features_torch,
)
from wnn.streaming import StreamingGraph, initial_state  # noqa: E402


def _rows(t=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, 112)).astype(np.float32)
    hold = rng.random((t, 112)) < 0.7
    hold[0] = False
    for i in range(1, t):
        x[i, hold[i]] = x[i - 1, hold[i]]
    joint = rng.random(t) < 0.3
    for i in range(1, t):
        if not joint[i]:
            x[i, 0:22] = x[i - 1, 0:22]
    return x


def _model(hidden=24, layers=2):
    torch.manual_seed(0)
    m = RNNModel(112 + 99, hidden, layers).eval()
    with torch.no_grad():
        m.mean.normal_()
        m.std.uniform_(0.5, 2.0)
    return m


def test_row_stepping_matches_batch_features_and_model():
    x = _rows()
    model = _model()
    graph = StreamingGraph(model).eval()
    with torch.no_grad():
        feats = torch.from_numpy(event_features(x))
        want, _ = model(torch.cat([torch.from_numpy(x), feats], -1)[None])
        state = initial_state(model)
        got = []
        for row in x:
            y, *state = graph(torch.from_numpy(row)[None], *state)
            got.append(y[0])
    np.testing.assert_allclose(torch.stack(got).numpy(), want[0].numpy(), atol=2e-5)


def test_reset_state_reproduces_first_output():
    x = _rows(10)
    graph = StreamingGraph(_model()).eval()
    with torch.no_grad():
        state = initial_state(graph.model)
        first, *state = graph(torch.from_numpy(x[0])[None], *state)
        for row in x[1:]:
            _, *state = graph(torch.from_numpy(row)[None], *state)
        again, *_ = graph(torch.from_numpy(x[0])[None], *initial_state(graph.model))
    torch.testing.assert_close(first, again)


@pytest.mark.parametrize("control", ["none", "newest"])
def test_causal_innovation_graph_matches_batch_and_reset(control):
    x = _rows(140, seed=11)
    prior = fit_causal_moment_prior([_rows(256, seed=12)])
    model = RNNModel(112 + 99 + 80, hidden=12, layers=1).eval()
    graph = StreamingGraph(model, "causal_innovations", prior, control).eval()
    with torch.no_grad():
        expected_inputs = torch.cat(
            (
                torch.from_numpy(x),
                torch.from_numpy(event_features(x)),
                research_features_torch(
                    torch.from_numpy(x)[None], "causal_innovations", prior, control=control
                )[0],
            ),
            dim=-1,
        )
        want, _ = model(expected_inputs[None])
        state = initial_state(model, "causal_innovations", prior, control)
        got = []
        for row in x:
            output, *state = graph(torch.from_numpy(row)[None], *state)
            got.append(output[0])
        torch.testing.assert_close(torch.stack(got), want[0, :, :2], rtol=3e-4, atol=5e-5)

        first, *_ = graph(
            torch.from_numpy(x[0])[None],
            *initial_state(model, "causal_innovations", prior, control),
        )
        torch.testing.assert_close(first[0], want[0, 0, :2], rtol=3e-4, atol=5e-5)


def test_causal_innovation_onnx_state_inputs_round_trip(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    x = _rows(12, seed=23)
    prior = fit_causal_moment_prior([_rows(256, seed=24)])
    model = RNNModel(112 + 99 + 80, hidden=8, layers=1).eval()
    control = "newest"
    graph = StreamingGraph(model, "causal_innovations", prior, control).eval()
    state = initial_state(model, "causal_innovations", prior, control)
    input_names = ["row", "prev", "step", "last", "held", "h", "first", "moment_state"]
    output_names = [
        "pred",
        "prev_out",
        "step_out",
        "last_out",
        "held_out",
        "h_out",
        "first_out",
        "moment_state_out",
    ]
    path = tmp_path / "causal-stream.onnx"
    torch.onnx.export(
        graph,
        (torch.zeros(1, 112), *state),
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {name: value.numpy() for name, value in zip(input_names[1:], state, strict=True)}
    with torch.no_grad():
        for row in x:
            expected, *state = graph(torch.from_numpy(row)[None], *state)
            result = session.run(None, {"row": row[None], **feeds})
            np.testing.assert_allclose(result[0][0], expected[0].numpy(), rtol=1e-5, atol=1e-6)
            feeds = dict(zip(input_names[1:], result[1:], strict=True))
