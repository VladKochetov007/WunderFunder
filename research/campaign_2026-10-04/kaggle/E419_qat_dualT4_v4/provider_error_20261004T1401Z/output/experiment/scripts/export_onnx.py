"""Export a trained RNN run as a one-row stateful ONNX graph and verify it.

    scripts/limited .venv/bin/python scripts/export_onnx.py RUN [--out submissions/RUN/model.onnx]
Checks ORT streaming output against batch torch predictions on a validation
sequence and times per-row inference with the baseline's single-thread session.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from wnn.data import SequenceFile
from wnn.features import event_features
from wnn.nn import RNNModel
from wnn.streaming import StreamingGraph, initial_state

ROOT = Path(__file__).resolve().parents[1]
NAMES_IN = ["row", "prev", "step", "last", "held", "h", "first"]
NAMES_OUT = ["pred", "prev_out", "step_out", "last_out", "held_out", "h_out", "first_out"]


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    ap.add_argument("--rows", type=int, default=20_000)
    args = ap.parse_args()
    ck = torch.load(ROOT / "artifacts" / "runs" / args.run / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    if a.get("arch", "gru") != "gru" or a.get("prop_input"):
        raise SystemExit("only plain RNNModel runs are exportable here")
    n_out = 2 + 2 * len([v for v in a.get("aux", "").split(",") if v])
    model = RNNModel(ck["n_in"], a["hidden"], a["layers"], 0.0, n_out=n_out).eval()
    model.load_state_dict(ck["model"])
    graph = StreamingGraph(model).eval()
    out = Path(args.out) if args.out else ROOT / "submissions" / args.run / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = initial_state(model)
    torch.onnx.export(graph, (torch.zeros(1, 112), *state), str(out), input_names=NAMES_IN,
                      output_names=NAMES_OUT, opset_version=17, dynamo=False)

    seq = SequenceFile(ROOT / "datasets" / "valid.parquet").read(0)
    x = seq.features[: args.rows]
    with torch.no_grad():
        want, _ = model(torch.cat([torch.from_numpy(x), torch.from_numpy(event_features(x))], -1)[None])
    want = want[0, :, :2].numpy()

    sess = session(out)
    feeds = {k: v.numpy() for k, v in zip(NAMES_IN[1:], state, strict=True)}
    got = np.empty((len(x), 2), np.float32)
    t0 = time.perf_counter()
    for i, row in enumerate(x):
        res = sess.run(None, {"row": row[None], **feeds})
        got[i] = res[0][0]
        feeds = dict(zip(NAMES_IN[1:], res[1:], strict=True))
    us = (time.perf_counter() - t0) / len(x) * 1e6
    err = float(np.abs(got - want).max())
    print(json.dumps({"run": args.run, "hidden": a["hidden"], "layers": a["layers"], "onnx": str(out),
                      "bytes": out.stat().st_size, "max_abs_err_vs_batch": err, "us_per_row": round(us, 2),
                      "est_minutes_valid_size": round(us * 37.46e6 / 60e6, 1)}))
    if err > 1e-3:
        raise SystemExit("streaming ONNX disagrees with batch model")


if __name__ == "__main__":
    main()
