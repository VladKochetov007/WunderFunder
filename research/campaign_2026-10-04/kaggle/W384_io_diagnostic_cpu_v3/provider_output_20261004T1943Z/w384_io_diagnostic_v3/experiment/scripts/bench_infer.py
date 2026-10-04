"""Per-row inference latency on one core for candidate runtimes.

    WNN_CPU_PCT=7 scripts/limited .venv/bin/python scripts/bench_infer.py --hidden 256
Pin with taskset for stable numbers; the scorer gives one vCPU.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wnn.data import N_FEATURES  # noqa: E402
from wnn.features import N_EVENT_FEATURES, StreamingEventFeatures  # noqa: E402
from wnn.infer import NumpyRNN, export_weights  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def per_row(fn, rows, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for r in rows:
            fn(r)
        best = min(best, (time.perf_counter() - t) / len(rows))
    return best * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    n_in = N_FEATURES + N_EVENT_FEATURES
    model = RNNModel(n_in, args.hidden, args.layers).eval()
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(args.rows, N_FEATURES)).astype(np.float32)
    x = rng.normal(size=(args.rows, n_in)).astype(np.float32)
    results = {}

    feats = StreamingEventFeatures()
    results["event features, numpy stream"] = per_row(feats.update, raw)

    rnn = NumpyRNN(export_weights(model))
    results["GRU numpy"] = per_row(rnn.step, x)

    h = [None]

    @torch.inference_mode()
    def torch_step(r):
        out, h[0] = model(torch.from_numpy(r).view(1, 1, -1), h[0])
        return out

    results["GRU torch eager"] = per_row(torch_step, x)

    try:
        import onnxruntime as ort

        path = os.path.join(tempfile.mkdtemp(), "m.onnx")
        h0 = torch.zeros(args.layers, 1, args.hidden)
        torch.onnx.export(
            model,
            (torch.zeros(1, 1, n_in), h0),
            path,
            input_names=["x", "h"],
            output_names=["y", "h_out"],
            dynamo=False,
            opset_version=17,
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        state = [np.zeros((args.layers, 1, args.hidden), np.float32)]

        def ort_step(r):
            y, state[0] = sess.run(None, {"x": r.reshape(1, 1, -1), "h": state[0]})
            return y

        results["GRU onnxruntime"] = per_row(ort_step, x)
    except Exception as e:  # noqa: BLE001
        print("onnx skipped:", type(e).__name__, e)

    try:
        from wnn_native import GruStepper  # built by native/ (maturin)

        w = export_weights(model)
        nat = GruStepper({k: np.ascontiguousarray(v, np.float32) for k, v in w.items()})
        results["GRU rust native"] = per_row(nat.step, x)
        np.testing.assert_allclose(nat.step(x[0]), rnn.step(x[0]), atol=1e-3)
    except ImportError:
        pass

    budget = 60 * 60 / 37.46e6 * 1e6
    print(
        f"hidden {args.hidden} layers {args.layers} inputs {n_in}; budget ~{budget:.0f} us/row total"
    )
    for k, v in results.items():
        print(f"  {k:32s} {v:8.2f} us/row")


if __name__ == "__main__":
    main()
