"""Streaming parity and per-row latency of the native Predictor against the PyTorch model.

    taskset -c 3 scripts/limited .venv/bin/python scripts/parity_native.py RUN [RUN ...] --native DIR
For each run: exports weights as package_submission.py does, streams validation row groups (--seqs)
row by row through wnn_native.Predictor (online_norm from the checkpoint args), and compares on
rows >= 99 with the model forward over the whole sequence on the training-time inputs
(train_rnn.build_inputs). Online z-scores are taken as the CUDA float32 run computes them
(wnn.features.online_zscore; --zdevice cuda computes build_inputs on the GPU instead); CPU torch
accumulates the cumsum in double, which is what "exact z" reports. Then times Predictor.step over
one sequence, runs interleaved, best of --reps. DIR holds wnn_native.abi3.so (e.g. pkg/v3).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_submission  # noqa: E402
import train_rnn  # noqa: E402

from wnn.data import N_FEATURES, WARMUP, SequenceFile  # noqa: E402
from wnn.features import N_EVENT_FEATURES, online_zscore  # noqa: E402

Z = slice(N_FEATURES + N_EVENT_FEATURES, 2 * N_FEATURES + N_EVENT_FEATURES)


def model_inputs(ck, x, z="train", zdevice="cpu"):
    """build_inputs on one sequence; z: 'train' (CUDA float32) or 'exact' (CPU torch, double cumsum)."""
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    if z == "train" and zdevice != "cpu":
        return train_rnn.build_inputs(torch.from_numpy(x)[None].to(zdevice)).cpu()[0]
    inputs = train_rnn.build_inputs(torch.from_numpy(x)[None])[0]
    if train_rnn.ONLINE_NORM and z == "train":
        inputs[:, Z] = torch.from_numpy(online_zscore(x))
    return inputs


def forward(ck, inputs, dtype):
    model = train_rnn.make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0])
    model.load_state_dict(ck["model"])
    model = model.to(dtype).eval()
    with torch.no_grad():
        out, _ = model(inputs.to(dtype)[None])
    return out[0, :, :2].double().numpy()


def stream(pred, x):
    pred.reset()
    out = np.empty((len(x), 2), np.float32)
    t = time.perf_counter()
    for i in range(len(x)):
        out[i] = pred.step(np.ascontiguousarray(x[i]))
    return out, (time.perf_counter() - t) / len(x) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--native", required=True, help="directory with the wnn_native extension")
    ap.add_argument("--seqs", default="0,1,2", help="valid.parquet row groups")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1, help="torch threads for the reference forward")
    ap.add_argument("--zdevice", default="cpu", help="'cuda' computes the training z-scores on the GPU")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(args.native).resolve()))
    import wnn_native

    print("native", wnn_native.__file__, "capabilities", getattr(wnn_native, "CAPABILITIES", ()))
    seqs = [int(s) for s in args.seqs.split(",")]
    sf = SequenceFile(ROOT / "datasets" / "valid.parquet")
    xs = {g: sf.read(g, targets=False, mask=False).features for g in seqs}
    tmp = Path(tempfile.mkdtemp())
    preds = {}
    for run in args.runs:
        ck = torch.load(ROOT / "artifacts" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
        spec = package_submission.export(run, tmp / f"{run}.npz")
        with np.load(tmp / f"{run}.npz") as w:
            weights = {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}
        kw = {"online_norm": True} if spec.get("online_norm") else {}
        preds[run] = wnn_native.Predictor(weights, fast_activations=True, **kw)
        exact_act = wnn_native.Predictor(weights, fast_activations=False, **kw)
        print(f"{run}: n_in {ck['n_in']} spec {spec}")
        for g, x in xs.items():
            train = model_inputs(ck, x, "train", args.zdevice)
            ref32, ref64 = forward(ck, train, torch.float32), forward(ck, train, torch.float64)
            ref_exact_z = forward(ck, model_inputs(ck, x, "exact"), torch.float32)
            got, us = stream(preds[run], x)
            got_exact, _ = stream(exact_act, x)
            r = slice(WARMUP, None)
            d = np.abs(got[r] - ref32[r])
            print(f"  seq {g}: max|native - torch32| {d.max():.3e} (mean {d.mean():.2e})"
                  f" | exact-activation native vs torch64 {np.abs(got_exact[r] - ref64[r]).max():.3e}"
                  f" | torch32 exact-z vs training-z {np.abs(ref_exact_z[r] - ref32[r]).max():.3e}"
                  f" | corr {np.corrcoef(got[r, 0], ref32[r, 0])[0, 1]:.7f} | {us:.1f} us/row")
    x = xs[seqs[0]]
    best = {run: float("inf") for run in args.runs}
    for _ in range(args.reps):
        for run in args.runs:
            best[run] = min(best[run], stream(preds[run], x)[1])
    print("latency, best of", args.reps, "passes over", len(x), "rows:",
          {run: f"{us:.2f} us/row" for run, us in best.items()})


if __name__ == "__main__":
    main()
