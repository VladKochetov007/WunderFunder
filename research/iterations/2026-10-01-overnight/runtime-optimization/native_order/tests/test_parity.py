"""Parity of the C++ engine with the PyTorch model it replaces.

    PYTHONPATH=src:native_cpp/build/v3 .venv/bin/python native_cpp/tests/test_parity.py RUN [--seqs 3]

RUN is a training run in artifacts/runs (model.pt). For a few validation sequences the torch
reference is RNNModel in float32 eval mode on [raw | event_features_torch(raw)], full sequence at once;
the C++ side streams the same rows one at a time. Reports max abs difference and output correlation,
and fails if the difference exceeds --tol.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from wnn.data import SequenceFile  # noqa: E402
from wnn.features import event_features_torch, online_zscore_torch  # noqa: E402


def torch_reference(ck: dict, x: np.ndarray) -> np.ndarray:
    from train_rnn import make_model

    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0])
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        raw = torch.from_numpy(x)[None]
        inputs = torch.cat([raw, event_features_torch(raw)], dim=-1)
        if ck["args"].get("online_norm"):
            # the z-scores exactly as training computed them: float32 cumsum on CUDA
            z = online_zscore_torch(raw.cuda()).cpu() if torch.cuda.is_available() else online_zscore_torch(raw)
            inputs = torch.cat([inputs, z], dim=-1)
        out, _ = model(inputs)
    return out[0, :, :2].numpy()


def main() -> None:
    import wnn_cpp

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--exact", action="store_true", help="libm activations instead of the fast forms")
    ap.add_argument("--half", action="store_true", help="fp16 weight storage (v3 builds)")
    args = ap.parse_args()
    ck = torch.load(ROOT / "artifacts" / "runs" / args.run / "model.pt", map_location="cpu", weights_only=False)
    weights = {k: v.float().numpy() for k, v in ck["model"].items()}
    engine = wnn_cpp.Predictor(weights, fast_activations=not args.exact, half_weights=args.half)
    print("half weights active:", engine.half_weights)
    f = SequenceFile(ROOT / "datasets" / "valid.parquet")
    worst = 0.0
    for g in np.linspace(0, len(f) - 1, args.seqs).astype(int):
        x = f.read(int(g)).features
        ref = torch_reference(ck, x)
        t0 = time.perf_counter()
        got = engine.run(x)
        us = (time.perf_counter() - t0) / len(x) * 1e6
        diff = float(np.abs(ref - got).max())
        corr = float(np.corrcoef(ref[99:, 0], got[99:, 0])[0, 1])
        worst = max(worst, diff)
        print(f"seq {g}: max abs diff {diff:.2e}, corr t0 {corr:.8f}, {us:.1f} us/row")
    print("PASS" if worst <= args.tol else "FAIL", f"worst {worst:.2e} tol {args.tol:.0e}")
    sys.exit(0 if worst <= args.tol else 1)


if __name__ == "__main__":
    main()
