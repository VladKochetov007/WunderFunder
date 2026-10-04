"""Time one training step stage by stage (research throughput profiling; no duty cycle).

    scripts/limited .venv/bin/python scripts/bench_step.py [--batch 32] [--steps 3] [--enc linear]
Stages: parquet read per sequence (CPU), host->device copy, build_inputs, propensity forward,
TBPTT chunks forward+backward (with aux losses as in train_rnn), optimizer step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn as T  # noqa: E402
from wnn.data import N_FEATURES, SequenceFile  # noqa: E402
from wnn.features import N_EVENT_FEATURES  # noqa: E402
from wnn.nn import weighted_mse_loss  # noqa: E402
from wnn.propensity import load_propensity, propensity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--enc", default="linear")
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    sf = SequenceFile(ROOT / "datasets" / "train.parquet")
    t0 = time.perf_counter()
    seqs = [sf.read(g) for g in range(a.batch)]
    t_read = (time.perf_counter() - t0) / a.batch
    x = torch.from_numpy(np.stack([s.features for s in seqs]))
    y = torch.from_numpy(np.stack([s.targets for s in seqs]))
    if a.pin:
        x, y = x.pin_memory(), y.pin_memory()
    model = T.make_model({"hidden": 128, "layers": 2, "dropout": 0.0, "cell": "gru", "enc": a.enc}, N_FEATURES + N_EVENT_FEATURES, 21).to(dev)
    prop = load_propensity(ROOT / "artifacts" / "runs" / "prop_v1" / "model.pt", dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    res = {k: [] for k in ("h2d", "inputs", "prop", "chunks", "opt", "total")}
    for step in range(a.steps + 1):
        e = [ev() for _ in range(6)]
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        e[0].record()
        xd, yd = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        e[1].record()
        inputs = T.build_inputs(xd)
        e[2].record()
        w = propensity(prop, inputs[..., : N_FEATURES + N_EVENT_FEATURES]) ** 2 + 0.03
        e[3].record()
        h = None
        for s in range(0, inputs.shape[1], a.chunk):
            sl = slice(s, s + a.chunk)
            with torch.autocast("cuda", dtype=torch.float16):
                out, h = T.forward_chunk(model, inputs, sl, h)
            h = T.detach_state(h)
            loss = weighted_mse_loss(out[..., :2], yd[:, sl, :2], w[:, sl])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if s == inputs.shape[1] - a.chunk:
                e[4].record()
            scaler.step(opt)
            scaler.update()
        e[5].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - w0
        if step == 0:
            continue
        for k, (i, j) in zip(("h2d", "inputs", "prop", "chunks", "opt"), ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))):
            res[k].append(e[i].elapsed_time(e[j]))
        res["total"].append(wall * 1000)
    print(f"parquet read per sequence: {t_read*1000:.0f} ms  -> {1/t_read:.1f} seq/s per worker")
    for k, v in res.items():
        print(f"{k:7s} {np.mean(v):8.1f} ms")
    print(f"GPU-side step for batch {a.batch}: {np.mean(res['total']):.0f} ms -> {a.batch / np.mean(res['total']) * 1000:.1f} seq/s (no loader, no duty cycle)")


if __name__ == "__main__":
    main()
