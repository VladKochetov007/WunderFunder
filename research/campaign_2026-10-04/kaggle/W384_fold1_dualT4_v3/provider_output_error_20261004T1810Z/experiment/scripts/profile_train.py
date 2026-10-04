"""Break down one training batch: parquet decode, host->device, feature build, fwd/bwd.

scripts/limited .venv/bin/python scripts/profile_train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from wnn.data import N_FEATURES, WARMUP, SequenceFile
from wnn.features import N_EVENT_FEATURES, event_features_torch
from wnn.nn import RNNModel, weighted_corr_loss, weighted_mse_loss
from wnn.resources import configure_torch

ROOT = Path(__file__).resolve().parents[1]


def timed(fn, reps=3, sync=True):
    out = None
    best = float("inf")
    for _ in range(reps):
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if sync:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    configure_torch()
    batch, chunk = 32, 2000
    f = SequenceFile(ROOT / "datasets" / "train.parquet")
    t_read, seqs = timed(lambda: [f.read(i) for i in range(batch)], reps=2, sync=False)
    x = torch.from_numpy(np.stack([s.features for s in seqs]))
    y = torch.from_numpy(np.stack([s.targets for s in seqs]))
    t_pin, xp = timed(lambda: x.pin_memory(), sync=False)
    dev = torch.device("cuda")
    t_h2d, xg = timed(lambda: xp.to(dev, non_blocking=True))
    yg = y.to(dev)
    t_feat, inputs = timed(lambda: torch.cat([xg, event_features_torch(xg)], -1))
    model = RNNModel(N_FEATURES + N_EVENT_FEATURES).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    scaler = torch.amp.GradScaler()
    valid = torch.arange(20_000, device=dev)[None].expand(batch, -1) >= WARMUP

    def train_batch():
        h = None
        for s in range(0, 20_000, chunk):
            sl = slice(s, s + chunk)
            with torch.autocast("cuda", dtype=torch.float16):
                out, h = model(inputs[:, sl], h)
            h = h.detach()
            loss = weighted_corr_loss(out, yg[:, sl], valid[:, sl]) + 0.5 * weighted_mse_loss(
                out, yg[:, sl], valid[:, sl]
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

    timed(train_batch, reps=1)  # cudnn autotune warmup
    t_step, _ = timed(train_batch, reps=2)
    rows = {
        "parquet decode (1 core, 32 seqs)": t_read,
        "pin_memory": t_pin,
        "host->device": t_h2d,
        "event features (GPU)": t_feat,
        "fwd+bwd+opt, 10 chunks (GPU)": t_step,
    }
    for k, v in rows.items():
        print(f"{k:36s} {v * 1000:9.1f} ms")
    print(
        f"GPU-bound ceiling {batch / (t_feat + t_step + t_h2d):.1f} seq/s; "
        f"decode needs {t_read / (t_feat + t_step + t_h2d):.1f} workers to keep up"
    )


if __name__ == "__main__":
    main()
