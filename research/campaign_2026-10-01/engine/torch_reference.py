"""Torch CPU reference outputs for native GRU/LSTM parity (run with the main .venv, read-only).

    CUDA_VISIBLE_DEVICES= .venv/bin/python research/campaign_2026-10-01/engine/torch_reference.py OUT_DIR

Writes OUT_DIR/{cell}_h{hidden}.npz: the float32 state dict (wnn.nn.RNNModel, default PyTorch
init, random normalization buffers), inputs X (T x 211: dense raw-like block, sparse event-like
block) and the model outputs in float64 (y64) and float32 (y32).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.nn import RNNModel  # noqa: E402

torch.set_num_threads(1)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
T, N_IN = 4000, 211
for cell, hidden, seed in [("gru", 128, 1), ("lstm", 128, 2), ("lstm", 256, 3)]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RNNModel(N_IN, hidden=hidden, layers=2, cell=cell).eval()
    mean = rng.normal(0, 1, N_IN).astype(np.float32)
    std = (0.5 + rng.gamma(2.0, 0.5, N_IN)).astype(np.float32)
    model.mean.copy_(torch.from_numpy(mean))
    model.std.copy_(torch.from_numpy(std))
    dense = mean[:112] + std[:112] * rng.standard_normal((T, 112))
    sparse = rng.standard_normal((T, 99)) * std[112:] * (rng.random((T, 99)) < 0.2)
    x = np.concatenate([dense, sparse], axis=1).astype(np.float32)
    with torch.no_grad():
        y32 = model(torch.from_numpy(x)[None])[0][0].numpy()
        y64 = model.double()(torch.from_numpy(x).double()[None])[0][0].numpy()
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    np.savez(out / f"{cell}_h{hidden}.npz", X=x, y64=y64, y32=y32, **{f"w/{k}": v for k, v in sd.items()})
    print(cell, hidden, y64.shape, float(np.abs(y64).mean()), float(np.abs(y32 - y64).max()))
