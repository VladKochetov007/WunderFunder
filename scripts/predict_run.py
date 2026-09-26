"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import train_rnn
from torch.utils.data import DataLoader
from train_rnn import Sequences, make_model, predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    name = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "valid"
    ck = torch.load(ROOT / "artifacts" / "runs" / name / "model.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    loader = DataLoader(Sequences(split, groups), batch_size=16, num_workers=2, prefetch_factor=1)
    out = np.lib.format.open_memmap(ROOT / "cache" / f"pred_{name}_{split}.npy", mode="w+", dtype=np.float32,
                                    shape=(len(groups), 20_000, 2))
    i = 0
    for x, _, _ in loader:
        p = predict(model, x, 4000, device)
        out[i:i + len(p)] = p[..., :2]
        i += len(p)
    out.flush()


if __name__ == "__main__":
    main()
