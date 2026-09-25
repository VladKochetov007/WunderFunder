"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN
Writes cache/pred_<RUN>_valid.npy (B, T, 2) float32.
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
    ck = torch.load(ROOT / "artifacts" / "runs" / name / "model.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    groups = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    loader = DataLoader(Sequences("valid", groups), batch_size=16, num_workers=2, prefetch_factor=1)
    out = np.empty((len(groups), 20_000, 2), np.float32)
    i = 0
    for x, _, _ in loader:
        p = predict(model, x, 4000, device)
        out[i:i + len(p)] = p[..., :2]
        i += len(p)
    np.save(ROOT / "cache" / f"pred_{name}_valid.npy", out)


if __name__ == "__main__":
    main()
