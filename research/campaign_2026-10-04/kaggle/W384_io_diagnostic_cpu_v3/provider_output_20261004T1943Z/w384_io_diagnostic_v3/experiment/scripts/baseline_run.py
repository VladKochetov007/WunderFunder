"""Wrap the exact organizer baseline (submissions/_parts/baseline_rnnmodel.npz) as a training run.

    .venv/bin/python scripts/baseline_run.py [SCALE]
Writes artifacts/runs/base_rnn/model.pt: the RNNModel rewrite of the ONNX baseline with its output
scale (default 0.25) folded into the last head layer, so fine-tunes (train_rnn.py --init base_rnn
--freeze inp,gru,head.0 --aux-struct 0) start from exactly the blended member.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from wnn.features import N_EVENT_FEATURES
from wnn.data import N_FEATURES

ROOT = Path(__file__).resolve().parents[1]


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    sd = dict(np.load(ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"))
    sd["head.2.weight"] = sd["head.2.weight"] * scale
    sd["head.2.bias"] = sd["head.2.bias"] * scale
    state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in sd.items()}
    args = json.loads((ROOT / "artifacts" / "logs" / "ab29_mask_f0.log").read_text().splitlines()[0])
    args.update(name="base_rnn", hidden=128, layers=2, dropout=0.0, arch="gru", cell="gru", aux_struct=0.0,
                aux_next=0.0, aux_next_k="", aux="", raw_only=True, loss="wmse", init="", base_scale=scale)
    out = ROOT / "artifacts" / "runs" / "base_rnn"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "args": args, "n_in": N_FEATURES + N_EVENT_FEATURES, "readout": None,
                "readout_tilted": False}, out / "model.pt")
    print("wrote", out / "model.pt", {k: tuple(v.shape) for k, v in state.items()})


if __name__ == "__main__":
    main()
