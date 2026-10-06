"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import train_rnn
from torch.utils.data import DataLoader
from train_rnn import Sequences, make_model, predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--gate-mode", choices=["soft", "hard", "confident"], default="soft")
    args = parser.parse_args()
    name, split = args.name, args.split
    checkpoint_name = f"model_ep{args.epoch}.pt" if args.epoch is not None else "model.pt"
    ck = torch.load(
        ROOT / "artifacts" / "runs" / name / checkpoint_name, map_location="cpu", weights_only=False
    )
    device = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    if ck["args"].get("arch") == "gated":
        model.gate_mode = args.gate_mode
    elif args.gate_mode != "soft":
        parser.error("--gate-mode requires a gated checkpoint")
    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))
    if ck["args"].get("zero_groups"):
        train_rnn.ZERO_COLS = torch.tensor(train_rnn.zero_group_cols(ck["args"]["zero_groups"]), dtype=torch.long)
    train_rnn.A_MODE = ck["args"].get("a_mode") or "none"
    train_rnn.DOMAIN_FLAG = bool(ck["args"].get("domain_flag"))
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    train_rnn.STATE_RESET = int(ck["args"].get("state_reset") or 0)
    train_rnn.LOOKAHEAD = tuple(int(v) for v in str(ck["args"].get("lookahead") or "").split(",") if v)
    train_rnn.QUANT = bool(ck["args"].get("quant_feats"))
    train_rnn.QUANT_I1 = bool(ck["args"].get("quant_i1"))
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    loader = DataLoader(Sequences(split, groups), batch_size=16, num_workers=2, prefetch_factor=1)
    final = ROOT / "cache" / f"pred_{args.output_name or name}_{split}.npy"
    tmp = final.with_name(final.name + ".partial")  # renamed only after every sequence is written
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(groups), 20_000, 2)
    )
    i = 0
    for x, _, _ in loader:
        p = predict(model, x, 4000, device)
        out[i : i + len(p)] = p[..., :2]
        i += len(p)
    out.flush()
    del out
    if i != len(groups):
        raise RuntimeError(f"wrote {i} of {len(groups)} sequences")
    stale = (
        ROOT / "cache" / f"seqsums_{args.output_name or name}.npz"
    )  # cached metric sums of an older prediction file
    if stale.exists():
        stale.unlink()
    tmp.replace(final)


if __name__ == "__main__":
    main()
