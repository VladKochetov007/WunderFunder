"""Online market-class detector from a GRU aux-struct class head, evaluated on held-out pure validation sequences.

    .venv/bin/python scripts/class_detect.py RUN FOLD [--rows 99] [--from-row 50]

Class probabilities (softmax of outputs [17, 21)) are averaged over rows [from_row, rows) of each sequence; the
decision is argmax. Prints the confusion matrix against market-classes-v1 and the class-3 precision / recall.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import train_rnn
from fold_eval import ROOT
from torch.utils.data import DataLoader
from train_rnn import Sequences, build_inputs, forward_chunk, make_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("fold", type=int)
    ap.add_argument("--rows", type=int, default=99)
    ap.add_argument("--from-row", type=int, default=50)
    a = ap.parse_args()
    run, _, ep = a.run.partition("@ep")
    ck = torch.load(ROOT / "artifacts" / "runs" / run / (f"model_ep{ep}.pt" if ep else "model.pt"),
                    map_location="cpu", weights_only=False)
    dev = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(dev).eval()
    model.load_state_dict(ck["model"])
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    groups = np.array([g for g in rg if info[int(g)] == (a.fold, True)])
    loader = DataLoader(Sequences("valid", groups), batch_size=32, num_workers=2)
    conf = np.zeros((4, 4), int)
    i = 0
    with torch.no_grad():
        for x, _, _ in loader:
            inp = build_inputs(x.to(dev))
            out, _ = forward_chunk(model, inp, slice(0, a.rows), None)
            p = torch.softmax(out[..., 17:21].float(), -1)[:, a.from_row:a.rows].mean(1).argmax(-1).cpu().numpy()
            for j, c in enumerate(p):
                conf[cls[groups[i + j]], c] += 1
            i += len(p)
    print(conf)
    print(f"class-3 precision {conf[3, 3] / conf[:, 3].sum():.4f} recall {conf[3, 3] / conf[3].sum():.4f}, "
          f"accuracy {np.trace(conf) / conf.sum():.4f}")


if __name__ == "__main__":
    main()
