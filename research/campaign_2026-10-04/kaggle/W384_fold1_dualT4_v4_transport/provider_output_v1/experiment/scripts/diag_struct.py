"""Accuracy of auxiliary structure heads (spread ticks, branch, class) on scored devB rows.

    scripts/limited .venv/bin/python scripts/diag_struct.py RUN [n_seqs]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_rnn import Sequences, build_inputs, forward_chunk, make_model

from wnn.outcomes import N_SP_AUX, sequence_class, structure_labels
from wnn.splits import dev_halves

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ck = torch.load(ROOT / "artifacts" / "runs" / name / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    head = ck["model"]["head.2.weight"].shape[0]
    at = head - (N_SP_AUX + 7)
    device = torch.device("cuda")
    model = make_model(a, ck["n_in"], head).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    groups = np.random.default_rng(12345).permutation(dev_halves(ROOT / "datasets" / "valid.parquet")[1])[:n]
    ds = Sequences("valid", groups)
    rows = {k: [] for k in ("cls", "sp", "sp_hat", "br", "br_p", "cl_hat")}
    for i in range(len(ds)):
        x, y, m = ds[i]
        x, y = x[None].to(device), y[None].to(device)
        inputs = build_inputs(x)
        state, outs = None, []
        for s in range(0, inputs.shape[1], 4000):
            with torch.autocast("cuda", dtype=torch.float16):
                o, state = forward_chunk(model, inputs, slice(s, s + 4000), state)
            outs.append(o.float())
        o = torch.cat(outs, 1)[0]
        cls = sequence_class(y)
        sp, br, _ = structure_labels(y, cls)
        sel = m.numpy().astype(bool)
        rows["cls"].append(np.full(sel.sum(), int(cls[0])))
        rows["sp"].append(sp[0].cpu().numpy()[sel])
        rows["sp_hat"].append(o[:, at:at + N_SP_AUX].argmax(-1).cpu().numpy()[sel])
        rows["br"].append(br[0].cpu().numpy()[sel])
        rows["br_p"].append(torch.softmax(o[:, at + N_SP_AUX:at + N_SP_AUX + 3], -1).cpu().numpy()[sel])
        rows["cl_hat"].append(o[:, at + N_SP_AUX + 3:at + N_SP_AUX + 7].argmax(-1).cpu().numpy()[sel])
    r = {k: np.concatenate(v) for k, v in rows.items()}
    out = {}
    for k in range(4):
        s = r["cls"] == k
        out[k] = {"sp_acc": round(float((r["sp_hat"][s] == r["sp"][s]).mean()), 4),
                  "class_acc": round(float((r["cl_hat"][s] == k).mean()), 4),
                  "auc_above": round(float(roc_auc_score(r["br"][s] == 2, r["br_p"][s][:, 2])), 4),
                  "auc_below": round(float(roc_auc_score(r["br"][s] == 0, r["br_p"][s][:, 0])), 4)}
    print(json.dumps({"run": name, "seqs": n, "by_class": out}))


if __name__ == "__main__":
    main()
