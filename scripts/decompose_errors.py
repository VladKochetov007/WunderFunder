"""H-024: where is WP lost? Per-row Pearson gap = ideal contribution - actual contribution.

For a block and target: c_i = w_i (y_i - ybar)(p_i - pbar) / (W sy sp), sum c_i = rho;
ideal_i = w_i (y_i - ybar)^2 / (W sy^2), sum = 1. Gap sums to 1 - rho. Gaps are averaged
over blocks and targets and sliced by class, rows to next i0 price event and |y|.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from train_rnn import build_inputs, forward_chunk, make_model  # noqa: E402

from wnn.data import SequenceFile, market_class  # noqa: E402
from wnn.splits import valid_row_groups  # noqa: E402

REM_BINS = [1, 2, 6, 21, 101, 1001, 20001]
Y_BINS = [0, 0.5, 1.0, 2.0, 99]


def main():
    run, n = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(4)
    ck = torch.load(f"artifacts/runs/{run}/model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    model = make_model(a, ck["n_in"], 2 + 2 * len([v for v in a.get("aux", "").split(",") if v])).eval()
    model.load_state_dict(ck["model"])
    dev, _ = valid_row_groups("datasets/valid.parquet")
    v = SequenceFile("datasets/valid.parquet")
    groups = np.random.default_rng(3).permutation(dev)[:n]
    table = {}
    totals = {}
    for g in groups:
        s = v.read(int(g))
        c = market_class(s.targets)
        x = torch.from_numpy(s.features)[None]
        with torch.no_grad():
            inp = build_inputs(x)
            state, outs = None, []
            for st in range(0, 20000, 4000):
                o, state = forward_chunk(model, inp, slice(st, st + 4000), state)
                outs.append(o[..., :2])
            p = torch.cat(outs, 1)[0].numpy().astype(np.float64)
        xs = s.features
        ev = np.r_[True, (np.diff(xs[:, 0]) != 0) & (np.diff(xs[:, 11]) != 0)]
        idx = np.flatnonzero(ev)
        nxt = np.searchsorted(idx, np.arange(20000), side="right")
        rem = np.where(nxt < len(idx), idx[np.minimum(nxt, len(idx) - 1)] - np.arange(20000), 20000)
        m = s.scored & (np.arange(20000) >= 99)
        for k in range(2):
            y = np.clip(s.targets[:, k].astype(np.float64), -2, 2)
            pk = np.clip(p[:, k], -2, 2)
            if not ((y[m] > 0).any() and (y[m] < 0).any()):
                continue
            w = np.abs(y) * m
            W = w.sum()
            yb, pb = (w * y).sum() / W, (w * pk).sum() / W
            sy = np.sqrt((w * (y - yb) ** 2).sum() / W)
            sp = np.sqrt((w * (pk - pb) ** 2).sum() / W)
            if sy <= 1e-8 or sp <= 1e-8:
                continue
            contrib = w * (y - yb) * (pk - pb) / (W * sy * sp)
            ideal = w * (y - yb) ** 2 / (W * sy * sy)
            rb = np.digitize(rem, REM_BINS) - 1
            ybk = np.digitize(np.abs(y), Y_BINS) - 1
            for rbin in range(len(REM_BINS) - 1):
                for ybin in range(len(Y_BINS) - 1):
                    sel = m & (rb == rbin) & (ybk == ybin)
                    key = (c, rbin, ybin)
                    t = table.setdefault(key, [0.0, 0.0])
                    t[0] += ideal[sel].sum()
                    t[1] += contrib[sel].sum()
            tot = totals.setdefault(c, [0.0, 0])
            tot[0] += 1 - contrib.sum()
            tot[1] += 1
    rem_lab = [f"{REM_BINS[i]}-{REM_BINS[i + 1] - 1}" for i in range(len(REM_BINS) - 1)]
    y_lab = [f"|y|{Y_BINS[i]}-{Y_BINS[i + 1]}" for i in range(len(Y_BINS) - 1)]
    out = {}
    for c in sorted(totals):
        nblk = totals[c][1]
        rows = []
        for (cc, rbin, ybin), (ideal, got) in table.items():
            if cc != c:
                continue
            rows.append({"rows_to_break": rem_lab[rbin], "absy": y_lab[ybin], "ideal": round(ideal / nblk, 4),
                         "captured": round(got / nblk, 4), "gap": round((ideal - got) / nblk, 4)})
        rows.sort(key=lambda r: -r["gap"])
        out[c] = {"mean_wp": round(1 - totals[c][0] / nblk, 4), "target_blocks": nblk, "top_gaps": rows[:6]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
