"""Card H-L4: is the within-segment drift of the prediction information or noise?

p' = p + g * (p_t - p_{segment start}); g = -1 holds the segment-start value (segment-reset hold),
g > 0 extrapolates the within-segment update (under-reaction). g is fit per class on three folds and applied to the
fourth (cross-fitted); the in-sample optimum is reported too. Also segment anatomy (weight by segment length).

    scripts/limited .venv/bin/python scripts/cr_dtw_labels_drift.py
"""

from __future__ import annotations

import json

import numpy as np
from cr_dtw_labels_common import OUT, cells, corrected_cells, load, report_cells, wp6

GRID = np.round(np.arange(-1.0, 1.0001, 0.02), 3)


def main():
    t = load()
    res = {"anatomy": {}, "drift": {}}
    L = t["ev_length"][t["sg_sidx"]]
    edges = [1, 2, 4, 16, 64, 256, 1024, 20001]
    for c in range(4):
        sel = t["sg_cls"] == c
        Wc = t["sg_sums"][sel, 1, :, 0].sum(1)
        b = np.digitize(L[sel], edges) - 1
        res["anatomy"][c] = {"n_scored_segments": int(sel.sum()),
                             "weight_share_by_len": {f"{edges[i]}-{edges[i+1]-1}": round(float(Wc[b == i].sum() / Wc.sum()), 4)
                                                     for i in range(len(edges) - 1)}}
    for m, name in enumerate(("parent", "headft")):
        cs = cells(t, t["sg_sums"][:, m].astype(np.float64))
        ref = cs[..., :6]
        curve = {c: [float(wp6(corrected_cells(cs[c:c + 1], np.full((1, 4), g)).sum((0, 1))).mean()) for g in GRID]
                 for c in range(4)}
        ins = {c: float(GRID[int(np.argmax(curve[c]))]) for c in range(4)}
        G = np.zeros((4, 4))
        for k in range(4):
            for c in range(4):
                oth = [f for f in range(4) if f != k]
                sc = [wp6(corrected_cells(cs[c:c + 1][:, oth], np.full((1, 3), g)).sum((0, 1))).mean() for g in GRID]
                G[c, k] = GRID[int(np.argmax(sc))]
        res["drift"][name] = {"insample_g": ins, "crossfit_g": G.tolist(),
                              "curve_class_delta": {c: {float(g): round(v - curve[c][int(np.argmin(np.abs(GRID)))], 5)
                                                        for g, v in zip(GRID, curve[c]) if abs(g * 10 - round(g * 10)) < 1e-9}
                                                    for c in range(4)},
                              "crossfit": report_cells(t, ref, corrected_cells(cs, G)),
                              "hold_segment_start": report_cells(t, ref, corrected_cells(cs, np.full((4, 4), -1.0)))}
        print(name, json.dumps(res["drift"][name], indent=0, default=float))
    (OUT / "drift.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["anatomy"], indent=0))


if __name__ == "__main__":
    main()
