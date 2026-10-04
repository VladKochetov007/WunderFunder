"""DTW streams: which train-fitted move table decodes i1 in each market class (coverage + co-movement)?

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_i1table.py [N_PER_CLASS]

For sampled validation sequences of each i0 class, decode i1 with each class table (1, 2, 3) and
report the unknown-transition share and the B-row return correlation with the decoded i0 mid
(i0 decoded with its own class table; class 0 has no table and is skipped for i0).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.data import SequenceFile  # noqa: E402
from wnn.ticks import reconstruct  # noqa: E402


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    z = np.load(ROOT / "cache/valid_targets.npz")
    cls, rg, Y = z["market_class"], z["row_group"], z["y"]
    sf = SequenceFile(ROOT / "datasets/valid.parquet")
    tabs = {c: pickle.load(open(ROOT / f"cache/class{c}_moves_full.pkl", "rb")) for c in (1, 2, 3)}
    rng = np.random.default_rng(0)
    res = {}
    for c0 in (0, 1, 2, 3):
        idx = rng.choice(np.flatnonzero(cls == c0), n_per, replace=False)
        acc = {c: {"unk": [], "xc": np.zeros(3), "xcl": np.zeros(3), "yd": np.zeros(3)} for c in (1, 2, 3)}
        n1 = []
        for i in idx:
            x = sf.read(int(rg[i]), targets=False, mask=False).features
            x1 = x.copy()
            x1[:, 0:22] = x[:, 52:74]
            ev1 = max(int((x1[1:, :22] != x1[:-1, :22]).any(1).sum()), 1)
            n1.append(len(np.unique(np.round(x1[:, :22], 4), axis=0)))
            m0 = None
            if c0 >= 1:
                a, b, _ = reconstruct(x, *tabs[c0])
                m0 = 0.5 * (a + b).astype(float)
            yD = (np.clip(Y[i][:, 0], -2, 2) - np.clip(Y[i][:, 1], -2, 2)) / 2
            for c in (1, 2, 3):
                a1, b1, u = reconstruct(x1, *tabs[c])
                m1 = 0.5 * (a1 + b1).astype(float)
                acc[c]["unk"].append(u / ev1)
                r1 = np.diff(m1[::100])
                # i1 past 100-row move vs target direction (raw, sign of association only)
                f1 = m1[99:] - m1[:-99]
                yy = yD[99:]
                acc[c]["yd"] += [np.dot(f1 - f1.mean(), yy - yy.mean()), np.dot(f1 - f1.mean(), f1 - f1.mean()),
                                 np.dot(yy - yy.mean(), yy - yy.mean())]
                if m0 is not None:
                    r0 = np.diff(m0[::100])
                    acc[c]["xc"] += [np.dot(r0, r1), np.dot(r0, r0), np.dot(r1, r1)]
                    acc[c]["xcl"] += [np.dot(r0[1:], r1[:-1]), np.dot(r0[1:], r0[1:]), np.dot(r1[:-1], r1[:-1])]
        out = {"i1_unique_codes_median": float(np.median(n1))}
        for c in (1, 2, 3):
            s, sl, sy = acc[c]["xc"], acc[c]["xcl"], acc[c]["yd"]
            out[f"table{c}"] = {"unknown_i1": round(float(np.mean(acc[c]["unk"])), 4),
                                "corr_r0_r1_B100": round(float(s[0] / np.sqrt(s[1] * s[2] + 1e-12)), 4),
                                "corr_r0_next_r1_lead_B100": round(float(sl[0] / np.sqrt(sl[1] * sl[2] + 1e-12)), 4),
                                "corr_i1_99row_move_yD": round(float(sy[0] / np.sqrt(sy[1] * sy[2] + 1e-12)), 4)}
        res[f"i0_class{c0}"] = out
        print(c0, json.dumps(out), flush=True)
    (ROOT / "research/campaign_2026-10-02/dtw/streams/i1_table.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
