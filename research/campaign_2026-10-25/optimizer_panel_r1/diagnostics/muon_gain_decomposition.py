"""Zero-fit: where does Muon's WP gain over AdamW live? (saved F0/F2 evaluation predictions)

Slices the official pooled weighted Pearson (|y|-weighted, clip ±2, per-target then averaged)
of matched AdamW vs Muon predictions by market class, rows since the label last changed
(run age), row position in the sequence, and |y| decile. Within-slice WP is reported for both
models; slices are diagnostics, not additive contributions to the pooled score.
Usage: muon_gain_decomposition.py OUT_JSON
"""

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from wnn.gmetric import global_wp  # noqa: E402

PAIRS = {"F0": ("i1lead_f0_s1001_control_r1", "optim_f0_s1001_muon_r1", 0),
         "F2": ("i1lead_f2_s1002_control_r1", "optim_f2_s1002_muon_r1", 2)}


def load_pred(name: str, order: list[int]) -> np.ndarray:
    side = json.load(open(f"cache/pred_{name}_valid.json"))
    pred = np.load(f"cache/pred_{name}_valid.npy", mmap_mode="r")
    pos = {int(g): i for i, g in enumerate(side["row_groups"])}
    return np.stack([np.asarray(pred[pos[g]], np.float32) for g in order])


def wp(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 50:
        return float("nan")
    return float(global_wp(y, p, np.ones(len(y), bool))["wp"])


def main() -> None:
    out = {}
    z = np.load("cache/valid_targets.npz")
    tpos = {int(g): i for i, g in enumerate(z["row_group"])}
    for cell, (ctrl, muon, fold) in PAIRS.items():
        split = json.load(open(f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"))
        groups = [int(g) for g in split["evaluation"]["valid"]]
        idx = [tpos[g] for g in groups]
        y = z["y"][idx].astype(np.float32)
        m = z["mask"][idx].copy()
        m[:, :99] = False
        klass = np.repeat(z["market_class"][idx][:, None], y.shape[1], 1)
        change = np.zeros(m.shape, bool)
        change[:, 1:] = (np.abs(np.diff(y, axis=1)) > 1e-6).any(-1)
        pos = np.arange(y.shape[1])[None].repeat(len(y), 0)
        last = np.where(change, pos, 0)
        np.maximum.accumulate(last, axis=1, out=last)
        age = pos - last
        pa, pm = load_pred(ctrl, groups), load_pred(muon, groups)
        Y, A, M = y[m], pa[m], pm[m]
        K, AGE, POS = klass[m], age[m], pos[m]
        mag = np.abs(np.clip(Y, -2, 2)).mean(1)
        res = {"overall": {"adamw": wp(Y, A), "muon": wp(Y, M)}}
        slices = {"class": {str(k): K == k for k in range(4)},
                  "run_age": {"0": AGE == 0, "1-3": (AGE >= 1) & (AGE <= 3), "4-15": (AGE >= 4) & (AGE <= 15),
                              "16-63": (AGE >= 16) & (AGE <= 63), "64+": AGE >= 64},
                  "position": {"99-999": POS < 1000, "1k-5k": (POS >= 1000) & (POS < 5000),
                               "5k-12k": (POS >= 5000) & (POS < 12000), "12k+": POS >= 12000}}
        edges = np.quantile(mag[mag > 0], np.linspace(0, 1, 6))
        slices["abs_y_quintile"] = {f"q{i}": (mag > edges[i]) & (mag <= edges[i + 1]) for i in range(5)}
        for family, parts in slices.items():
            res[family] = {}
            for label, sel in parts.items():
                weight = float(np.abs(np.clip(Y[sel], -2, 2)).sum() / np.abs(np.clip(Y, -2, 2)).sum())
                a, b = wp(Y[sel], A[sel]), wp(Y[sel], M[sel])
                res[family][label] = {"weight_share": weight, "adamw": a, "muon": b, "delta": b - a}
        out[cell] = res
        print(cell, json.dumps({f: {k: round(v["delta"], 4) for k, v in d.items()} if f != "overall" else d
                                for f, d in res.items()}), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
