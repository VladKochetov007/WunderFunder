"""Check the trade-PnL reading of the targets on validation sequences.

Hypothesis (research/findings-2026-09-26.md): t0 = F - ask and t1 = bid - F for one future
reference price F, so S = t0 + t1 = -(spread) is known at prediction time and all the unknown
signal sits in D = (t0 - t1) / 2 = F - mid.

Per market class this reports, on scored rows:
  - share of the metric's weighted target variance carried by S versus D,
  - how often S changes on rows without an i0 price-code event (S should be a function of the book),
  - the distribution of -S in ticks (spread),
  - the weighted Pearson ceiling of predicting only S exactly (D set to its weighted mean).

    .venv/bin/python scripts/check_target_law.py --seqs 400
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wnn.data import TARGET_QUANTA, SequenceFile, market_class

ROOT = Path(__file__).resolve().parents[1]
CLIP = 2.0


def pooled_wp(y: np.ndarray, p: np.ndarray) -> float:
    """Pooled clipped |y|-weighted Pearson for one target column."""
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    p = np.clip(p.astype(np.float64), -CLIP, CLIP)
    w = np.abs(y)
    tot = w.sum()
    my, mp = (w * y).sum() / tot, (w * p).sum() / tot
    cov = (w * (y - my) * (p - mp)).sum()
    return float(cov / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=400, help="validation sequences, evenly spaced")
    ap.add_argument("--out", default="research/artifacts/target_law_check.json")
    args = ap.parse_args()

    f = SequenceFile(ROOT / "datasets" / "valid.parquet")
    picks = np.linspace(0, len(f) - 1, args.seqs).astype(int)
    acc = defaultdict(lambda: defaultdict(list))
    for g in picks:
        s = f.read(g)
        cls = market_class(s.targets)
        m = s.scored.copy()
        m[:99] = False
        y = s.targets.astype(np.float64)
        ssum = y[:, 0] + y[:, 1]
        price_event = np.r_[False, (s.features[1:, 0:22] != s.features[:-1, 0:22]).any(1)]
        s_change = np.r_[False, np.abs(np.diff(ssum)) > 1e-6]
        a = acc[cls]
        a["rows"].append(m.sum())
        a["s_change_rows"].append((s_change & ~price_event).sum())
        a["s_change_total"].append(s_change.sum())
        a["y"].append(y[m])
        if cls >= 0:
            a["spread_ticks"].append(np.rint(-ssum[m] / TARGET_QUANTA[cls]).astype(int))

    report = {}
    for cls, a in sorted(acc.items()):
        y = np.concatenate(a["y"])
        ssum, d = y[:, 0] + y[:, 1], (y[:, 0] - y[:, 1]) / 2
        w = np.abs(np.clip(y, -CLIP, CLIP)).sum(1)
        wvar = lambda v: float((w * (v - (w * v).sum() / w.sum()) ** 2).sum() / w.sum())  # noqa: E731
        vs, vd = wvar(ssum / 2), wvar(d)
        # ceiling when only the half-sum is known: prediction = S/2 +- mean(D)
        md = float((w * d).sum() / w.sum())
        pred = np.column_stack([ssum / 2 + md, ssum / 2 - md])
        ceil_s = np.mean([pooled_wp(y[:, k], pred[:, k]) for k in range(2)])
        sp = np.concatenate(a["spread_ticks"]) if a["spread_ticks"] else np.array([], int)
        vals, counts = np.unique(sp, return_counts=True)
        report[str(cls)] = {
            "sequences": len(a["rows"]),
            "scored_rows": int(sum(a["rows"])),
            "weighted_var_share_halfsum": vs / (vs + vd),
            "weighted_var_share_halfdiff": vd / (vs + vd),
            "wp_ceiling_halfsum_only": float(ceil_s),
            "s_changes_off_price_event_frac": float(sum(a["s_change_rows"]) / max(1, sum(a["s_change_total"]))),
            "spread_ticks_top": {int(v): float(c / counts.sum()) for v, c in sorted(zip(vals, counts), key=lambda t: -t[1])[:6]},
        }
        print(cls, json.dumps(report[str(cls)]))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
