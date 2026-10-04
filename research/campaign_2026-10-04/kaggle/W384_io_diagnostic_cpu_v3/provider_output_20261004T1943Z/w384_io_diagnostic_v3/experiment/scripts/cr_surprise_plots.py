"""Surprise miner figure: where class-3 metric mass is lost in event time (reads the forensic row table).

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_plots.py OUT.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_surprise_lib import load, row_loss, weight, wp  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# Run-weighted corr(run target, p) at the run's first (E) row and last row, from cr_surprise_events (t0/t1 mean).
RUN_CORR = {"start": [0.701, 0.651, 0.466, 0.105], "end": [0.874, 0.848, 0.853, 0.809]}


def main() -> None:
    d = load(["cls", "y0", "y1", "ft0", "ft1", "since_E", "nc_until_E", "mask_by_since"])
    L, w = row_loss(d), weight(d)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, a, key in (("rows until next quote move (non-causal)", ax[0, 0], "nc_until_E"),
                         ("rows since last quote move (causal)", ax[0, 1], "since_E")):
        b = np.clip(np.searchsorted(EDGES, d[key], side="right") - 1, 0, len(EDGES) - 1)
        for c in range(4):
            xs, ys = [], []
            for k in range(len(EDGES)):
                s = (d["cls"] == c) & (b == k)
                if s.sum() > 2000:
                    xs.append(k)
                    ys.append(wp(d, "ft", s))
            a.plot(xs, ys, color=COLORS[c], marker=MARKERS[c], lw=2, ms=6, label=f"class {c}")
        a.set_xticks(range(len(EDGES)), [str(e) for e in EDGES], fontsize=8)
        a.set_xlabel(name + ", log2 bins")
        a.set_ylabel("within-bin Global WP")
        a.set_ylim(-0.05, 1.0)
        a.grid(alpha=0.25)
        a.legend(frameon=False, fontsize=8)
    a = ax[1, 0]
    mbs = d["mask_by_since"]
    for c in range(4):
        rate = mbs[c, :12, 1] / np.maximum(mbs[c, :12, 0], 1)
        a.plot(range(12), rate, color=COLORS[c], marker=MARKERS[c], lw=2, ms=6, label=f"class {c}")
    a.set_xticks(range(12), [str(e) for e in EDGES], fontsize=8)
    a.set_xlabel("rows since last quote move, log2 bins")
    a.set_ylabel("P(is_scored)")
    a.grid(alpha=0.25)
    a.legend(frameon=False, fontsize=8)
    a = ax[1, 1]
    for c in range(4):
        a.plot([0, 1], [RUN_CORR["start"][c], RUN_CORR["end"][c]], color=COLORS[c], marker=MARKERS[c], lw=2,
               ms=7)
        a.annotate(f"class {c}", (-0.04, RUN_CORR["start"][c]), fontsize=8, va="center", ha="right",
                   color="#52514e")
    a.set_xticks([0, 1], ["first row of run\n(quote-move row)", "last row of run"])
    a.set_xlim(-0.4, 1.15)
    a.set_ylabel("corr(run target, p), weight |y| x scored rows")
    a.grid(alpha=0.25)
    loss3 = L[d["cls"] == 3].sum() / L.sum()
    fig.suptitle(f"Since-FT pair, clean OOF on 1873 valid sequences: class 3 = {100 * w[d['cls'] == 3].sum():.0f}% "
                 f"of weight, {100 * loss3:.0f}% of 1-WP", fontsize=11)
    fig.savefig(sys.argv[1], dpi=110)


if __name__ == "__main__":
    main()
