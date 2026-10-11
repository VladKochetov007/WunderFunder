"""Plots for SCALING_DEEP_DIVE_20261011.md. All numbers are fold-1 official WP scored with scripts/score_prediction_remote.py
or quoted from LOOP_LOG_20261010.md (noted in the report). Run: scripts/limited .venv/bin/python <this file>."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[5]

# 1 width
w = [256, 320, 384, 448, 512]
ep3 = [.66613, .66722, .66795, .66812, .66722]
ep4 = [.66559, .66664, .66724, .66565, .65800]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axhspan(.6656, .6662, color="0.85", label="control seeds range (0.6656-0.6662)")
ax.axhspan(.6659 - .002, .6659 + .002, color="0.93", zorder=0, label="+-1 one-run sd (0.002) around 0.6659")
ax.plot(w, ep3, "o-", label="3 epochs, seed 3")
ax.plot(w, ep4, "s--", label="4 epochs, seed 3")
ax.plot([256, 384], [.66591, .66847], "k*", ms=12, label="3 epochs, 3-seed means")
ax.set_xlabel("GRU hidden width"); ax.set_ylabel("fold-1 WP"); ax.set_title("Teacher width ladder (h256 base, 2 layers, Muon)")
ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(out / "width_vs_wp.png", dpi=130); plt.close(fig)

# 2 epochs
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ex = [2, 3, 4, 6, 8, 12, 14, 15]
ey = [.66395, .66591, .66642, .64903, .63019, .66259, .65312, .65696]
lab = ["2", "3 (mean of 3)", "4 (mean of 4)", "6", "8 (+EMA, mu3)", "12 (Kaggle stable lr+SWA)", "14 (stable+SWA)", "15 (AdamW wd0.1)"]
axs[0].plot(ex, ey, "o-"); axs[0].axhspan(.6656, .6662, color="0.85")
axs[0].set_xlabel("epochs of training (final checkpoint)"); axs[0].set_ylabel("fold-1 WP"); axs[0].set_title("Final checkpoint vs epoch budget (mixed recipes, see report)")
def curve(p):
    r = [json.loads(l) for l in open(ROOT / p)]
    return [(x["epoch"] + (0 if x["batch"] is None else x["batch"] / 331.0) if not x["epoch_end"] else x["epoch"] + 1, x["wp"]) for x in r]
for n, c in (("vt_f1_tr_ep6 (6 ep, lr 1e-3)", "vt_f1_tr_ep6"), ("vt_f1_tr_base (4 ep)", "vt_f1_tr_base")):
    pts = curve(f"artifacts/runs/{c}/snapshot_curve.jsonl"); pts.sort()
    axs[1].plot([a for a, b in pts], [b for a, b in pts], "o-", ms=3, label=n)
la = [.646, .657, .656, .655, .663, .662, .663, .663, .662, .659, .665, .664, .661, .658, .657]
lm = [.654, .661, .661, .660, .664, .663, .664, .663, .663, .658, .663, .665, .6645, .6671, .658]
axs[1].plot(range(1, 16), la, "-", alpha=.6, label="long AdamW e15 (epoch ends, loop log)")
axs[1].plot(range(1, 16), lm, "-", alpha=.6, label="long Muon-all WSD e15 (epoch ends)")
axs[1].set_ylim(.64, .672); axs[1].axhspan(.6656, .6662, color="0.85")
axs[1].set_xlabel("epochs elapsed"); axs[1].set_title("Intra-run learning curves, fold 1"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "epoch_ladder_and_curves.png", dpi=130); plt.close(fig)

# 3 teacher vs student
rows = [("h256 ep4 teachers s4,s5 (avg)", .66936, "dsq_t256 student", .66938),
        ("h384 ep4 teachers s4,s5 (avg)", .67059, "dsq_t384 student", .66836),
        ("no teachers", None, "dsq_none student", .66861)]
fig, ax = plt.subplots(figsize=(7.5, 4))
import numpy as np
x = np.arange(3)
ax.bar(x - .2, [r[1] or 0 for r in rows], .4, label="teacher average (measured)")
ax.bar(x + .2, [r[3] for r in rows], .4, label="distilled h230 student")
ax.set_ylim(.664, .6715); ax.set_xticks(x); ax.set_xticklabels(["h256 teachers", "h384 teachers", "no teachers"])
ax.axhline(.6659, color="k", ls=":", label="single scratch teacher mean 0.6659")
ax.set_ylabel("fold-1 WP"); ax.legend(fontsize=7); ax.set_title("Teacher ensemble vs student")
fig.tight_layout(); fig.savefig(out / "teacher_vs_student.png", dpi=130); plt.close(fig)

# 4 LR/schedule dot plot
d = [("AdamW lr 5e-4", .66515), ("AdamW lr 1e-3", .66231), ("AdamW lr 2e-3", .66132),
     ("Muon lr 5e-4", .66425), ("Muon lr 1e-3 (base)", .66559), ("Muon lr 2e-3", .66346),
     ("Muon lr-mult 3", .66487), ("Muon lr-mult 6", .66092), ("Muon scope hidden", .66534),
     ("sched WSD 0.2 decay", .66782), ("trapezoid 0.2 to 0", .66744), ("linear to zero", .66504),
     ("SGDR 2 cycles", .66675), ("SGDR 4", .66636), ("SGDR 8", .66621), ("SWA 0.25", .66576),
     ("batch 16", .66583), ("wd 0.05", .66705), ("pure EMA .999 (mu3)", .66493), ("chunk 250", .66724), ("chunk 500 (mean of 3)", .6673)]
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.axvspan(.6656, .6667, color="0.88", label="4-epoch control seeds / mean 0.6664")
ys = range(len(d))[::-1]
ax.plot([v for _, v in d], list(ys), "o")
ax.set_yticks(list(ys)); ax.set_yticklabels([n for n, _ in d], fontsize=8)
ax.set_xlabel("fold-1 WP (single run unless noted; one-run sd 0.002)"); ax.legend(fontsize=7, loc="lower left")
ax.set_title("Optimizer / LR / schedule arms (4-epoch h256 teacher)")
fig.tight_layout(); fig.savefig(out / "lr_schedule_dotplot.png", dpi=130); plt.close(fig)
