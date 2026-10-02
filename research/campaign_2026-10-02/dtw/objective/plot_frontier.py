"""Bias-noise frontier of distillation-target constructions (analysis_stopped.json + coherent.json) -> frontier.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
st = json.loads((D / "analysis_stopped.json").read_text())
co = json.loads((D / "coherent.json").read_text())
coh = dict(zip(co["names"], co["all"]["coherent_all"]))
per = dict(zip(co["names"], co["all"]["per_row_all"]))
key = {"y": "y", "a.5": "a.5", "a.7": "a.7", "a.85": "a.85", "T_h8": "T_h8", "T_h32": "T_h32",
       ".5T_h8+.5TE": ".5T_h8+.5TE", ".3T_h8+.7TE": ".3T_h8+.7TE", ".3T_h32+.7TE": ".3T_h32+.7TE",
       "H-059": "H-059 a(u)=.9-.4g", "TD run end": "TE_td (run end, E-174)"}
fam = {"y": 0, "a.5": 0, "a.7": 0, "a.85": 0, "T_h8": 1, "T_h32": 1, ".5T_h8+.5TE": 1, ".3T_h8+.7TE": 1,
       ".3T_h32+.7TE": 1, "H-059": 2, "TD run end": 2}
colors = ["#2a78d6", "#1baf7a", "#eb6834"]
names = ["label/teacher blend (1-a)y + a TE", "stopped bootstrap T_h (this card)", "run-end dependent (H-059, TD)"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, noise, title in ((axes[0], per, "per-row label noise"), (axes[1], coh, "run-coherent label noise (training weights)")):
    for f in range(3):
        xs, ys, ls = [], [], []
        for nm, k in key.items():
            if fam[nm] != f:
                continue
            b = st[k]["bias_mean"]
            xs.append(max(noise[nm], 5e-4)); ys.append(max(1e3 * b, -1.25))
            ls.append(nm if 1e3 * b > -1.25 else f"{nm} ({1e3 * b:.1f}, off scale)")
        ax.scatter(xs, ys, s=46, color=colors[f], edgecolor="white", linewidth=1.5, label=names[f], zorder=3)
        for x, yv, l in zip(xs, ys, ls):
            ax.annotate(l, (x, yv), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel(f"{title}, relative to y (log)")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylim(-1.35, 0.6)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("population bias: dWP x1000 vs target y\n(cross-fitted LS projection on causal [P, TE])")
axes[0].legend(fontsize=8, loc="lower left", frameon=False)
fig.suptitle("Distillation targets: bias vs label noise (OOF panel, 1873 valid sequences)", fontsize=11)
fig.tight_layout()
fig.savefig(D / "frontier.png", dpi=120, bbox_inches="tight")
