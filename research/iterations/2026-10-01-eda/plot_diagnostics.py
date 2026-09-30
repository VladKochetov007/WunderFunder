"""Render only frozen JSON readouts; no source data, fitting or score queries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
def read(name):
    return json.loads((HERE / name).read_text())

t = read("training-gradient-result.json")
e = read("ensemble-statistics-result.json")["scopes"]["pooled"]
c = read("clock-output/clock-result.json")["by_class"]["3"]
q = read("quant-mark-result.json")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
ax = axs[0, 0]
values = np.array([[w["shared_cosine_main_aux"] for w in case["windows"]] for case in t["cases"]])
im = ax.imshow(values, vmin=-.22, vmax=.22, cmap="RdBu", aspect="auto")
ax.set_yticks(range(8), [f"fold {r['fold']} / seed {r['seed']} / {'parent' if r['stage']=='parent_ep3' else 'head1'}" for r in t["cases"]])
ax.set_xticks(range(3), ["rows 0–1999", "8000–9999", "18000–19999"])
for i in range(8):
    for j in range(3):
        ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white" if values[i,j]>.12 else "black")
ax.set_title("A. Main / combined-aux shared-gradient cosine\nFixed checkpoints; all full-batch clip factors = 1", loc="left")

ax = axs[0, 1]
v = np.array(e["derivative"]["mean_targets"])*1e4
ci = np.array(e["conditional_chain_derivative_uncertainty"]["ci95"])*1e4
ax.errorbar(v, range(4), xerr=np.stack([v-ci[:,0], ci[:,1]-v]), fmt="o", capsize=4, color="#166b90")
ax.axvline(0, color=".6", lw=1)
ax.set_yticks(range(4), ["distilled 256", "128 seed 3", "128 seed 4", "128 chunk 500"])
ax.invert_yaxis()
ax.set_xlabel("Local score sensitivity per unit member scale × 10⁴\nConditional whole-chain 95% interval; zero convention at kinks")
ax.set_title("B. Fixed C member-scale sensitivity\nNo coefficient fit or estimated finite score gain", loc="left")

ax = axs[1, 0]
blocks = ["i0_volume_code", "i0_trade_code", "i1_feature_code"]
vals = np.array([c["interval_EDA_fixed_A"][b]["nonempty"]["estimate"] for b in blocks])*100
cis = np.array([c["interval_EDA_fixed_A"][b]["nonempty"]["conditional_chain_ci95"] for b in blocks])*100
ax.barh(range(3), vals, color=["#338a88", "#338a88", "#8bb9b4"], height=.6)
ax.errorbar(vals, range(3), xerr=np.stack([vals-cis[:,0], cis[:,1]-vals]), fmt="none", color="black", capsize=4)
ax.set_yticks(range(3), ["i0 volume codes", "i0 trade codes", "i1 feature codes"])
ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("Closed class-3 intervals containing an interior input update (%)")
ax.set_title("C. Observed history between fixed-A clock updates\n~60.9% is almost exactly the fraction with gap > 1", loc="left")
ax.text(.98, .05, "Clock A/B XOR: 8 / 735,555 code events\nJoint fit + audit 95% upper bound: 0.00476%\nStable learned partition ≠ physical quote identification", transform=ax.transAxes, fontsize=8, va="bottom", ha="right", bbox={"facecolor":"white", "edgecolor":"none", "alpha":.95})

ax = axs[1, 1]
keys = list(q["summary"])
vals = [q["summary"][k]["mean_chain_gain_F_minus_T"] for k in keys]
ax.bar(range(7), vals, color=["#166b90"]+["#aaaaaa"]*3+["#6f9a91"]*3)
lo, hi = q["bootstrap"]["chain_gain"]["ci95"]
ax.errorbar([0], [vals[0]], yerr=[[vals[0]-lo], [hi-vals[0]]], fmt="none", color="black", capsize=4)
ax.axhline(0, color=".5", lw=1)
ax.set_xticks(range(7), ["original", "full 1", "full 2", "full 3", "block 1", "block 2", "block 3"], rotation=25)
ax.set_ylabel("Adaptive-frequency LL − transition LL (nats / mark)")
ax.set_title("D. Next detected code-mark prediction\nFrozen recipe; all three seeds for each offline shuffle", loc="left")
ax.text(.98, .96, "Detected-code regularity; not price alpha\nOriginal priors retained under shuffling", transform=ax.transAxes, ha="right", va="top", fontsize=9)
fig.suptitle("Additional EDA round — fixed analyses, no new forecasting-model training", fontsize=15)
fig.savefig(HERE / "diagnostics.png", dpi=160)
fig.savefig(HERE / "diagnostics.svg")
