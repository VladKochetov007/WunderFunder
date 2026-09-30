"""Figures from immutable control outputs only; no new data or model queries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
p = json.loads((HERE / "prior-result.json").read_text())
d = json.loads((HERE / "detector-output/detector-result.json").read_text())
assert p["status"] == d["status"] == "complete"
slices = p["summary"]["pooled"]["slices"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
ax = axs[0, 0]
models = ["F", "static_R", "T_independent", "T_original"]
names = ["Adaptive\nfrequencies", "Static\ntransition prior", "Adaptive T\nindependent prior", "Adaptive T\nlearned prior"]
values = [slices["all"]["model_losses"]["original"][m]["macro_chain"] for m in models]
ax.bar(range(4), values, color=["#999999", "#78a9ad", "#41878f", "#175c79"])
ax.set_xticks(range(4), names)
ax.set_ylim(0, 2.2)
for i, v in enumerate(values):
    ax.text(i, v+.04, f"{v:.3f}", ha="center")
ax.set_ylabel("Mean-chain log loss (nats / mark), lower is better")
ax.set_title("A. Original detected-code sequences\nSame marks, centers and cross-fit halves", loc="left")

ax = axs[0, 1]
contrasts = ["original_T_independent_minus_T_original", "original_static_R_minus_T_original"]
labels = ["Benefit of transferred R", "Benefit of online adaptation"]
for j, (key, label, color) in enumerate(zip(contrasts, labels, ["#175c79", "#8d7250"])):
    vals = np.array([slices[c]["contrasts"][key]["macro_chain"]["point"] for c in ("first16", "after16")])
    cis = np.array([slices[c]["contrasts"][key]["macro_chain"]["ci95"] for c in ("first16", "after16")])
    x = np.arange(2)+(j-.5)*.32
    ax.bar(x, vals, width=.30, color=color, label=label)
    ax.errorbar(x, vals, yerr=np.array([vals-cis[:,0], cis[:,1]-vals]), fmt="none", color="black", capsize=3)
ax.set_xticks(range(2), ["First 16 marks", "Remaining marks"])
ax.set_ylabel("Control LL − original T LL (nats / mark)")
ax.set_ylim(0, .88)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("B. Prior and adaptation contrasts\nConditional chain 95% intervals; not additive shares", loc="left")

ax = axs[1, 0]
seeds = (20261011, 20261012, 20261013)
for j, seed in enumerate(seeds):
    losses = slices["all"]["model_losses"][f"block128_{seed}"]
    vals = [losses["F"]["macro_chain"]-losses[m]["macro_chain"] for m in ("T_old_R", "T_matched_R")]
    ax.plot([0, 1], vals, "o-", label=f"seed {seed}", alpha=.8)
ax.axhline(0, color=".5", linewidth=1)
ax.set_xticks([0, 1], ["Original-order R", "Shuffle-matched R"])
ax.set_xlim(-.25, 1.3); ax.set_ylim(-.105, .015)
ax.set_ylabel("F LL − T LL (nats / shuffled mark)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("C. Block128 MARK shuffle: correct the prior\nIdentical evaluation marks and online counts", loc="left")

ax = axs[1, 1]
keys = ["original"] + [f"run128_{seed}" for seed in seeds] + ["mean_three_surrogates"]
stats = d["bootstrap"]["equal_chain"]
vals = np.array([stats[k]["point"] for k in keys])
cis = np.array([stats[k]["ci95"] for k in keys])
ax.bar(range(5), vals, color=["#175c79"]+["#83a4a0"]*3+["#3b7773"])
ax.errorbar(range(5), vals, yerr=np.array([vals-cis[:,0], cis[:,1]-vals]), fmt="none", color="black", capsize=3)
ax.axhline(0, color=".5", linewidth=1)
ax.set_xticks(range(5), ["Original", "Run seed 1", "Run seed 2", "Run seed 3", "Mean 3"], rotation=15)
ax.set_ylabel("F LL − T LL (nats / detected mark)")
ax.set_title("D. Block128 RAW-RUN shuffle + redetection\nMatched priors; each condition has its own events", loc="left")
fig.suptitle("Quant controls: nuisance-code prediction, not price-prediction gains", fontsize=15)
fig.savefig(HERE / "controls.png", dpi=160)
fig.savefig(HERE / "controls.svg")
