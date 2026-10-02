"""Learning-dynamics read of existing training logs (no GPU, no data): loss vs optimizer step, epoch means,
within-epoch slopes, logging noise, AMP skips, and epoch-wise validation curves where the run evaluated.

    scripts/limited .venv/bin/python research/campaign_2026-10-02/opt/learning_dynamics.py
Writes opt_*.png and dynamics.json next to this file.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CLOG = ROOT / "research/campaign_2026-09-29/logs"
ALOG = ROOT / "artifacts/logs"
STEP = re.compile(r"^ep (\d+) batch (\d+)/(\d+) loss ([\d.naninf]+) lr ([\d.e+-]+)")

FAMILIES = {
    "GRU128 teacher (since, EMA-4)": [CLOG / f"c20260929_since_s{s}_f{f}.log" for s in (3, 4) for f in range(4)],
    "GRU256 student a0.5": [CLOG / f"c20261001_d256_f{f}.log" for f in range(4)],
    "GRU256 student a0.7": [CLOG / "c20261001_d256a07_f0.log", CLOG / "c20261002_d256a07_f1r.log"],
    "GRU256 student a0.85": [CLOG / "c20261001_d256a085_f0.log"],
    "GRU256 teacher-ens a0.7": [CLOG / "c20261002_te07_f0.log", CLOG / "c20261002_te07_f1.log"],
    "LSTM192 student a0.5": [CLOG / "c20261002_dlstm192_f0.log", CLOG / "c20261002_dlstm192_f1.log"],
    "GRU128 12ep wd0.01 (no EMA)": [ALOG / "ab47_long_wd0p01_f0.log"],
    "GRU128 12ep wd0.1 (no EMA)": [ALOG / "ab47_long_wd0p1_f0.log"],
    "GRU128 EMA-6": [ALOG / "ab68_ema6_since_f0.log", ALOG / "ab68_ema6_since_f1.log"],
    "GRU192 EMA-6": [ALOG / "ab81_h192_f0.log"],
}


def parse(path: Path) -> dict:
    lines = path.read_text().splitlines()
    cfg = json.loads(lines[0])
    steps_per_batch = math.ceil(20_000 / cfg["chunk"])
    pts, valid, skips = [], [], []
    for line in lines:
        m = STEP.match(line)
        if m:
            ep, b, nb, loss, lr = int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5])
            pts.append((ep, (ep * nb + b + 1) * steps_per_batch, loss, lr, nb))
        elif line.startswith(("VALID ", "FIT ")):
            rec = json.loads(line.split(" ", 1)[1])
            if "weighted_pearson" in rec:
                valid.append(rec["weighted_pearson"])
            diag = rec.get("update_diagnostics", {}).get("epoch")
            if diag:
                skips.append((diag["amp_skipped_updates"], diag["optimizer_attempts"]))
    a = np.array(pts, dtype=float)
    return {"cfg": cfg, "ep": a[:, 0].astype(int), "step": a[:, 1], "loss": a[:, 2], "lr": a[:, 3],
            "batches": int(a[0, 4]), "valid": valid[: cfg["epochs"]], "skips": skips}


def epoch_stats(r: dict) -> dict:
    """Epoch means of the logged (last-chunk, single-batch) loss, first/second-half slope and logging noise."""
    out = {"mean": [], "half_drop": [], "noise_sd": []}
    for e in range(r["cfg"]["epochs"]):
        v = r["loss"][r["ep"] == e]
        v = v[np.isfinite(v)]
        if len(v) < 6:
            continue
        h = len(v) // 2
        out["mean"].append(float(v.mean()))
        out["half_drop"].append(float(v[:h].mean() - v[h:].mean()))
        # noise of one logged point around a 5-point running median (batch composition + chunk position)
        med = np.array([np.median(v[max(0, i - 2): i + 3]) for i in range(len(v))])
        out["noise_sd"].append(float(np.std(v - med) * 1.25))
    return out


def smooth(x: np.ndarray, k: int = 7) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.median(pad[i: i + k]) for i in range(len(x))])


def main() -> None:
    runs = {fam: [parse(p) for p in paths if p.exists()] for fam, paths in FAMILIES.items()}
    summary = {}
    for fam, rs in runs.items():
        st = [epoch_stats(r) for r in rs]
        ne = min(len(s["mean"]) for s in st)
        means = np.array([s["mean"][:ne] for s in st])
        summary[fam] = {
            "runs": [r["cfg"]["name"] for r in rs],
            "hidden": rs[0]["cfg"]["hidden"], "cell": rs[0]["cfg"].get("cell", "gru"),
            "epochs": rs[0]["cfg"]["epochs"], "steps_total": int(rs[0]["batches"] * math.ceil(20_000 / rs[0]["cfg"]["chunk"])
                                                                * rs[0]["cfg"]["epochs"]),
            "epoch_mean_loss": means.mean(0).round(4).tolist(),
            "epoch_mean_loss_rel_last": (means / means[:, -1:]).mean(0).round(4).tolist(),
            "within_epoch_half_drop": np.array([s["half_drop"][:ne] for s in st]).mean(0).round(4).tolist(),
            "logged_point_noise_sd": np.array([s["noise_sd"][:ne] for s in st]).mean(0).round(4).tolist(),
            "valid_wp": [r["valid"] for r in rs if r["valid"]],
            "amp_skips": [sum(s for s, _ in r["skips"]) for r in rs if r["skips"]],
            "amp_attempts": [sum(a for _, a in r["skips"]) for r in rs if r["skips"]],
        }
    (OUT / "dynamics.json").write_text(json.dumps(summary, indent=1) + "\n")

    # 1. loss vs step, campaign recipe families (4-epoch OneCycle, 14,920 steps)
    fams = ["GRU128 teacher (since, EMA-4)", "GRU256 student a0.5", "GRU256 student a0.7", "LSTM192 student a0.5",
            "GRU256 teacher-ens a0.7"]
    colors = dict(zip(fams, ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B3"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam in fams:
        for i, r in enumerate(runs[fam]):
            ok = np.isfinite(r["loss"])
            axes[0].plot(r["step"][ok], smooth(r["loss"][ok]), color=colors[fam], alpha=0.75, lw=1,
                         label=fam if i == 0 else None)
            last = np.mean(r["loss"][(r["ep"] == r["cfg"]["epochs"] - 1) & ok])
            axes[1].plot(r["step"][ok], smooth(r["loss"][ok]) / last, color=colors[fam], alpha=0.75, lw=1,
                         label=fam if i == 0 else None)
    r0 = runs[fams[0]][0]
    for ax in axes:
        for e in range(1, 4):
            ax.axvline(e * r0["batches"] * 10, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("optimizer step (TBPTT chunk)")
    axes[0].set_ylabel("logged train loss (last chunk of batch, 7-pt running median)")
    axes[0].set_ylim(0.6, 2.6)
    axes[1].set_ylabel("loss / run's final-epoch mean")
    axes[1].set_ylim(0.85, 1.6)
    ax2 = axes[1].twinx()
    ax2.plot(r0["step"], r0["lr"], color="black", lw=1, ls="--")
    ax2.set_ylabel("OneCycle lr (dashed)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Train loss: teacher vs distilled students (targets differ, levels not comparable)")
    axes[1].set_title("Normalized to the last epoch: how much is still moving")
    fig.tight_layout()
    fig.savefig(OUT / "opt_loss_vs_step.png", dpi=110)
    plt.close(fig)

    # 2. per-epoch mean train loss relative to last epoch vs per-epoch valid gain
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam in fams:
        s = summary[fam]
        axes[0].plot(range(len(s["epoch_mean_loss_rel_last"])), s["epoch_mean_loss_rel_last"], "o-",
                     color=colors[fam], label=fam)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("epoch-mean train loss / last epoch")
    axes[0].set_title("Train loss keeps falling in the last epoch (memorization budget)")
    axes[0].legend(fontsize=8)
    curves = {
        "GRU128 EMA-4 since f0 (ab62)": "ab62_since_f0", "GRU128 EMA-4 since f1 (ab62)": "ab62_since_f1",
        "GRU128 EMA-6 f0 (ab68)": "ab68_ema6_since_f0", "GRU128 EMA-6 f1 (ab68)": "ab68_ema6_since_f1",
        "GRU192 EMA-6 f0 (ab81)": "ab81_h192_f0", "GRU128 EMA-6 wd0.1 f0": "ab81_wd01_f0",
        "GRU128 12ep wd0.01 noEMA": "ab47_long_wd0p01_f0", "GRU128 12ep wd0.1 noEMA": "ab47_long_wd0p1_f0",
        "GRU128 chunk500 EMA-4 f0": "ab66_ch500_f0",
        "final GRU128 since s5 (holdout)": "c20261001_final_since_s5",
        "final GRU256 student a0.5 (holdout)": "c20261001_final_d256_s3",
    }
    vc = {}
    for label, run in curves.items():
        h = json.loads((ROOT / "artifacts/runs" / run / "history.json").read_text())
        wp = np.array([x["weighted_pearson"] for x in h])
        vc[label] = wp.round(5).tolist()
        frac = (np.arange(len(wp)) + 1) / len(wp)
        ls = "--" if "noEMA" in label else "-"
        axes[1].plot(frac, (wp - wp.max()) * 1e4, ls, marker="o", ms=3, label=f"{label} max {wp.max():.4f}")
    axes[1].set_xlabel("fraction of the schedule completed (end of epoch)")
    axes[1].set_ylabel("valid WP - run max (1e-4)")
    axes[1].set_ylim(-60, 5)
    axes[1].set_title("Validation by epoch: EMA saturates, wider/no-EMA decline as LR decays")
    axes[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "opt_epoch_curves.png", dpi=110)
    plt.close(fig)
    summary["_valid_curves"] = vc
    (OUT / "dynamics.json").write_text(json.dumps(summary, indent=1) + "\n")

    # 3. 12-epoch no-EMA runs: loss vs lr phase (memorization as the LR decays)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam, c in [("GRU128 12ep wd0.01 (no EMA)", "#C44E52"), ("GRU128 12ep wd0.1 (no EMA)", "#4C72B0")]:
        r = runs[fam][0]
        ax.plot(r["step"], smooth(r["loss"], 9), color=c, lw=1, label=f"{fam} train loss")
        nb = r["batches"] * 10
        ax.plot((np.arange(len(r["valid"])) + 1) * nb, 10 * (np.array(r["valid"]) - 0.6), "o-", color=c, ms=4,
                label=f"{fam} 10x(valid WP-0.6)")
    ax2 = ax.twinx()
    ax2.plot(r["step"], r["lr"], "k--", lw=1)
    ax2.set_ylabel("lr")
    ax.set_xlabel("optimizer step")
    ax.legend(fontsize=7)
    ax.set_title("12-epoch OneCycle without EMA: valid peaks at mid-LR, falls as LR -> 0")
    fig.tight_layout()
    fig.savefig(OUT / "opt_long12_memorization.png", dpi=110)
    plt.close(fig)
    # 4. schedules: the recipe's OneCycle (lr and the beta1 it cycles) vs the proposed WSD, per-epoch valid gains
    import sys

    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from wnn.optim import wsd_lambda

    steps = 3730
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for epochs, ls in ((4, "-"), (6, "--")):
        total = steps * epochs
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.05,
                                                  final_div_factor=100)
        lr, b1 = [], []
        for _ in range(total):
            lr.append(opt.param_groups[0]["lr"])
            b1.append(opt.param_groups[0]["betas"][0])
            opt.step()
            sch.step()
        x = np.arange(total) / steps
        axes[0].plot(x, lr, "k" + ls, lw=1, label=f"OneCycle {epochs} ep (recipe)" if epochs == 4 else "OneCycle 6 ep")
        f = wsd_lambda(total, 0.05, 0.2, 0.01)
        axes[0].plot(x, [1e-3 * f(i) for i in range(total)], color="#C44E52", ls=ls, lw=1, label=f"WSD {epochs} ep")
        if epochs == 4:
            ax2 = axes[0].twinx()
            ax2.plot(x, b1, color="#55A868", lw=0.8)
            ax2.set_ylabel("OneCycle beta1 (green)")
    axes[0].axvline(4 * 0.42, color="grey", ls=":", lw=0.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("lr")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title("Schedules (EMA 0.999 horizon = 1,000 steps = 0.27 epoch)")
    gains = {
        "GRU128 EMA-4 f0/f1": [vc["GRU128 EMA-4 since f0 (ab62)"], vc["GRU128 EMA-4 since f1 (ab62)"]],
        "GRU128 EMA-6 f0/f1": [vc["GRU128 EMA-6 f0 (ab68)"], vc["GRU128 EMA-6 f1 (ab68)"]],
        "GRU192 EMA-6 f0": [vc["GRU192 EMA-6 f0 (ab81)"]],
        "final GRU256 student (holdout)": [vc["final GRU256 student a0.5 (holdout)"]],
        "final GRU128 since s5 (holdout)": [vc["final GRU128 since s5 (holdout)"]],
    }
    for label, curves in gains.items():
        g = np.mean([np.diff(c) for c in curves], axis=0) * 1e4
        axes[1].plot(np.arange(1, len(g) + 1), g, "o-", label=label)
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_yscale("symlog", linthresh=5)
    axes[1].set_xlabel("epoch e (gain from e-1 to e)")
    axes[1].set_ylabel("valid WP gain (1e-4, symlog)")
    axes[1].set_title("Per-epoch validation gain: the low-lr tail buys ~0 (students) or loses (wider)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "opt_schedule_gains.png", dpi=110)
    plt.close(fig)
    for fam, s in summary.items():
        if fam.startswith("_"):
            continue
        print(f"{fam:32s} rel {s['epoch_mean_loss_rel_last']} halfdrop {s['within_epoch_half_drop']} "
              f"noise {s['logged_point_noise_sd']} skips {s['amp_skips']}/{s['amp_attempts']}")


if __name__ == "__main__":
    main()
