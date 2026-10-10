"""Figures and component-bootstrap CIs for CURRENT_BEST_SOLUTION_EXPLAINED.md (read-only; fold 1 bank + logs)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from crossfold_blend import load_fold  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/solution_report"
PL = OUT / "plots"
BANK = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl"
DI = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/external_ideation_20261010/round3"


def sums(y, p, m):
    """Per-target raw sums, shape (2, 6): the official WP is the mean of the two per-target pooled correlations."""
    yc, pc = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = np.abs(yc) * m[..., None]
    ax = tuple(range(yc.ndim - 1))
    return np.stack([w.sum(ax), (w * yc).sum(ax), (w * pc).sum(ax), (w * yc * yc).sum(ax), (w * pc * pc).sum(ax), (w * yc * pc).sum(ax)], -1)


def wp(s):
    w, wy, wp_, wyy, wpp, wyp = np.moveaxis(s, -1, 0)
    my, mp = wy / w, wp_ / w
    return float(np.mean((wyp / w - my * mp) / np.sqrt((wyy / w - my * my) * (wpp / w - mp * mp))))


def lr_curves():
    def onecycle(total, max_lr, pct=0.05):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=max_lr)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total, pct_start=pct, final_div_factor=100)
        out = []
        for _ in range(total):
            out.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
        return np.array(out)
    t_total = 22380  # teacher: 373 batches x 20 chunks x 3 epochs (log: steps 22380)
    s_total = 19890  # student: batch 16 -> 663 batches x 10 chunks(2000 rows) x 3 epochs
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(np.arange(t_total) / 7460, onecycle(t_total, 1e-3), label="teacher: Muon(rnn)+AdamW, max lr 1e-3")
    ax[0].plot(np.arange(s_total) / 6630, onecycle(s_total, 1e-4), label="student: AdamW, max lr 1e-4")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("learning rate"); ax[0].set_title("OneCycle (5% warm-up, cosine to lr/ (25*100))"); ax[0].legend(fontsize=7)
    x = np.arange(t_total) / 7460
    warm = int(0.05 * t_total); dec = int(0.2 * t_total)
    wsd = np.where(np.arange(t_total) < warm, np.arange(t_total) / warm, np.where(np.arange(t_total) < t_total - dec, 1.0, 1 - np.sqrt(np.clip((np.arange(t_total) - (t_total - dec)) / dec, 0, 1))))
    ax[1].plot(x, onecycle(t_total, 1e-3), label="onecycle (default)")
    ax[1].plot(x, 1e-3 * np.clip(wsd, 0.01, 1), label="wsd_sqrt (HPO option; best trials use wsd_linear/sqrt)")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Schedule options searched by HPO"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PL / "lr_schedules.png", dpi=130); plt.close(fig)


def curves():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, f in (("default 4 epochs", "learning_curve_base_f1.jsonl"), ("6 epochs", "learning_curve_ep6_f1.jsonl")):
        rows = [json.loads(l) for l in (DI / f).read_text().splitlines()]
        xs = [r["epoch"] + (0.5 if r["batch"] == 123 else 0.75 if r["batch"] == 248 else 1.0) for r in rows]
        ax.plot(xs, [r["wp"] for r in rows], marker="o", ms=3, label=name)
    ax.set_xlabel("epoch (snapshots at ~0.33, 0.67 and end of each epoch)"); ax.set_ylabel("fold-1 WP"); ax.set_ylim(0.64, 0.675)
    ax.set_title("Learning curves (scored from saved snapshots)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PL / "learning_curves.png", dpi=130); plt.close(fig)


def ladder():
    leads = [0, 2, 8, 32, 128, 512]; wps = [0.666, 0.680, 0.711, 0.780, 0.891, 0.960]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([0.5, 2, 8, 32, 128, 512], wps, marker="o"); ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 2, 8, 32, 128, 512]); ax.set_xticklabels(["0 (causal)", "2", "8", "32", "128", "512"])
    ax.set_xlabel("rows of FUTURE features visible (teacher-only probe)"); ax.set_ylabel("fold-1 WP"); ax.grid(alpha=0.3)
    ax.set_title("Information ceiling: label is a function of the future path")
    fig.tight_layout(); fig.savefig(PL / "lookahead_ladder.png", dpi=130); plt.close(fig)


def hpo():
    ev = [json.loads(l) for l in (ROOT / "hpo/studies/teacher_v2/events.jsonl").read_text().splitlines()]
    stage = {}
    for e in ev:
        if e["ev"] == "stage_result":
            stage.setdefault(e["trial"], {})[e["fold"]] = e["wp"]
    pts = [(s[1], s[0]) for s in stage.values() if 0 in s and 1 in s]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18)
    ax.axvline(0.66642, color="gray", ls="--", lw=0.8); ax.text(0.66645, min(p[1] for p in pts), "default 4-seed mean (fold 1)", fontsize=7, rotation=90)
    ax.set_xlabel("fold 1 WP"); ax.set_ylabel("fold 0 WP (about +0.013 easier)"); ax.set_title(f"HPO trials with both dev folds (n={len(pts)})")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(PL / "hpo_fold_scatter.png", dpi=130); plt.close(fig)
    d = json.loads((ROOT / "hpo/evidence/scratch_effects_fold1.json").read_text())
    rows = d["rows"][:12][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, r in enumerate(rows):
        ax.plot(r["ci95"], [i, i], color="C0"); ax.plot(r["delta"], i, "o", color="C0")
        ax.plot(r.get("shrunk", 0), i, "x", color="C3")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"][:34] for r in rows], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("effect vs 4-seed default mean (o raw delta with 95% interval, x shrunk estimate)")
    ax.set_title("Evidence report, fold 1: biggest raw deltas (+0.002) vs noise"); fig.tight_layout(); fig.savefig(PL / "evidence_forest.png", dpi=130); plt.close(fig)


def ci_table():
    y, mask, comp, models = load_fold(BANK, 1, ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json", ROOT / "cache/valid_targets.npz")
    order = next(iter(models.values()))[2]
    names = {"h230_f1_s3_ep3": "student: H230 ep3 (distilled from 4 families)", "vt_f1_dsq_t256": "student: distilled from h256 scratch teachers",
             "vt_f1_chunk500_h384_e3": "teacher: scratch h384 chunk500 3ep", "vt_f1_tr_base": "teacher: scratch default (telemetry run)",
             "h230_f1_s3_ep3_boot8": "student: ep3 + bootstrap 8"}
    extra = {"r3_ctrl_f1_s3_e3": "teacher: scratch default v6 s3", "r3_ctrl_f1_s4_e3": "teacher: scratch default v6 s4"}
    arrays = {}
    for st, (n, arr, o) in models.items():
        if n in names: arrays[n] = np.asarray(arr[o], dtype=np.float32)
    for n in extra:
        arrays[n] = np.asarray(np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r")[order], dtype=np.float32)
    labels = {**names, **extra}
    comps = np.unique(comp)
    idx = {c: np.flatnonzero(comp == c) for c in comps}
    rng = np.random.default_rng(0)
    res = {}
    base_sums = {n: np.array([sums(y[idx[c]], a[idx[c]], mask[idx[c]]) for c in comps]) for n, a in arrays.items()}
    ref = base_sums["r3_ctrl_f1_s3_e3"]
    boots = rng.integers(0, len(comps), size=(1000, len(comps)))
    for n, S in base_sums.items():
        full = wp(S.sum(0))
        b = np.array([wp(S[bi].sum(0)) for bi in boots])
        d = np.array([wp(S[bi].sum(0)) - wp(ref[bi].sum(0)) for bi in boots])
        res[n] = {"label": labels[n], "wp": float(full), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                  "delta_vs_ctrl_s3": float(full - wp(ref.sum(0))), "delta_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    (OUT / "fold1_ci.json").write_text(json.dumps(res, indent=1))
    order_names = sorted(res, key=lambda n: res[n]["wp"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, n in enumerate(order_names):
        r = res[n]; ax.plot(r["ci95"], [i, i], color="C0"); ax.plot(r["wp"], i, "o", color="C0")
    ax.set_yticks(range(len(order_names))); ax.set_yticklabels([res[n]["label"] for n in order_names], fontsize=7)
    ax.set_xlabel(f"fold-1 WP, 95 percent component-bootstrap interval (1000 draws, {len(comps)} components)"); ax.grid(alpha=0.3)
    ax.set_title("Local validation, fold 1: intervals overlap heavily"); fig.tight_layout(); fig.savefig(PL / "fold1_ci.png", dpi=130); plt.close(fig)
    for n in order_names[::-1]:
        r = res[n]; print(f"{r['wp']:.5f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}] d={r['delta_vs_ctrl_s3']:+.4f} [{r['delta_ci95'][0]:+.4f},{r['delta_ci95'][1]:+.4f}] {r['label']}")


if __name__ == "__main__":
    lr_curves(); curves(); ladder(); hpo(); ci_table()
