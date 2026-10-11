"""Cache-only distillation-headroom diagnostics on banked fold predictions (run where the predictions live).

usage: distill_ceiling_diag.py FOLD SINGLE ENS1+ENS2+... [PARENT]
 1. run-level oracle: single with each true-label-run mean replaced by the ensemble's run mean (non-causal bound for 'event-run level distillation')
 2. corr_w(ensemble-parent, y-parent) over scored rows (is the diverse-teacher delta predictable from labels at all?)
 3. channel split: ensemble delta applied only on the sum channel (p0+p1) or only on the difference channel (p0-p1)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import GlobalAccumulator  # noqa: E402

FROM = 99


def run_mean_replace(base: np.ndarray, level: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """base - run_mean(base) + run_mean(level) over runs of constant y (per sequence, per target)."""
    out = base.copy()
    for i in range(base.shape[0]):
        for t in range(2):
            ys, m = y[i, :, t], mask[i]
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            yy = ys[idx]
            starts = np.concatenate([[0], np.nonzero(np.diff(yy) != 0)[0] + 1])
            for a_, b_ in zip(starts, list(starts[1:]) + [idx.size]):
                rows = idx[a_:b_]
                out[i, rows, t] = base[i, rows, t] - base[i, rows, t].mean() + level[i, rows, t].mean()
    return out


def main() -> int:
    fold = int(sys.argv[1]); single = sys.argv[2]; ens = sys.argv[3].split("+"); parent = sys.argv[4] if len(sys.argv) > 4 else single
    split = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json").read_text())
    groups = [int(g) for g in split["evaluation"]["valid"]]
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as c:
        position = {int(g): i for i, g in enumerate(c["row_group"])}
        index = np.array(sorted(position[g] for g in groups))
        y = np.asarray(c["y"][index], dtype=np.float32)
        mask = np.asarray(c["mask"][index], dtype=bool)
    mask &= (np.arange(y.shape[1]) >= FROM)[None, :]
    mm = lambda n: np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r")
    S, E_list, Pn = mm(single), [mm(n) for n in ens], mm(parent)
    accs = {k: GlobalAccumulator() for k in ("single", "ens", "runlevel_swap", "runavg_single", "runavg_ens", "sum_only", "diff_only", "parent")}
    num = den1 = den2 = 0.0
    for s in range(0, len(index), 20):
        part = index[s:s + 20]; yy = y[s:s + 20]; m = mask[s:s + 20]
        sp = np.asarray(S[part], dtype=np.float32); ep = sum(np.asarray(e[part], dtype=np.float32) for e in E_list) / len(E_list)
        pp = np.asarray(Pn[part], dtype=np.float32)
        swap = run_mean_replace(sp, ep, yy, m)
        ravg_s = run_mean_replace(sp, sp, yy, m); ravg_e = run_mean_replace(ep, ep, yy, m)
        # run averages: replace by own mean -> base - mean(base) + mean(base) = base ; need the pure run average instead
        zero = np.zeros_like(sp)
        ravg_s = run_mean_replace(zero, sp, yy, m); ravg_e = run_mean_replace(zero, ep, yy, m)
        d = ep - sp
        ssum = sp + 0.5 * np.stack([d[..., 0] + d[..., 1]] * 2, -1)
        sdiff = sp + 0.5 * np.stack([d[..., 0] - d[..., 1], -(d[..., 0] - d[..., 1])], -1)
        for k, v in (("single", sp), ("ens", ep), ("runlevel_swap", swap), ("runavg_single", ravg_s), ("runavg_ens", ravg_e), ("sum_only", ssum), ("diff_only", sdiff), ("parent", pp)):
            accs[k].add(yy, v, m)
        w = np.abs(np.clip(yy, -2, 2)) * m[..., None]
        a, b = (ep - pp), (np.clip(yy, -2, 2) - pp)
        num += float((w * a * b).sum()); den1 += float((w * a * a).sum()); den2 += float((w * b * b).sum())
    for k, acc in accs.items():
        r = acc.result(); print(f"{k:15s} {r['wp']:.5f} t0 {r['t0']:.5f} t1 {r['t1']:.5f}", flush=True)
    print(f"weighted corr(ens-parent, y-parent) = {num / np.sqrt(den1 * den2 + 1e-30):.4f}  (uncentred, w=|clip y|)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
