"""Cross-fitted post-hoc test: does value rarity (shared-vocabulary columns) improve the saved student's weighted Pearson via a per-decile output scale?"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from utils import FEATURE_COLUMNS  # noqa: E402
from crossfold_blend import load_fold  # noqa: E402


def wp(y, p, m):
    w = np.abs(np.clip(y, -2, 2))[m]; yy = np.clip(y, -2, 2)[m]; pp = p[m]
    mu_y, mu_p = (w * yy).sum() / w.sum(), (w * pp).sum() / w.sum()
    c = (w * (yy - mu_y) * (pp - mu_p)).sum()
    return c / np.sqrt((w * (yy - mu_y) ** 2).sum() * (w * (pp - mu_p) ** 2).sum())


def main() -> int:
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    y, mask, comp, models = load_fold(ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl", 1, split, ROOT / "cache/valid_targets.npz")
    name, arr, order = next(v for v in models.values() if v[0] == "vt_f1_chunk500")
    P = np.asarray(arr[order], dtype=np.float32)
    groups = [int(g) for g in json.loads(split.read_text())["evaluation"]["valid"]]
    rng = np.random.default_rng(4)
    pt, pv = pq.ParquetFile(ROOT / "datasets/train.parquet"), pq.ParquetFile(ROOT / "datasets/valid.parquet")
    def cols(pf, g, n=None):
        t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
        return np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS])[:n].astype(np.float64)
    tr = [cols(pt, g, 4000) for g in rng.choice(pt.num_row_groups, 100, replace=False)]
    uniq = [np.unique(np.concatenate([s[:, j] for s in tr]), return_counts=True) for j in range(112)]
    lat = [j for j in range(112) if np.isin(tr[0][:, j], uniq[j][0]).mean() > 0.5 and len(uniq[j][0]) < 400000]
    pick = rng.choice(len(groups), 120, replace=False)
    R = np.zeros((len(pick), y.shape[1]), np.float32)
    for k, i in enumerate(pick):
        x = cols(pv, groups[i])
        r = np.zeros(len(x))
        for j in lat:
            u, n = uniq[j]
            pos = np.clip(np.searchsorted(u, x[:, j]), 0, len(u) - 1)
            r += np.log1p(np.where(u[pos] == x[:, j], n[pos], 0))
        R[k, :len(r)] = r[: y.shape[1]] / len(lat)
    Y, Pm, M = y[pick], P[pick], mask[pick]
    base = np.mean([wp(Y[:, :, t], Pm[:, :, t], M) for t in (0, 1)]) if Y.ndim == 3 else wp(Y, Pm, M)
    edges = np.quantile(R[M], np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(edges, R, side="right") - 1, 0, 9)
    out = Pm.copy(); half = (np.arange(len(pick)) % 2 == 0)
    for a, b in ((half, ~half), (~half, half)):
        for d in range(10):
            f = (dec[a] == d) & M[a]
            yy, pp = Y[a][f], Pm[a][f]  # (n,2)
            s = (np.abs(np.clip(yy, -2, 2)) * np.clip(yy, -2, 2) * pp).sum() / max((np.abs(np.clip(yy, -2, 2)) * pp * pp).sum(), 1e-9)
            out[b] = np.where((dec[b] == d)[..., None], Pm[b] * np.clip(s, 0.5, 2.0), out[b])
    new = np.mean([wp(Y[:, :, t], out[:, :, t], M) for t in (0, 1)]) if Y.ndim == 3 else wp(Y, out, M)
    print(f"{len(lat)} vocab columns; 120 sequences; WP base {base:.5f} -> rarity-decile scale (cross-fitted) {new:.5f} ({new - base:+.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
