"""Santander-style frequency signal: does the global frequency (rarity) of a row's values in the shared-vocabulary columns relate to |y| or to model error?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def cols(pf, g, n):
    t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
    return np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS])[:n].astype(np.float64)


def main() -> int:
    rng = np.random.default_rng(3)
    pt, pv = pq.ParquetFile(ROOT / "datasets/train.parquet"), pq.ParquetFile(ROOT / "datasets/valid.parquet")
    tr = [cols(pt, g, 6000) for g in rng.choice(pt.num_row_groups, 100, replace=False)]
    ref = tr[0]
    uniq = [np.unique(np.concatenate([s[:, j] for s in tr]), return_counts=True) for j in range(112)]
    reuse = np.array([np.isin(ref[:, j], uniq[j][0]).mean() for j in range(112)])
    lat = [j for j in range(112) if np.median([np.isin(cols(pv, g, 2000)[:, j], uniq[j][0]).mean() for g in (5, 700)]) > 0.5]
    print("shared-vocabulary columns:", len(lat))
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as c:
        groups, y, mask = c["row_group"], c["y"], c["mask"]
    pred = np.load(ROOT / "cache/pred_vt_f1_chunk500_valid.npy", mmap_mode="r")
    fold = np.array(__import__("json").load(open(ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json")).get("indices", []))
    rar, wy, err = [], [], []
    for idx in rng.choice(len(groups), 30, replace=False):
        x = cols(pv, groups[idx], 6000)
        r = np.zeros(len(x))
        for j in lat:
            u, n = uniq[j]
            pos = np.clip(np.searchsorted(u, x[:, j]), 0, len(u) - 1)
            r += np.log1p(np.where(u[pos] == x[:, j], n[pos], 0))
        m = mask[idx][:6000]
        rar.append(r[m] / len(lat)); wy.append(np.abs(np.clip(y[idx][:6000][m], -2, 2)).mean(-1) if y[idx].ndim > 1 else np.abs(np.clip(y[idx][:6000][m], -2, 2)))
    rar, wy = np.concatenate(rar), np.concatenate(wy)
    print(f"rows {len(rar)}; corr(rarity, |y|) = {np.corrcoef(rar, wy)[0, 1]:.4f}; decile mean |y|:", np.round([wy[(rar >= np.quantile(rar, q)) & (rar <= np.quantile(rar, q + 0.1))].mean() for q in np.arange(0, 1, 0.1)], 3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
