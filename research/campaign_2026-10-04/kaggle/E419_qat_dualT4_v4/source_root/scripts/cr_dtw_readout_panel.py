"""Build the pooled 4-fold OOF panel of the distilled GRU256 family plus causal event flags (dtw/readout agent).

Outputs (scratch dir, memory-mappable):
  ft.npy, par.npy   (S, T, 2) f32  pooled out-of-fold head-FT / parent predictions (fold k file for fold k sequences)
  E.npy             (S, T) bool    a6 == lower endpoint (quote-move event marker, feature only)
  meta.npz          fold (S,), cls (S,)
Rows follow cache/valid_targets.npz (row_group order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

ENDPOINT = np.float32(-5.199337482452393)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(ROOT / "cache/valid_targets.npz")
rg = z["row_group"]
assert (rg == np.arange(len(rg))).all()
S, T = z["mask"].shape
fold = np.full(S, -1, np.int64)
for k in range(4):
    sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json", ROOT)
    groups = np.asarray(sorted(sp.evaluation_groups), np.int64)
    assert (fold[groups] == -1).all()
    fold[groups] = k
print("fold sizes", np.bincount(fold[fold >= 0]), "unassigned", (fold < 0).sum())

for name, pat in (("ft", "pred_c20261001_ft_d256_s3_f{k}h_valid.npy"), ("par", "pred_c20261001_d256_s3_f{k}_valid.npy")):
    out = np.lib.format.open_memmap(OUT / f"{name}.npy", mode="w+", dtype=np.float32, shape=(S, T, 2))
    out[:] = np.nan
    for k in range(4):
        src = np.load(ROOT / "cache" / pat.format(k=k), mmap_mode="r")
        idx = np.flatnonzero(fold == k)
        for i in idx:
            out[i] = src[i]
    out.flush()
    del out

E = np.lib.format.open_memmap(OUT / "E.npy", mode="w+", dtype=bool, shape=(S, T))
f = pq.ParquetFile(ROOT / "datasets/valid.parquet")
for g in range(S):
    t = f.read_row_group(g, columns=["seq_ix", "step_in_seq", "a6"], use_threads=False)
    assert int(t.column("seq_ix")[0].as_py()) == int(z["seq_ix"][g])
    E[g] = t.column("a6").to_numpy().astype(np.float32) == ENDPOINT
E.flush()
np.savez(OUT / "meta.npz", fold=fold, cls=z["market_class"])
print("done", OUT)
