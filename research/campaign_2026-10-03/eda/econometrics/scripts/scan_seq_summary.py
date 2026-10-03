# Per-sequence summary over ALL valid sequences (or a train sample): lattice class, target moments,
# scored-row metric mass, i0 price-event rate, distinct book states, warm-up statistics (causal).
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pf = pq.ParquetFile(path)
has_sc = "is_scored" in pf.schema_arrow.names
G = pf.metadata.num_row_groups
gs = np.arange(G) if n >= G else np.sort(np.random.default_rng(11).choice(G, n, replace=False))
cols = ["seq_ix", "t0", "t1"] + (["is_scored"] if has_sc else []) + I0P + I0DV + I1P + ["a2", "a7"]
rows = []
for g in gs:
    t = pf.read_row_group(g, columns=cols).to_pandas()
    a, b = t.t0.values.astype(np.float64), t.t1.values.astype(np.float64)
    c = lattice(a, b)
    P = t[I0P].values
    ev = np.r_[False, np.any(P[1:] != P[:-1], 1)]
    tv = np.r_[False, np.any(t[I0DV].values[1:] != t[I0DV].values[:-1], 1)]
    e1 = np.r_[False, np.any(t[I1P].values[1:] != t[I1P].values[:-1], 1)]
    sc = t.is_scored.values if has_sc else (np.arange(len(t)) >= 99)
    ya, yb = np.clip(a, -2, 2)[sc], np.clip(b, -2, 2)[sc]
    w = 99
    r = dict(g=g, seq=t.seq_ix.iloc[0], c=c, n_sc=int(sc.sum()),
             mean0=a.mean(), mean1=b.mean(), sd0=a.std(), sd1=b.std(), corr01=np.corrcoef(a, b)[0, 1],
             sum_abs0=np.abs(ya).sum(), sum_abs1=np.abs(yb).sum(),
             mass0=(np.abs(ya) * ya ** 2).sum(), mass1=(np.abs(yb) * yb ** 2).sum(),
             zero0=(a == 0).mean(), ev_rate=ev.mean(), tr_rate=tv.mean(), i1ev_rate=e1.mean(),
             n_states=len(np.unique(P, axis=0)),
             wu_ev_rate=ev[:w].mean(), wu_states=len(np.unique(P[:w], axis=0)), wu_tr_rate=tv[:w].mean(),
             wu_i1ev=e1[:w].mean(),
             ev1000_states=len(np.unique(P[:1000], axis=0)), ev1000_rate=ev[:1000].mean(),
             p0_mode_frac=pd.Series(t.i0_p0.values).value_counts(normalize=True).iloc[0],
             a7_floor=(t.a7.values < -5).mean(), a2_mean=t.a2.mean())
    if has_sc:
        r.update(sc_ev=ev[sc].mean(), sc_tr=tv[sc].mean())
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(df.describe().T.to_string())
