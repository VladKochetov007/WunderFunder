# Per-sequence scan of valid: target lattice, event indicators, is_scored conditional rates.
import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
import pyarrow.parquet as pq
n = int(sys.argv[1]); out = sys.argv[2]
pf = pq.ParquetFile(VALID)
rng = np.random.default_rng(0); gs = np.sort(rng.choice(pf.metadata.num_row_groups, n, replace=False))
rows = []; ev_tab = []
for g in gs:
    t = pf.read_row_group(g).to_pandas()
    sc = t.is_scored.values
    c = lattice(t.t0.values, t.t1.values)
    X = t[FEATS].values
    ch = np.r_[[True], np.any(X[1:] != X[:-1], 1)]
    def changed(cs):
        v = t[cs].values; return np.r_[True, np.any(v[1:] != v[:-1], 1)]
    ev = {
        "i0p": changed(I0P), "i0v": changed(I0V), "i0dp": changed(I0DP), "i0dv": changed(I0DV),
        "i1p": changed(I1P), "i1v": changed(I1V), "i1dp": changed(I1DP), "i1dv": changed(I1DV),
        **{f"a{k}": changed([f"a{k}"]) for k in range(8)},
        "tchg": np.r_[True, (t.t0.values[1:] != t.t0.values[:-1]) | (t.t1.values[1:] != t.t1.values[:-1])],
    }
    m = t.step_in_seq.values >= 99
    r = dict(g=g, seq=t.seq_ix.iloc[0], c=c, frac_scored=sc[m].mean(), sd0=t.t0.std(), sd1=t.t1.std(),
             any_change=ch.mean(), corr01=np.corrcoef(t.t0, t.t1)[0, 1])
    for k, v in ev.items():
        r[f"rate_{k}"] = v[m].mean()
        r[f"P(sc|{k})"] = sc[m & v].mean() if (m & v).any() else np.nan
        r[f"P(sc|~{k})"] = sc[m & ~v].mean() if (m & ~v).any() else np.nan
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 100)
print(df[["seq", "c", "frac_scored", "sd0", "sd1", "corr01", "any_change"]].describe().T)
cc = [c for c in df.columns if c.startswith("rate_") or c.startswith("P(")]
print(df[cc].mean().to_string())
