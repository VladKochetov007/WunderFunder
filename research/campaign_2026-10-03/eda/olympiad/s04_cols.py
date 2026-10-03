import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
names = pf.schema_arrow.names
inp = [c for c in names if c not in ("seq_ix","step_in_seq","need_prediction","t0","t1")]
df = pf.read_row_group(0).to_pandas()
SENT = -5.199337
out=[]
for c in inp:
    x = df[c].values
    out.append((c, len(np.unique(x)), 1+np.sum(np.diff(x)!=0), np.mean(np.isclose(x,SENT)), np.mean(x==0), np.isnan(x).mean()))
o = pd.DataFrame(out, columns=["col","nuniq","nruns","frac_sent","frac0","fracnan"])
print(o.to_string())
# pooled value sets
for grp in ["_p","_v","_dp","_dv"]:
    for inst in ["i0","i1"]:
        cs=[c for c in inp if c.startswith(inst+grp) and c[len(inst+grp)].isdigit()]
        s=np.unique(df[cs].values.ravel())
        print(inst+grp, len(cs),"cols pooled nuniq", len(s), "sum of per-col nuniq", sum(len(np.unique(df[c])) for c in cs))
sp0=set(np.unique(df[[f"i0_p{k}" for k in range(22)]].values.ravel()).tolist())
sp1=set(np.unique(df[[f"i1_p{k}" for k in range(22)]].values.ravel()).tolist())
print("p i0∩i1", len(sp0&sp1), len(sp0), len(sp1))
sd=set(np.unique(df[[f"i0_dp{k}" for k in range(4)]].values.ravel()).tolist())
print("dp∩p", len(sd&sp0), len(sd))
