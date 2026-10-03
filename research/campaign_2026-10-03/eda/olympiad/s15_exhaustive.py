# Exhaustive checks over all sequences: (F2) target changes only at rows where i0 pA, pB and dp all change;
# per-seq stats for class identification; value sets of i0_p0 per class.
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys, collections
split = sys.argv[1]
pf = pq.ParquetFile(f"datasets/{split}.parquet")
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
rows = []; valsets = collections.defaultdict(set)
cols = ["i0_p0","i0_p11","i0_dp0","i0_dv0","i0_v0","i0_v11","i1_p0","i1_p11","t0","t1","a7"]
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols)
    d = {c: t.column(c).to_numpy() for c in cols}
    ch = {c: np.r_[False, d[c][1:]!=d[c][:-1]] for c in cols}
    T = ch["t0"] | ch["t1"]
    a = np.abs(np.r_[d["t0"], d["t1"]]); q = a[a>0].min()
    cls = int(np.argmin(np.abs(Q-q)))
    A, B, DP = ch["i0_p0"], ch["i0_p11"], ch["i0_dp0"]
    rows.append(dict(rg=rg, cls=cls, nT=T.sum(), T_notA=(T&~A).sum(), T_notB=(T&~B).sum(), T_notDP=(T&~DP).sum(),
        T_t0only=(ch["t0"]&~ch["t1"]).sum(), T_t1only=(ch["t1"]&~ch["t0"]).sum(),
        A=A.sum(), B=B.sum(), AB=(A&B).sum(), ABD=(A&B&DP).sum(), DP=DP.sum(), DV=ch["i0_dv0"].sum(),
        A1=ch["i1_p0"].sum(), B1=ch["i1_p11"].sum(), nuA=len(np.unique(d["i0_p0"])), nuB=len(np.unique(d["i0_p11"])),
        a7sent=np.mean(d["a7"]<-5)))
    if rg % 3 == 0:
        valsets[cls].update(np.unique(d["i0_p0"]).tolist())
    if rg % 1000 == 0: print(rg, rows[-1], flush=True)
r = pd.DataFrame(rows); r.to_parquet(f"research/campaign_2026-10-03/eda/olympiad/exh_{split}.parquet")
print("TOTAL target-change rows", r.nT.sum(), "not on A:", r.T_notA.sum(), "not on B:", r.T_notB.sum(), "not on DP:", r.T_notDP.sum())
print("seqs with any violation A/B/DP:", (r.T_notA>0).sum(), (r.T_notB>0).sum(), (r.T_notDP>0).sum())
print("t0-only changes", r.T_t0only.sum(), "t1-only", r.T_t1only.sum())
print(r.groupby("cls")[["A","B","AB","ABD","DP","DV","A1","B1","nuA","nuB","nT","a7sent"]].median())
print("frac AB/A by cls", r.assign(f=r.AB/r.A.clip(1)).groupby("cls").f.describe())
ks = sorted(valsets)
for i in ks:
    for j in ks:
        if i<j: print("p0 value-set overlap cls", i, j, len(valsets[i]&valsets[j]), "sizes", len(valsets[i]), len(valsets[j]))
