import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
for rg in [0, 1, 500]:
    df = pf.read_row_group(rg).to_pandas()
    def ch(c): return np.r_[False, np.diff(df[c].values)!=0]
    A, B, DP, DV, VA, VB = ch("i0_p0"), ch("i0_p11"), ch("i0_dp0"), ch("i0_dv0"), ch("i0_v0"), ch("i0_v11")
    A1, B1, DP1, VA1, VB1 = ch("i1_p0"), ch("i1_p11"), ch("i1_dp0"), ch("i1_v0"), ch("i1_v11")
    T = ch("t0") | ch("t1")
    print("rg", rg, "A",A.sum(),"B",B.sum(),"A&B",(A&B).sum(),"A|B",(A|B).sum(),"DP",DP.sum(),"DP&(A|B)",(DP&(A|B)).sum(), "DV",DV.sum(), "VA",VA.sum(),"VB",VB.sum(), "VA|VB", (VA|VB).sum())
    print("   T",T.sum(),"T&A&B",(T&A&B).sum(), "T & ~DP", (T&~DP).sum())
    print("   i1: A1",A1.sum(),"B1",B1.sum(),"A1&B1",(A1&B1).sum(),"DP1",DP1.sum(), "VA1", VA1.sum(), "VB1", VB1.sum())
    any0 = A|B|DP|DV|VA|VB; any1 = A1|B1|DP1|VA1|VB1
    print("   any i0", any0.sum(), "any i1", any1.sum(), "both", (any0&any1).sum(), "neither", (~any0&~any1).sum())
    # what about a cols
    for c in ["a0","a1","a2","a3","a4","a5","a6","a7"]:
        x=ch(c); print("   ",c, x.sum(), "on rows with no i0/i1 change:", (x & ~any0 & ~any1).sum(), "/", (~any0&~any1).sum())
