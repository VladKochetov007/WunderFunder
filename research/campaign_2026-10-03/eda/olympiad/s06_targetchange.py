import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
df = pf.read_row_group(0).to_pandas()
N=len(df)
G = {"i0_pA":"i0_p0","i0_pB":"i0_p11","i1_pA":"i1_p0","i1_pB":"i1_p11","i0_dp":"i0_dp0","i0_dv":"i0_dv0","i1_dp":"i1_dp0","i1_dv":"i1_dv0","i0_vA":"i0_v0","i0_vB":"i0_v11","i1_vA":"i1_v0","i1_vB":"i1_v11","a7":"a7"}
ch = {g: np.r_[False, np.diff(df[c].values)!=0] for g,c in G.items()}  # ch[g][t]: changed between t-1 and t
for tg in ["t0","t1"]:
    tc = np.r_[False, np.diff(df[tg].values)!=0]
    idx = np.where(tc)[0]
    print(tg, "n changes", len(idx))
    for g in G:
        c = ch[g]
        best=[]
        for H in list(range(0,12))+[15,20,25,30,40,50,60,75,90,100,120,150,200,250,300,400,500,1000]:
            ok = c[idx] | np.r_[c, np.zeros(H+1,bool)][idx+H]
            best.append((H, round(ok.mean(),3)))
        best.sort(key=lambda z:-z[1])
        print("  ", g, "rate", round(c.mean(),3), "top", best[:4])
