import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option("display.width",250)
pf = pq.ParquetFile("datasets/train.parquet")
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
ex = pd.read_parquet("research/campaign_2026-10-03/eda/olympiad/targets_train.parquet")
ex["cls"] = [int(np.argmin(np.abs(Q-q))) for q in ex.q]
for c in [3,2,1,0]:
    rgs = ex[ex.cls==c].rg.values[::max(1,len(ex[ex.cls==c])//30)][:30]
    K=[]
    for r in rgs:
        t = pf.read_row_group(int(r), columns=["t0","t1"])
        K.append(np.c_[np.round(t.column("t0").to_numpy()/Q[c]), np.round(t.column("t1").to_numpy()/Q[c])].astype(int))
    K=np.concatenate(K)
    ct = pd.crosstab(np.clip(K[:,0],-6,6), np.clip(K[:,1],-6,6))
    print("cls",c,"rows",len(K)); print((ct/len(K)*1000).round(1).to_string())
    print("  both>0:", np.sum((K[:,0]>0)&(K[:,1]>0)), " k0>0&k1==0:", np.sum((K[:,0]>0)&(K[:,1]==0)), " k0==0&k1>0:", np.sum((K[:,0]==0)&(K[:,1]>0)))
