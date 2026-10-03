import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pf = pq.ParquetFile("datasets/train.parquet")
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
fig, ax = plt.subplots(2,4, figsize=(20,9))
picked = {0:[],1:[],2:[],3:[]}
rg = 0
while min(len(v) for v in picked.values()) < 10 and rg < 10607:
    t = pf.read_row_group(rg, columns=["t0","t1"])
    t0 = t.column("t0").to_numpy(); a = np.abs(t0[t0!=0]); 
    if len(a):
        c = int(np.argmin(np.abs(Q-a.min()))) if np.isclose(a.min(), Q, atol=1e-4).any() else None
        if c is not None and len(picked[c])<10: picked[c].append(rg)
    rg += 37
print(picked)
for c in range(4):
    K0=[];K1=[]; fd=[]
    for r in picked[c]:
        df = pf.read_row_group(r, columns=["i0_p0","i0_p11","i0_dp0","t0","t1"]).to_pandas()
        k0 = np.round(df.t0.values/Q[c]).astype(int); k1 = np.round(df.t1.values/Q[c]).astype(int)
        K0.append(k0); K1.append(k1)
        s = k0+k1
        g = pd.DataFrame({"a":df.i0_p0,"b":df.i0_p11,"s":s}).groupby(["a","b"]).s.nunique()
        fd.append((g>1).mean())
    K0=np.concatenate(K0); K1=np.concatenate(K1)
    print("cls",c,"q",Q[c],"P(k0+k1<0)",np.mean(K0+K1<0).round(3),"P(=0)",np.mean(K0+K1==0).round(3),"P(>0)",np.mean(K0+K1>0).round(3),
          "mean k0",K0.mean().round(3),"mean k1",K1.mean().round(3), "frac states where k0+k1 not fn of (p0,p11):", np.round(np.mean(fd),3))
    vs = pd.Series(K0+K1).value_counts().sort_index(); print("   k0+k1 dist:", vs[(vs.index>=-8)&(vs.index<=6)].to_dict())
    lim = 8
    H, xe, ye = np.histogram2d(K0, K1, bins=np.arange(-lim-.5, lim+1.5))
    ax[0,c].imshow(np.log10(H.T+1), origin="lower", extent=[-lim-.5,lim+.5,-lim-.5,lim+.5]); ax[0,c].set_title(f"cls{c} q={Q[c]:.4f} log10 count (k0,k1)"); ax[0,c].set_xlabel("k0"); ax[0,c].set_ylabel("k1")
    ax[1,c].bar(vs.index, vs.values); ax[1,c].set_xlim(-12,8); ax[1,c].set_title(f"cls{c} dist of k0+k1")
plt.tight_layout(); plt.savefig("research/campaign_2026-10-03/eda/olympiad/f03_joint_k0k1.png", dpi=60)
