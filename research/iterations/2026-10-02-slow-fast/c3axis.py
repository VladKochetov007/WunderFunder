import numpy as np,pyarrow.parquet as pq
exec(open("c3regime.py").read().split("tc=np.array")[0].replace("print(\"class3 seqs\"","_=(\"\"").replace("np.nanmedian(wps).round(3))","0)"))
feats=["i0_v12_nuniq","i0_dp0_std","a5_mean","a3_std"];sign=[1,1,-1,-1]
tc=np.array(st["cls"])==3;vc=np.array(sv["cls"])==3
T=np.column_stack([np.array(st[k],float)[tc] for k in feats]);V=np.column_stack([np.array(sv[k],float)[vc] for k in feats])
mu=T.mean(0);sd=T.std(0)
z=lambda A:((A-mu)/sd*sign).mean(1)
zt,zv=z(T),z(V)
print("activity z (train-standardized): train q10/50/90",np.percentile(zt,[10,50,90]).round(2)," valid",np.percentile(zv,[10,50,90]).round(2))
zi=z(np.column_stack([np.array([sv[k][rgi[rg[i]]] for i in idx],float) for k in feats]))
ok=~np.isnan(wps);b=np.polyfit(zi[ok],wps[ok],1);print("per-seq WP slope per z unit",b[0].round(4),"intercept",b[1].round(3))
for lo,hi in((-9,-1),(-1,-.5),(-.5,0),(0,.5),(.5,9)):
    m=ok&(zi>=lo)&(zi<hi);print(f"z in [{lo},{hi}) n={m.sum():3d} medianWP {np.median(wps[m]) if m.any() else np.nan:.3f}")
for k in feats:print(k,"train median",np.median(np.array(st[k],float)[tc]).round(3),"valid median",np.median(np.array(sv[k],float)[vc]).round(3))
