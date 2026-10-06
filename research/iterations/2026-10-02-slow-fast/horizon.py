import numpy as np
from scipy.stats import spearmanr
R="/home/grinchenkoeg/WunderFunder-creative/"
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"];cls=t["market_class"]
Z=np.load("quiet_z.npz")["valid"]
idx=np.where(cls==3)[0];gap=[];chg=[];sc_gap=[];absy=[]
for i in idx:
    d=Y[i,:,0]-Y[i,:,1];c=np.r_[False,(np.diff(Y[i],axis=0)!=0).any(1)]
    pos=np.where(c)[0];chg.append(c.mean())
    # rows from each scored row to next target change
    nxt=np.searchsorted(pos,np.arange(20000),side="right");valid=nxt<len(pos)
    dist=np.where(valid,pos[np.minimum(nxt,len(pos)-1)]-np.arange(20000),np.nan)
    m=M[i]&valid;sc_gap.append(np.nanmedian(dist[m]) if m.any() else np.nan)
    absy.append(np.abs(np.clip(Y[i],-2,2))[M[i]].mean())
chg=np.array(chg);sc_gap=np.array(sc_gap);absy=np.array(absy);z=Z[idx]
print("spearman z vs target-change rate",round(spearmanr(z,chg).correlation,3)," vs median rows to next change (scored)",round(spearmanr(z,sc_gap,nan_policy='omit').correlation,3)," vs mean|y|",round(spearmanr(z,absy).correlation,3))
for lo,hi in((-9,-1),(-1,-.5),(-.5,9)):
    m=(z>=lo)&(z<hi);print(f"z[{lo},{hi}) n={m.sum()} change-rate {np.median(chg[m]):.3f} rows-to-next-change {np.nanmedian(sc_gap[m]):.0f} mean|y| {np.median(absy[m]):.3f}")
