"""Diagnostic 8: were rank transforms fit per split? Compare per-column clip fraction and quantiles, train vs valid."""
import sys,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"/home/grinchenkoeg/WunderFunder/src")
from wnn.data import FEATURE_COLUMNS
R="/home/grinchenkoeg/WunderFunder-creative/datasets/"
rng=np.random.default_rng(0)
def sample(split,n):
    pf=pq.ParquetFile(R+f"{split}.parquet");ids=rng.choice(pf.num_row_groups,n,replace=False)
    return np.concatenate([np.stack([pf.read_row_group(i,columns=FEATURE_COLUMNS).column(c).to_numpy() for c in FEATURE_COLUMNS],1)[99::4] for i in ids]).astype(np.float32)
T=sample("train",300);V=sample("valid",300)
q=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]
print("col  min/max train | valid | clipfrac train valid | q01 q50 q99 train vs valid | nuniq t v")
rows=[]
for j,c in enumerate(FEATURE_COLUMNS):
    t,v=T[:,j],V[:,j];mx=max(np.abs(t).max(),np.abs(v).max())
    ct=(np.abs(t)>=mx-1e-4).mean();cv=(np.abs(v)>=mx-1e-4).mean()
    qt=np.quantile(t,q);qv=np.quantile(v,q)
    rows.append((c,t.min(),t.max(),v.min(),v.max(),ct,cv,qt,qv,len(np.unique(t)),len(np.unique(v)),t.mean(),v.mean(),t.std(),v.std()))
for r in rows:
    print(f"{r[0]:8s} {r[1]:6.3f}/{r[2]:6.3f} | {r[3]:6.3f}/{r[4]:6.3f} | {r[5]:.3f} {r[6]:.3f} | {r[7][0]:6.3f} {r[7][3]:6.3f} {r[7][6]:6.3f} vs {r[8][0]:6.3f} {r[8][3]:6.3f} {r[8][6]:6.3f} | {r[9]} {r[10]} | mean {r[11]:+.3f} {r[12]:+.3f} std {r[13]:.3f} {r[14]:.3f}")
m=np.array([[r[11],r[12],r[13],r[14]] for r in rows])
print("\nmean |mean diff| %.4f, mean std ratio %.4f; overall marginal mean train %.4f valid %.4f, std %.4f %.4f"%(np.abs(m[:,0]-m[:,1]).mean(),(m[:,3]/m[:,2]).mean(),T.mean(),V.mean(),T.std(),V.std()))
