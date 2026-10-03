# Can the per-sequence tick quantum q (target scale) be inferred causally from the first t rows?
import pyarrow.parquet as pq, numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
P0=[f'i0_p{i}' for i in range(22)]; P1=[f'i1_p{i}' for i in range(22)]
DV0=[f'i0_dv{i}' for i in range(4)]
cols=P0+P1+DV0+['a2','a3','a5','a6','a7','t0','t1']
QV=np.array([0.12543,0.24133,0.46854,0.65687])
def stats(t,T):
    X=t[P0].values[:T]; X1=t[P1].values[:T]
    r={}
    r['bk0']=(np.diff(X,axis=0)!=0).any(1).mean(); r['bk1']=(np.diff(X1,axis=0)!=0).any(1).mean()
    r['tr0']=(np.diff(t[DV0].values[:T],axis=0)!=0).any(1).mean()
    r['nst']=len(np.unique(X[:,0]))/T
    for a in ['a5','a6','a7']:
        x=t[a].values[:T]; r[a+'fl']=(x<-5.19).mean(); r[a+'m']=x[x>-5.19].mean() if (x>-5.19).any() else -5
    r['a2']=t.a2.values[T-1]; r['a3']=t.a3.values[T-1]
    for j in [0,3,11,19]: r[f'p{j}m']=X[:,j].mean(); r[f'p{j}sd']=X[:,j].std()
    return r
def build(path,gs,Ts):
    pf=pq.ParquetFile(path); rows=[]
    for g in gs:
        t=pf.read_row_group(int(g),columns=cols).to_pandas()
        v=np.unique(np.abs(np.r_[np.diff(t.t0.values),np.diff(t.t1.values),t.t0.values,t.t1.values])); q=v[v>1e-6].min()
        lab=int(np.argmin(np.abs(QV-q)))
        for T in Ts:
            r=stats(t,T); r['T']=T; r['y']=lab; r['g']=g; rows.append(r)
    return pd.DataFrame(rows)
t0=time.time()
rng=np.random.default_rng(1)
tr=build('datasets/train.parquet',rng.choice(10607,300,replace=False),[100,300,1000,3000,10000])
va=build('datasets/valid.parquet',np.arange(0,1873,9),[100,300,1000,3000,10000])
print('built',round(time.time()-t0),'s; train class dist',tr[tr['T']==100].y.value_counts().to_dict(),'valid',va[va['T']==100].y.value_counts().to_dict())
F=[c for c in tr.columns if c not in ('y','g')]
clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1,random_state=0).fit(tr[F],tr.y)
for T in [100,300,1000,3000,10000]:
    s=va[va['T']==T]; print('T',T,'valid acc',round(clf.score(s[F],s.y),3))
pd.concat([tr.assign(split='train'),va.assign(split='valid')]).to_csv('research/campaign_2026-10-03/eda/hft/regime_stats.csv',index=False)
# simple 1-feature rules
s=va[va['T']==100]
print(s.groupby('y')[['bk0','tr0','nst','a7fl','a5fl','a6fl','p0sd']].median().round(3))
