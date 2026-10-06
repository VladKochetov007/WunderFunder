import numpy as np,pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
R="/home/grinchenkoeg/WunderFunder-creative/";D=R+"research/iterations/2026-10-02-slow-fast/"
st=pq.read_table(R+"cache/seqstats_train.parquet").to_pydict();sv=pq.read_table(R+"cache/seqstats_valid.parquet").to_pydict()
bad=("t0","t1","S_","D_","ev_target","scored","row_group","seq_ix")
keys=[k for k in st if not k.startswith(bad)]
Xt=np.column_stack([np.array(st[k],float) for k in keys]);Xv=np.column_stack([np.array(sv[k],float) for k in keys])
X=np.r_[Xt,Xv];y=np.r_[np.zeros(len(Xt)),np.ones(len(Xv))];p=np.zeros(len(X))
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_leaf_nodes=15).fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1]
from sklearn.metrics import roc_auc_score
print("train-vs-valid AUC",round(roc_auc_score(y,p),4),"features",len(keys))
pt=np.clip(p[:len(Xt)],1e-4,1-1e-4);odds=pt/(1-pt);w=odds**0.3;w=np.clip(w/w.mean(),0.25,3);w/=w.mean()
rg=np.array(st["row_group"]);W=np.ones(10607);W[rg]=w
cls=np.load(R+"research/artifacts/market-classes-v1.npz")["train"]
print("weight pct",np.percentile(w,[5,25,50,75,95]).round(2),"mean weight by class",[round(float(W[cls==c].mean()),2) for c in range(4)])
np.savez(D+"rep_advval.npz",train=W,valid=np.ones(1873))
