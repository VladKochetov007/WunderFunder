import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
sv=pq.read_table(R+"cache/seqstats_valid.parquet").to_pydict();st=pq.read_table(R+"cache/seqstats_train.parquet").to_pydict()
feats=["i0_v12_nuniq","i0_dp0_std","a5_mean","a3_std"];sign=np.array([1,1,-1,-1])
cls={"train":np.load(R+"research/artifacts/market-classes-v1.npz")["train"],"valid":np.load(R+"research/artifacts/market-classes-v1.npz")["valid"]}
tc=np.array(st["cls"])==3;T=np.column_stack([np.array(st[k],float)[tc] for k in feats]);mu,sd=T.mean(0),T.std(0)
Z={}
for name,s,n in (("train",st,10607),("valid",sv,1873)):
    z=np.zeros(n);rg=np.array(s["row_group"]);X=np.column_stack([np.array(s[k],float) for k in feats])
    z[rg]=np.nan_to_num(((X-mu)/sd*sign).mean(1));Z[name]=z
for beta in (0,1,2):
    W={}
    for sp in ("train","valid"):
        w=np.ones(len(Z[sp]));c3=cls[sp]==3;w[c3]=np.clip(np.exp(-beta*Z[sp][c3]),0.25,4);W[sp]=w
    allc3=np.r_[W["train"][cls["train"]==3],W["valid"][cls["valid"]==3]];m=allc3.mean()
    for sp in W:W[sp][cls[sp]==3]/=m
    np.savez(f"seqw_b{beta}.npz",**W)
    print("beta",beta,"class3 weight train median %.2f valid median %.2f  max %.2f"%(np.median(W["train"][cls["train"]==3]),np.median(W["valid"][cls["valid"]==3]),max(W["train"].max(),W["valid"].max())))
np.savez("quiet_z.npz",**Z)
