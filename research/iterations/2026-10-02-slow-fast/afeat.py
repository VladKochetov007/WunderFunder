import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
Z=np.load("quiet_z.npz")["valid"];cls=np.load(R+"research/artifacts/market-classes-v1.npz")["valid"]
pf=pq.ParquetFile(R+"datasets/valid.parquet");names=pf.schema_arrow.names
ac=[n for n in names if n[0]=="a" and n[1:].isdigit()]
c3=np.where(cls==3)[0];quiet=c3[np.argsort(Z[c3])[:3]];act=c3[np.argsort(Z[c3])[-3:]]
for lab,ids in(("quiet",quiet),("active",act)):
    for g in ids:
        t=pf.read_row_group(int(g),columns=ac+["i0_p0","i0_v0","i0_dp0","i0_dv0"]);X=np.column_stack([c.to_numpy() for c in t.columns]).astype(float)
        ch=(np.diff(X,axis=0)!=0).mean(0)
        print(lab,g,"z=%.2f"%Z[g],"a-mean",X[:,:8].mean(0).round(2),"a-std",X[:,:8].std(0).round(2),"chg a/p0/v0/dp0/dv0",ch.round(3))
