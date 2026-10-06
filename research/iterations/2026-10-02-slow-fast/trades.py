import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
cls=np.load(R+"research/artifacts/market-classes-v1.npz")["valid"]
pf=pq.ParquetFile(R+"datasets/valid.parquet");names=pf.schema_arrow.names
cols=[f"i0_dp{k}" for k in range(4)]+[f"i0_dv{k}" for k in range(4)]
g=int(np.where(cls==3)[0][0])
X=np.column_stack([c.to_numpy() for c in pf.read_row_group(g,columns=cols).columns]).astype(float)
print("frac zero per col",(X==0).mean(0).round(3))
print("frac rows where col changes",(np.diff(X,axis=0)!=0).mean(0).round(3))
print("distinct values per col",[len(np.unique(X[:,k])) for k in range(8)])
print("mode value per col",[float(np.round(np.bincount(np.unique(X[:,k],return_inverse=True)[1]).argmax(),3)) for k in range(8)])
for k in range(8):
    u,c=np.unique(X[:,k],return_counts=True);o=np.argsort(-c)[:3];print(cols[k],"top values",np.round(u[o],3),"counts",c[o])
i=np.where(np.diff(X[:,4])!=0)[0][:5]
for r in i:print("row",r+1,np.round(X[r:r+3],3).tolist())
