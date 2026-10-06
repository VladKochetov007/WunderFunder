import sys,time,json,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"v3");import wnn_cpp
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet")
cols=pf.schema_arrow.names;fcols=cols[4:116]
X=np.concatenate([np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=fcols).columns]).astype(np.float32) for i in range(5)])
def load(n):
    with np.load(f"members/{n}.npz") as d: return {k:np.ascontiguousarray(d[k],dtype=np.float32) for k in d.files}
res={}
for n in sys.argv[1:]:
    e=wnn_cpp.Ensemble([load(n)],[1.0],fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto")
    t=time.perf_counter()
    for i,row in enumerate(X):
        if i%20000==0:e.reset()
        e.step(row)
    dt=time.perf_counter()-t;res[n]=dt/len(X)*39.4e6/60
    print(n,f"{res[n]:.1f} local-min projected",flush=True)
