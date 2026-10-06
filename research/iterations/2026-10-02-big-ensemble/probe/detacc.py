import sys,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"../v3");import wnn_cpp
print([a for a in dir(wnn_cpp) if not a.startswith("_")])
def load(n):
    with np.load(n) as d:return {k:np.ascontiguousarray(d[k],dtype=np.float32) for k in d.files}
dets=[wnn_cpp.Predictor(load(n),fast_activations=True) for n in ("det_a.npz","det_b.npz")]
t=np.load("/home/grinchenkoeg/WunderFunder-creative/cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"]
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
ok=0;n=0;conf=np.zeros((4,4),int)
for i in range(0,len(rg),3):
    X=np.column_stack([c.to_numpy() for c in pf.read_row_group(int(rg[i]),columns=fc).columns]).astype(np.float32)[:99]
    for d in dets:d.reset()
    probs=np.zeros(4)
    for s,row in enumerate(X):
        lg=np.concatenate([d.step(row) for d in dets]).astype(np.float64);e=np.exp(lg-lg.max())
        if s>=50:probs+=e/e.sum()
    c=int(np.argmax(probs));conf[cls[i],c]+=1
print(conf);print("acc",np.trace(conf)/conf.sum())
