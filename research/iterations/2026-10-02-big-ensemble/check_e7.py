import sys,time,json,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
sys.path.insert(0,".");from solution import PredictionModel
import wnn_cpp
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet")
fc=pf.schema_arrow.names[4:116]
seqs=[np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=fc).columns]).astype(np.float32) for i in range(5)]
m=PredictionModel();t=time.perf_counter();out=[]
for s,X in enumerate(seqs):
    for j,row in enumerate(X):
        r=m.predict(NS(seq_ix=s,step_in_seq=j,need_prediction=j>=99,state=row))
        if r is not None: out.append(np.array(r))
dt=time.perf_counter()-t;out=np.array(out)
print("solution.py projected local min:",dt/(5*20000)*39.4e6/60)
mem=json.load(open("members.json"));ref=0
for mm in mem:
    with np.load(mm["weights"]) as d:w={k:np.ascontiguousarray(d[k],dtype=np.float32) for k in d.files}
    e=wnn_cpp.Ensemble([w],[1.0],fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto");o=[]
    for s,X in enumerate(seqs):
        e.reset()
        for j,row in enumerate(X):
            v=e.step(row)
            if j>=99:o.append(np.array(v))
    ref=ref+mm["coef"]*np.array(o)
print("max abs diff vs member avg:",np.abs(out-ref).max(),"pred std",out.std(0))
m2=PredictionModel();o2=[]
for j,row in enumerate(seqs[0]):
    r=m2.predict(NS(seq_ix=0,step_in_seq=j,need_prediction=j>=99,state=row))
    if r is not None:o2.append(np.array(r))
print("deterministic:",np.array_equal(np.array(o2),out[:len(o2)]))
