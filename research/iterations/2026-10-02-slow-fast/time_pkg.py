import sys,time,importlib.util,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
d=sys.argv[1];sys.path.insert(0,d)
spec=importlib.util.spec_from_file_location("sol",d+"/solution.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
seqs=[np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=fc).columns]).astype(np.float32) for i in range(5)]
model=m.PredictionModel();t=time.perf_counter();out=[]
for s,X in enumerate(seqs):
    for j,row in enumerate(X):
        r=model.predict(NS(seq_ix=s,step_in_seq=j,need_prediction=j>=99,state=row))
        if r is not None:out.append(r)
dt=time.perf_counter()-t;print(d,"projected local min",round(dt/(5*20000)*39.4e6/60,1),"finite",np.isfinite(np.array(out)).all())
m2=m.PredictionModel();o2=[]
for j,row in enumerate(seqs[0]):
    r=m2.predict(NS(seq_ix=0,step_in_seq=j,need_prediction=j>=99,state=row))
    if r is not None:o2.append(r)
print("deterministic",np.array_equal(np.array(o2),np.array(out[:len(o2)])))
