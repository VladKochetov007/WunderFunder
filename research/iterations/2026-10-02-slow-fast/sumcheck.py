import sys,json,importlib.util,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
d="run_final_muQ5";sys.path.insert(0,d)
spec=importlib.util.spec_from_file_location("sol",d+"/solution.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import wnn_cpp
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
X=np.column_stack([c.to_numpy() for c in pf.read_row_group(7,columns=fc).columns]).astype(np.float32)[:3000]
model=m.PredictionModel();full=[]
for j,row in enumerate(X):
    r=model.predict(NS(seq_ix=7,step_in_seq=j,need_prediction=j>=99,state=row))
    if r is not None:full.append(r)
full=np.array(full);mem=json.load(open(d+"/members.json"));ref=0
for mm in mem:
    e=wnn_cpp.Ensemble([m._load(mm["weights"])],[1.0],fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto")
    s=np.zeros(8);o=[]
    for j,row in enumerate(X):
        r=row.copy()
        if mm["a_mode"]=="demean":
            s+=row[104:112];r[104:112]=(row[104:112].astype(np.float64)-s/(j+1)).astype(np.float32)
        v=e.step(r)
        if j>=99:o.append(np.array(v))
    ref=ref+mm["coef"]*np.array(o)
print("max abs diff full vs weighted member sum:",np.abs(full-ref).max())
