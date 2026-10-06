import importlib.util,numpy as np,pyarrow.parquet as pq,sys
from types import SimpleNamespace as NS
def mod(d):
    s=importlib.util.spec_from_file_location("m"+d,d+"/solution.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.PredictionModel()
t=np.load("/home/grinchenkoeg/WunderFunder-creative/cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"]
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
base=mod("../run_mid");m=mod("p_c3x070")
for c in (2,3):
    i=int(np.where(cls==c)[0][1]);X=np.column_stack([x.to_numpy() for x in pf.read_row_group(int(rg[i]),columns=fc).columns]).astype(np.float32)[:2000]
    a=[];b=[]
    for s,row in enumerate(X):
        dp=NS(seq_ix=int(rg[i]),step_in_seq=s,need_prediction=s>=99,state=row);r1=base.predict(dp);r2=m.predict(dp)
        if r1 is not None:a.append(r1);b.append(r2)
    a=np.array(a);b=np.array(b);print("class",c,"ratio",np.unique(np.round(b[a!=0]/a[a!=0],5)))
