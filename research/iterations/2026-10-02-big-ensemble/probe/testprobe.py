import sys,importlib.util,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
def mod(d):
    spec=importlib.util.spec_from_file_location("sol_"+d.replace("/","_"),d+"/solution.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.PredictionModel()
t=np.load("/home/grinchenkoeg/WunderFunder-creative/cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"]
pick=[int(np.where(cls==c)[0][0]) for c in range(4)]
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
models={d:mod(d) for d in ("../run_mid","p_c3zero","p_c3only")}
for i in pick:
    X=np.column_stack([c.to_numpy() for c in pf.read_row_group(int(rg[i]),columns=fc).columns]).astype(np.float32)[:3000]
    outs={}
    for d,m in models.items():
        o=[]
        for s,row in enumerate(X):
            r=m.predict(NS(seq_ix=int(rg[i]),step_in_seq=s,need_prediction=s>=99,state=row))
            if r is not None:o.append(np.array(r))
        outs[d]=np.array(o)
    b=outs["../run_mid"]
    print("class",cls[i],"c3zero==base",np.array_equal(outs["p_c3zero"],b),"c3zero all0",not outs["p_c3zero"].any(),"| c3only==base",np.array_equal(outs["p_c3only"],b),"c3only all0",not outs["p_c3only"].any())
