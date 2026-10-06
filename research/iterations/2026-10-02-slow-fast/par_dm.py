import sys,importlib.util,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
d=sys.argv[1];sys.path.insert(0,d)
spec=importlib.util.spec_from_file_location("sol",d+"/solution.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
R="/home/grinchenkoeg/WunderFunder-creative/"
t=np.load(R+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"]
ref=np.load(R+"cache/pred_findm128_s3h2_valid.npy",mmap_mode="r")
pf=pq.ParquetFile(R+"datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
model=m.PredictionModel()
for c in (0,3):
    i=int(np.where(cls==c)[0][0]);X=np.column_stack([x.to_numpy() for x in pf.read_row_group(int(rg[i]),columns=fc).columns]).astype(np.float32)
    o=[]
    for s,row in enumerate(X):
        r=model.predict(NS(seq_ix=int(rg[i]),step_in_seq=s,need_prediction=s>=99,state=row))
        if r is not None:o.append(r)
    o=np.array(o);rr=np.asarray(ref[i,99:])
    print("class",c,"max abs diff",np.abs(o-rr).max().round(5),"corr",np.corrcoef(o[:,0],rr[:,0])[0,1].round(6),"std",rr.std(0).round(3))
