import sys,json,shutil,os,numpy as np,pyarrow.parquet as pq
from types import SimpleNamespace as NS
tag=sys.argv[1];d=f"run_{tag}"
os.chdir(d);sys.path.insert(0,os.getcwd())
from solution import PredictionModel
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet")
t=pf.read_row_group(5);names=t.schema.names;feat=[c for c in names if c not in ("seq_ix","step_in_seq","need_prediction","is_scored","t0","t1")]
X=np.stack([t.column(c).to_numpy() for c in feat],1).astype(np.float32)[:3000];seq=int(t.column("seq_ix")[0].as_py())
# full package
pm=PredictionModel();full=np.array([pm.predict(NS(seq_ix=seq,state=X[i],need_prediction=True)) for i in range(len(X))])
# baseline-only via fp32 group
mem=json.load(open("members.json"));cb=[m for m in mem if m.get("fp32")][0]["coef"]
base_native=None
q=np.load("/tmp/onnx_base_rg5.npy")

nb=pm.models[1];nb.reset();bn=np.array([nb.step(X[i]) for i in range(len(X))])/cb
print("baseline native fp32 vs onnx: maxdiff",np.abs(bn-q).max(),"corr",np.corrcoef(bn[99:,0],q[99:,0])[0,1])
pm.models[0].reset();pn=np.array([pm.models[0].step(X[i]) for i in range(len(X))])
print("full == pair + base:",np.abs(full-(pn+bn*cb)).max(),"base share std",(bn*cb).std(0),"pair std",pn.std(0))
