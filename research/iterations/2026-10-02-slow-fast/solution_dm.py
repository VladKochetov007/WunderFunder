"""Fixed-members native GRU ensemble with two input views: raw rows, and rows whose additional features a0..a7
(state[104:112]) are replaced by value minus their causal running mean within the sequence (members trained with --a-mode demean)."""
import json,os,sys
from pathlib import Path
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
V3={"avx","avx2","bmi1","bmi2","f16c","fma","abm","movbe","xsave"}
AVX2={"avx","avx2","fma"}
def _level():
    flags=set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags=set(line.split(":",1)[1].split());break
    except OSError:pass
    return "v3" if flags>=V3 else "avx2" if flags>=AVX2 else "v2"
sys.path.insert(0,str(HERE/_level()))
import wnn_cpp
def _load(name):
    with np.load(HERE/name,allow_pickle=False) as d:return {k:np.ascontiguousarray(d[k],dtype=np.float32) for k in d.files}
def _ensemble(members):
    if not members:return None
    return wnn_cpp.Ensemble([_load(m["weights"]) for m in members],[float(m["coef"]) for m in members],
        fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto")
class PredictionModel:
    def __init__(self):
        members=json.loads((HERE/"members.json").read_text())
        total=sum(float(m["coef"]) for m in members)
        if abs(total-1.0)>1e-6:raise ValueError("member coefs must sum to 1")
        self.raw=_ensemble([m for m in members if m.get("a_mode","none")=="none"])
        self.dm=_ensemble([m for m in members if m.get("a_mode")=="demean"])
        self.seq_ix=None
    def predict(self,dp):
        if dp.seq_ix!=self.seq_ix:
            self.seq_ix=dp.seq_ix;self.n=0;self.asum=np.zeros(8,np.float64)
            if self.raw is not None:self.raw.reset()
            if self.dm is not None:self.dm.reset()
        row=np.ascontiguousarray(dp.state,dtype=np.float32)
        out=np.zeros(2,np.float32)
        if self.raw is not None:out=out+self.raw.step(row)
        if self.dm is not None:
            self.n+=1;a=row[104:112].astype(np.float64);self.asum+=a
            r2=row.copy();r2[104:112]=(a-self.asum/self.n).astype(np.float32)
            out=out+self.dm.step(r2)
        return out if dp.need_prediction else None
