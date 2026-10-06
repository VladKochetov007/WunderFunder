"""Midpoint fixed4 native ensemble + online market-class detector (rows 50-98) and fixed per-class output scales (LB composition probe)."""
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
class PredictionModel:
    def __init__(self):
        members=json.loads((HERE/"members.json").read_text());cfg=json.loads((HERE/"classes.json").read_text())
        self.model=wnn_cpp.Ensemble([_load(m["weights"]) for m in members],[float(m["coef"]) for m in members],
            fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto")
        self.dets=[wnn_cpp.Predictor(_load(n),fast_activations=True) for n in cfg["detector"]]
        self.from_row,self.decide_row=int(cfg["from_row"]),int(cfg["decide_row"])
        self.scales={int(c):float(s) for c,s in cfg["scale"].items()}
        self.seq_ix=None
    def predict(self,dp):
        if dp.seq_ix!=self.seq_ix:
            self.seq_ix=dp.seq_ix;self.model.reset()
            for d in self.dets:d.reset()
            self.probs=np.zeros(4);self.scale=None;self.step=0
        row=np.ascontiguousarray(dp.state,dtype=np.float32)
        out=self.model.step(row)
        if self.scale is None:
            lg=np.concatenate([d.step(row) for d in self.dets]).astype(np.float64);e=np.exp(lg-lg.max())
            if self.step>=self.from_row:self.probs+=e/e.sum()
            elif not self.probs.any():self.probs[:]=e/e.sum()
            if dp.need_prediction or self.step+1>=self.decide_row:
                self.scale=self.scales.get(int(np.argmax(self.probs)),1.0)
        self.step+=1
        if not dp.need_prediction:return None
        return out*self.scale
