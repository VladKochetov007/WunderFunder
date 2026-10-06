"""Heterogeneous-ensemble distillation: two h256 causal GRUs, final all-data seeds 5 and 6, dynamic int8."""
import json
import os
import sys
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
V3={'avx','avx2','bmi1','bmi2','f16c','fma','abm','movbe','xsave'}
AVX2={'avx','avx2','fma'}
def _level():
    flags=set()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('flags'):
                    flags=set(line.split(':',1)[1].split());break
    except OSError:pass
    return 'v3' if flags>=V3 else 'avx2' if flags>=AVX2 else 'v2'
LEVEL=_level()
sys.path.insert(0,str(HERE/LEVEL))
import wnn_cpp
class PredictionModel:
    def __init__(self):
        members=json.loads((HERE/'members.json').read_text())
        self.models=[]
        for exact in (False,True):
            grp=[m for m in members if bool(m.get('fp32'))==exact]
            if not grp:continue
            weights=[]
            for member in grp:
                with np.load(HERE/member['weights'],allow_pickle=False) as data:
                    weights.append({key:np.ascontiguousarray(data[key],dtype=np.float32) for key in data.files})
            coefs=[float(m['coef']) for m in grp]
            kw=dict(fast_activations=True,half_weights=exact,dynamic_int8=not exact,quant_backend="auto")
            self.models.append(wnn_cpp.Ensemble(weights,coefs,**kw))
        self.seq_ix=None
        self.member_count=len(members)
        self.quant_backend=wnn_cpp.quant_backend()
    def predict(self,data_point):
        if data_point.seq_ix!=self.seq_ix:
            self.seq_ix=data_point.seq_ix
            for m in self.models:m.reset()
        x=np.ascontiguousarray(data_point.state,dtype=np.float32)
        value=self.models[0].step(x)
        for m in self.models[1:]:value=value+m.step(x)
        return value if data_point.need_prediction else None
