"""Fixed 7-member native GRU ensemble (E7); deterministic CPU ISA selection, no time-based pruning."""
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
        weights=[];configs=[];coefs=[]
        for member in members:
            with np.load(HERE/member['weights'],allow_pickle=False) as data:
                weights.append({key:np.ascontiguousarray(data[key],dtype=np.float32) for key in data.files})
            configs.append(json.loads((HERE/member['feature_config']).read_text()) if member.get('feature_config') else None)
            coefs.append(float(member['coef']))
        kwargs=dict(fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend="auto")
        if any(c is not None for c in configs):kwargs['feature_configs']=configs
        self.model=wnn_cpp.Ensemble(weights,coefs,**kwargs)
        self.seq_ix=None
        self.member_count=len(members)
        self.quant_backend=wnn_cpp.quant_backend()
    def predict(self,data_point):
        if data_point.seq_ix!=self.seq_ix:
            self.seq_ix=data_point.seq_ix;self.model.reset()
        value=self.model.step(np.ascontiguousarray(data_point.state,dtype=np.float32))
        return value if data_point.need_prediction else None
