import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
import json
from pathlib import Path
import numpy as np
D=Path(__file__).parent
sys.path.insert(0,str(D/os.environ.get('NATIVE_VARIANT','native')/'build'/sys.argv[1]))
import wnn_cpp
probe=np.load(D/'width-probe.npz');x=probe['x'];report={}
for hidden in (128,256,384,512):
    w={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(D/f'benchmark_h{hidden}.npz').items()}
    outputs={}
    for backend in ('scalar','avx2','vnni'):
        model=wnn_cpp.Ensemble([w],[1.],fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend=backend)
        got=np.array([model.step(row) for row in x]);outputs[backend]=got
        model.reset();repeat=np.array([model.step(row) for row in x[:123]])
        assert np.array_equal(repeat,got[:123])
    assert all(np.array_equal(outputs['scalar'],v) for v in outputs.values())
    fp=wnn_cpp.Ensemble([w],[1.],fast_activations=True,half_weights=False,dynamic_int8=False,quant_backend='auto')
    fpout=np.array([fp.step(row) for row in x]);ref=probe[f'h{hidden}']
    error=float(np.max(abs(fpout-ref)));assert error<2e-5,error
    corr=[float(np.corrcoef(outputs['scalar'][:,k],ref[:,k])[0,1]) for k in range(2)]
    assert min(corr)>.999,corr
    report[str(hidden)]={'fp32_torch_max_abs':error,'int8_corr':corr,'backends_bit_identical':True,'reset_deterministic':True}
(D/f'width-parity-{sys.argv[1]}-{os.environ.get("NATIVE_VARIANT","native")}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
