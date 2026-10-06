"""Portable Python 3.11 native-engine parity and throughput audit."""
import json
import os
from pathlib import Path
import sys
import time

def main():
    import numpy as np
    weights_path,probe_path,engine_dir,dest=sys.argv[1:5]
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    root=Path(engine_dir)
    if root.is_file():
        import tarfile
        with tarfile.open(root) as z:z.extractall('engine')
        root=Path('engine')
    if (root/'v3').is_dir():
        lines=Path('/proc/cpuinfo').read_text().splitlines()
        flags=set(next(line for line in lines if line.startswith('flags')).split(':',1)[1].split())
        v3={'avx','avx2','bmi1','bmi2','f16c','fma','abm','movbe','xsave'}
        level='v3' if flags>=v3 else 'avx2' if flags>={'avx','avx2','fma'} else 'v2'
        root=root/level
    sys.path.insert(0,str(root.resolve()))
    import wnn_cpp
    weights={k:np.ascontiguousarray(a,dtype=np.float32) for k,a in np.load(weights_path).items()}
    z=np.load(probe_path);x=z['x'];ref=z['reference'];mask=z['mask'].copy();mask[:99]=False
    y=np.clip(z['y'],-2,2).astype('float64');w=np.abs(y)*mask[:,None]
    def wp(p):
        p=np.clip(p,-2,2).astype('float64');W=w.sum(0)
        yc=y-(w*y).sum(0)/W;pc=p-(w*p).sum(0)/W
        return float(np.mean((w*yc*pc).sum(0)/np.sqrt((w*yc**2).sum(0)*(w*pc**2).sum(0))))
    report={'python':sys.version,'engine':str(root),'rows':len(x),'native':{}}
    for quant in (False,True):
        engine=wnn_cpp.Ensemble([weights],[1.0],fast_activations=True,half_weights=False,dynamic_int8=quant,quant_backend='auto')
        got=np.empty_like(ref);t0=time.perf_counter()
        for i,row in enumerate(x):got[i]=engine.step(np.ascontiguousarray(row))
        elapsed=time.perf_counter()-t0
        engine.reset();repeat=np.array([engine.step(np.ascontiguousarray(row)) for row in x[:300]])
        assert np.array_equal(got[:300],repeat)
        corr=[float(np.corrcoef(ref[:,j],got[:,j])[0,1]) for j in range(2)]
        report['native']['int8' if quant else 'fp32']={'max_abs':float(np.max(np.abs(ref-got))),'corr':corr,
            'wp_delta':wp(got)-wp(ref),'us_per_row':elapsed/len(x)*1e6,'deterministic':True}
        assert min(corr)>.999,report
    Path(dest).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__ == '__main__':
    main()
