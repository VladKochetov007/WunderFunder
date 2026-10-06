"""Full-sequence audit of exact and quantized organizer paths on clean fold0 chains."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import time
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave13'
_models=None;_pf=None

def worker(group):
    global _models,_pf
    if _models is None:
        sys.path.insert(0,str(D/'native-linear/build/v3'))
        import wnn_cpp
        w={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(D/'baseline-linear.npz').items()}
        old={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw/weights_5.npz').items()}
        configs={'fp32':(w,{}),'half':(w,{'half_weights':True}), 'int8':(w,{'dynamic_int8':True}),
                 'weight8':(w,{'int8_weights':True}), 'oldhalf':(old,{'half_weights':True})}
        _models={name:wnn_cpp.Ensemble([weights],[1.],fast_activations=True,**kw) for name,(weights,kw) in configs.items()}
        _pf=pq.ParquetFile(R/'datasets/valid.parquet')
    columns=_pf.schema_arrow.names[4:116]
    table=_pf.read_row_group(int(group),columns=columns,use_threads=False)
    x=np.ascontiguousarray(np.column_stack([c.to_numpy() for c in table.columns]),dtype=np.float32)
    result={};timing={}
    for name,model in _models.items():
        start=time.perf_counter();result[name]=model.run(x);timing[name]=time.perf_counter()-start
        assert np.isfinite(result[name]).all()
    return group,result,timing

def main():
    cf=np.load(R/'cache/chain_folds.npz')
    seen=set(c for s,c,f in zip(cf['split'],cf['chain'],cf['fold']) if s=='train' or (s=='valid' and f!=0))
    cm={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s=='valid'}
    z=np.load(R/'cache/valid_targets.npz');groups=z['row_group'];y=z['y'];mask=z['mask'];pos={int(g):i for i,g in enumerate(groups)}
    selected=[g for g in groups if cm[g][1]==0 and cm[g][0] not in seen]
    names=('fp32','half','int8','weight8','oldhalf')
    paths={n:R/f'cache/pred_w13native_{n}_valid.npy' for n in names};assert all(not p.exists() for p in paths.values())
    dest={n:np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(len(groups),20000,2)) for n,p in paths.items()}
    for arr in dest.values():arr[:]=np.nan
    timing={n:[] for n in names}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for j,(g,result,times) in enumerate(pool.map(worker,selected,chunksize=1)):
            for n in names:dest[n][pos[g]]=result[n];timing[n].append(times[n])
            if j%50==0:print('native sequences',j,flush=True)
    for arr in dest.values():arr.flush()
    del dest
    for p in paths.values():Path(str(p)+'.partial').rename(p)
    report={'sequences':len(selected),'chains':len(set(cm[g][0] for g in selected)),
        'median_cpu_seconds_per_sequence':{n:float(np.median(t)) for n,t in timing.items()},
        'scope':'Timing during eight-worker replay is diagnostic only; use sequential paired timing for package budget. All variants replay full sequences and reset at each boundary.'}
    (D/'native-replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
