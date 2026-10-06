"""Whole-package determinism, member-sum parity, size and runtime projection."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as Point
import numpy as np
import pyarrow.parquet as pq

def main():
    folder=Path(sys.argv[1]).resolve();sys.path.insert(0,str(folder))
    from solution import PredictionModel
    import wnn_cpp
    pf=pq.ParquetFile(Path.home()/'WunderFunder-creative/datasets/valid.parquet')
    cols=pf.schema_arrow.names[4:116]
    sequences=[np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=cols).columns]).astype('float32') for i in (0,5,12,55,123)]
    def predict(model,seqs):
        out=[]
        for s,x in enumerate(seqs):
            for i,row in enumerate(x):
                value=model.predict(Point(seq_ix=s,step_in_seq=i,need_prediction=i>=99,state=row))
                if i<99:assert value is None
                else:
                    assert np.asarray(value).shape==(2,) and np.isfinite(value).all()
                    out.append(value.copy())
        return np.asarray(out)
    model=PredictionModel();start=time.perf_counter();got=predict(model,sequences);elapsed=time.perf_counter()-start
    repeat=predict(PredictionModel(),[sequences[0]])
    assert np.array_equal(repeat,got[:len(repeat)])
    members=json.loads((folder/'members.json').read_text());summed=np.zeros_like(got)
    for m in members:
        with np.load(folder/m['weights']) as z:w={k:np.ascontiguousarray(z[k],dtype='float32') for k in z.files}
        exact=bool(m.get('fp32'))
        engine=wnn_cpp.Ensemble([w],[1.],fast_activations=True,half_weights=exact,dynamic_int8=not exact,quant_backend='auto')
        offset=0
        for x in sequences:
            engine.reset()
            for i,row in enumerate(x):
                v=engine.step(row)
                if i>=99:summed[offset]+=m['coef']*v;offset+=int(i>=99)
    err=float(np.max(np.abs(got-summed)));assert err<2e-6,err
    local=elapsed/sum(map(len,sequences))*39.4e6/60
    report={'sequences':len(sequences),'rows':sum(map(len,sequences)),'deterministic':True,'finite':True,
            'weighted_member_sum_max_abs':err,'measured_seconds':elapsed,'projected_local_minutes':local,
            'historical_public_ratio_range':[.40,.49], 'projected_public_minutes_range':[local*.40,local*.49],
            'runtime_caveat':'Projection from historical hosts, not a current scorer measurement or full-test runtime certification.'}
    report['runtime_screen_pass']=local*.49<50
    dest=Path(sys.argv[2]) if len(sys.argv)>2 else folder.parent/'submission-audit.json'
    dest.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
