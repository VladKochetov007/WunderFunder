"""Independent raw-only CPU process/wall/member/callback decomposition, no metric access."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--native',type=Path);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 sys.path.insert(0,str(a.native or a.package/'v3'))
 import wnn_cpp
 spec=json.loads((a.package/'members.json').read_text())
 weights=[{k:np.asarray(v,dtype=np.float32) for k,v in np.load(a.package/m['weights']).items()} for m in spec]
 coef=[m['coef'] for m in spec]
 raw=np.ascontiguousarray(np.load(a.raw,mmap_mode='r')[0,:12000])
 engines=[('ensemble_native',wnn_cpp.Ensemble(weights,coef,half_weights=True),'run')]
 engines += [(f'member{k}_h{w["inp.weight"].shape[0]}',wnn_cpp.Predictor(w,half_weights=True),'run') for k,w in enumerate(weights)]
 engines += [('ensemble_callback',wnn_cpp.Ensemble(weights,coef,half_weights=True),'step')]
 measurements=[];preds={}
 for repeat in range(3):
  order=engines if repeat%2==0 else list(reversed(engines))
  for name,engine,mode in order:
   engine.run(raw[:300]);engine.reset()
   c0=time.process_time();t0=time.perf_counter()
   if mode=='run':y=engine.run(raw)
   else:y=np.array([engine.step(row) for row in raw])
   wall=time.perf_counter()-t0;cpu=time.process_time()-c0
   if name in preds:assert np.array_equal(y,preds[name]),f'repeat output mismatch{name}'
   else:preds[name]=y
   measurements.append(dict(name=name,repeat=repeat,wall_us=wall/len(raw)*1e6,cpu_us=cpu/len(raw)*1e6))
   print(json.dumps(measurements[-1]),flush=True)
 np.testing.assert_array_equal(preds['ensemble_native'],preds['ensemble_callback'])
 mix=np.zeros_like(preds['ensemble_native'])
 for k,w in enumerate(weights):mix+=np.float32(coef[k])*preds[f'member{k}_h{w["inp.weight"].shape[0]}']
 np.testing.assert_array_equal(mix,preds['ensemble_native'])
 result=dict(measurements=measurements,rows=len(raw),affinity=sorted(os.sched_getaffinity(0)),python=sys.version,
             native_sha256=sha(wnn_cpp.__file__),raw_sha256=sha(a.raw),native_callback_and_member_sum_bitexact=True,
             scored_metrics=0,targets_read=False,masks_read=False)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
