"""Single-condition deterministic execution-order benchmark, raw inputs/no score."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--members',default='0,1,2,3');p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',type=int,default=20000);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
spec=json.loads((a.package/'members.json').read_text());indices=[int(i) for i in a.members.split(',')]
weights=[dict(np.load(a.package/spec[i]['weights'])) for i in indices]
e=wnn_cpp.Ensemble(weights,[1/len(weights)]*len(weights),half_weights=True)
x=np.load(a.raw,mmap_mode='r');e.run(x[0,:300]);out=[];timing=[]
for seq in x:
 e.reset();c0=time.process_time();t0=time.perf_counter()
 y=np.array([e.step(row) for row in seq[:a.rows]])
 wall=time.perf_counter()-t0;cpu=time.process_time()-c0
 out.append(y);timing.append(dict(cpu_seconds=cpu,wall_seconds=wall))
out=np.stack(out);result=dict(members=indices,rows=int(out.shape[0]*out.shape[1]),timing=timing,
 cpu_us=sum(z['cpu_seconds'] for z in timing)/(out.shape[0]*out.shape[1])*1e6,
 wall_us=sum(z['wall_seconds'] for z in timing)/(out.shape[0]*out.shape[1])*1e6,
 prediction_sha256=hashlib.sha256(out.tobytes()).hexdigest(),native_sha256=hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),
 affinity=sorted(os.sched_getaffinity(0)),targets_read=False,WP_queries=0)
if a.reference:
 r=np.load(a.reference);r=r[:,:a.rows]
 # The critic fullcallback receipt stores zero for the99 warmup outputs.
 result['max_abs_vs_reference_after_warmup']=float(np.abs(r[:,99:]-out[:,99:]).max())
 result['bitwise_vs_reference_after_warmup']=bool(np.array_equal(r[:,99:],out[:,99:]))
 assert result['bitwise_vs_reference_after_warmup']
np.save(a.output.with_suffix('.npy'),out)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
