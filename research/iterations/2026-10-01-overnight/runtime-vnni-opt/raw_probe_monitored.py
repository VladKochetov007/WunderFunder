"""DYNINT8-01 raw-only fullcallback feasibility; no target columns or metrics."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['fp16','dynamic'],required=True);p.add_argument('--rows',type=int,default=20000);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
spec=json.loads((a.package/'members.json').read_text());weights=[dict(np.load(a.package/z['weights']))for z in spec];e=wnn_cpp.Ensemble(weights,[.25]*4,half_weights=True,dynamic_int8=a.mode=='dynamic')
def cpustat():
 return {line.split()[0]:list(map(int,line.split()[1:]))for line in Path('/proc/stat').read_text().splitlines()if line.startswith('cpu') and line.split()[0]!='cpu'}
x=np.load(a.raw,mmap_mode='r');e.run(x[0,:300]);out=[];timing=[];host_before=cpustat()
for seq in x:
 e.reset();c0=time.process_time();t0=time.perf_counter();y=np.array([e.step(row)for row in seq[:a.rows]]);timing.append(dict(cpu_seconds=time.process_time()-c0,wall_seconds=time.perf_counter()-t0));out.append(y)
host_after=cpustat();host_busy={}
for key in host_before:
 delta=[b-a for a,b in zip(host_before[key],host_after[key])];host_busy[key]=1-(delta[3]+delta[4])/sum(delta)if sum(delta)else None
out=np.stack(out);r=np.load(a.reference)[:,:a.rows];d=out-r;err=[]
for k in range(2):
 delta=d[:,99:,k].ravel();ref=r[:,99:,k].ravel();rmse=float(np.sqrt(np.mean(delta.astype('float64')**2)));err.append(dict(target=k,rmse=rmse,reference_std=float(np.std(ref)),relative_rmse=rmse/float(np.std(ref)),max_abs=float(np.abs(delta).max()),p99_abs=float(np.quantile(np.abs(delta),.99)),mean_bias=float(np.mean(delta))))
result=dict(mode=a.mode,rows=int(out.shape[0]*out.shape[1]),timing=timing,cpu_us=sum(z['cpu_seconds']for z in timing)/out.shape[0]/out.shape[1]*1e6,wall_us=sum(z['wall_seconds']for z in timing)/out.shape[0]/out.shape[1]*1e6,finite=bool(np.isfinite(out).all()),errors_after99=err,prediction_sha256=hashlib.sha256(out.tobytes()).hexdigest(),native_sha256=hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(r.tobytes()).hexdigest(),affinity=sorted(os.sched_getaffinity(0)),targets_read=False,WP_queries=0)
result['host_busy_fraction']=host_busy
result['numerical_feasible']=result['finite'] and all(z['relative_rmse']<=.03 and z['p99_abs']<=.05 and z['max_abs']<=.5 for z in err)
np.save(a.output.with_suffix('.npy'),out);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
