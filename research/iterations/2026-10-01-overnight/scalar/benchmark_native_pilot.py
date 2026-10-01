"""Frozen full callbacks onCPU22; compare immutable baseline3 vs nonzero PLE prototype."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib,json,os,resource,sys,tempfile,time,zipfile
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--arm',choices=['baseline','prototype'],required=True);ap.add_argument('--rows',choices=[100000,1000000],type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output=a.output.resolve();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_native_pilot_authorized') is True
for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
assert not a.output.exists() and sorted(os.sched_getaffinity(0))==[22]
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
raw=P.parent/'critic/runtime-probe/raw.npy';assert sha(raw)=='1e90c52f1332950b70d573618adad29324fb6705675e74cb75c7ca6efda993b4'
package=json.loads((P/'native-pilot/package.json').read_text());proto_ref=json.loads((P/'native-pilot/reference-v3.json').read_text());assert proto_ref['prototype_zip_sha256']==package['zip_sha256']
archive=ROOT/'submissions/overnight_baseline3/baseline3_fixed.zip' if a.arm=='baseline' else P/'native-pilot/scalar_nonzero_prototype.zip'
reference=P.parent/'delivery-baseline3/reference-v3-real.npy' if a.arm=='baseline' else P/'native-pilot/prototype-reference-v3.npy'
assert sha(archive)==(package['baseline_zip_sha256'] if a.arm=='baseline' else package['zip_sha256'])
assert sha(reference)==(proto_ref['baseline_reference_sha256'] if a.arm=='baseline' else proto_ref['prototype_reference_sha256'])
def counters():return {line.split()[0]:list(map(int,line.split()[1:9])) for line in Path('/proc/stat').read_text().splitlines() if line.split()[0] in ('cpu22','cpu62')}
def activity(b,e):
 r={}
 for k in b:
  d=[y-x for x,y in zip(b[k],e[k])];total=sum(d);busy=total-d[3]-d[4];r[k]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
 return r
class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
sys.meta_path.insert(0,NoTorch())
with tempfile.TemporaryDirectory(prefix='scalar-pilot-') as tmp:
 d=Path(tmp)/'package';d.mkdir();cwd=Path(tmp)/'empty-cwd';cwd.mkdir();os.chdir(cwd);t=time.perf_counter()
 with zipfile.ZipFile(archive) as z:
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(d);archived_native_hash=hashlib.sha256(z.read('v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so')).hexdigest()
 extraction=time.perf_counter()-t;sys.path.insert(0,str(d));t=time.perf_counter();module=importlib.import_module('solution');model=module.PredictionModel();startup=time.perf_counter()-t
 assert module.LEVEL=='v3' and model.member_count==3
 assert sha(module.wnn_cpp.__file__)==archived_native_hash
 if a.arm=='prototype':assert archived_native_hash==alloc['sources'][str((P/'native_v2/build/v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so').relative_to(ROOT))]
 startup_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 x=np.load(raw,mmap_mode='r');expected=np.load(reference,mmap_mode='r');assert x.shape==(5,20000,112) and expected.shape==(5,20000,2)
 pre0=counters();time.sleep(2);pre=activity(pre0,counters());whole0=counters();sequences=[];digest=hashlib.sha256()
 for k in range(a.rows//20000):
  seq=np.asarray(x[k%5]);pred=np.zeros((20000,2),np.float32);point=SimpleNamespace(seq_ix=10000+k,step_in_seq=0,need_prediction=False,state=None);b=counters();c0=time.process_time();t=time.perf_counter()
  for j,row in enumerate(seq):
   point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
   if j<99:assert value is None
   else:pred[j]=value
  elapsed=time.perf_counter()-t;cpu=time.process_time()-c0;e=counters();ref=np.asarray(expected[k%5]).copy();ref[:99]=0
  assert np.isfinite(pred).all();assert pred.tobytes()==ref.tobytes(),'callback byte parity';digest.update(pred.tobytes());sequences.append({'index':k,'wall_seconds':elapsed,'cpu_seconds':cpu,'activity':activity(b,e)})
 whole=activity(whole0,counters());timed={}
 for key in ['cpu22','cpu62']:
  total=sum(s['activity'][key]['total_ticks'] for s in sequences);busy=sum(s['activity'][key]['busy_ticks'] for s in sequences);timed[key]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
 wall=sum(s['wall_seconds'] for s in sequences);cpu=sum(s['cpu_seconds'] for s in sequences);projection=wall/a.rows*39400000/60;quiet=pre['cpu62']['busy_fraction']<.05 and timed['cpu62']['busy_fraction']<.05
 status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line);assert int(status['Threads'])==1 and not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
 r={'status':'NOISY' if not quiet else 'PASS' if projection<=50 else 'FAIL_RUNTIME','arm':a.arm,'rows':a.rows,'wall_seconds':wall,'process_cpu_seconds':cpu,'cpu_to_wall_ratio':cpu/wall,'us_per_row':wall/a.rows*1e6,'projected_minutes_39_4M':projection,'quiet_SMT_below_5pct':quiet,'precheck_activity':pre,'timed_activity':timed,'whole_activity':whole,'per_sequence':sequences,'startup_seconds':startup,'extraction_seconds':extraction,'startup_max_rss_KiB':startup_rss,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threads':int(status['Threads']),'affinity':sorted(os.sched_getaffinity(0)),'native_sha256':sha(module.wnn_cpp.__file__),'zip_sha256':sha(archive),'raw_sha256':sha(raw),'reference_sha256':sha(reference),'prediction_sha256':digest.hexdigest(),'callback_vs_direct_reference_exact':True,'full_callbacks_including_masked_rows':True,'source_sha256':sha(__file__),'allocation_sha256':sha(a.allocation),'targets_read':False,'WP_queries':0,'torch_runtime_absent':True,'policy':'fixed3 equal1/3, baseline immutable; prototype nontrained firstmember PLE'}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','arm','rows','us_per_row','projected_minutes_39_4M','quiet_SMT_below_5pct','prediction_sha256']}))
