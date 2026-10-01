"""Two prospectively fixed fallback ISA traces; one fresh process per ISA, no retries."""
from pathlib import Path
from types import SimpleNamespace
import argparse,datetime,hashlib,importlib,io,json,os,resource,signal,sys,tempfile,time,zipfile
from unittest.mock import patch
ENTRY_START=time.perf_counter()
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def counters():return {p[0]:list(map(int,p[1:9])) for p in (s.split() for s in Path('/proc/stat').read_text().splitlines()) if p and p[0] in ('cpu18','cpu58')}
def delta(b,e):
 out={}
 for k in b:
  d=[y-x for x,y in zip(b[k],e[k])];total=sum(d);busy=total-d[3]-d[4]
  out[k]={'ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
 return out
def threads():
 ids=list(Path('/proc/self/task').iterdir())
 for t in ids:assert sorted(os.sched_getaffinity(int(t.name)))==[18]
 return len(ids)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def terminate(signum,frame):raise TimeoutError('External fixed deadline sent SIGTERM')
def main():
 p=argparse.ArgumentParser();p.add_argument('--level',choices=['avx2','v2'],required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--allocation-sha256',required=True);a=p.parse_args()
 start=time.perf_counter();assert sorted(os.sched_getaffinity(0))==[18]
 assert sha(a.allocation)==a.allocation_sha256;allocation=json.loads(a.allocation.read_text());plan=json.loads((HERE/'plan.json').read_text())
 assert allocation['fallback_runtime_authorized'] is True and allocation['plan_sha256']==sha(HERE/'plan.json') and allocation['attempts_per_level']==1
 assert allocation['CPU18_58_explicitly_released'] is True
 assert sha(ROOT/plan['full_audit_result'])==allocation['full_audit_result_sha256']
 assert (ROOT/plan['full_audit_result']).exists(),'Full audit must finish before these traces'
 final=json.loads((ROOT/plan['full_audit_result']).read_text());assert final['status'] in ('COMPLETE','FAILED_OR_INTERRUPTED')
 for name,expected in plan['source_sha256'].items():assert sha(ROOT/name)==expected,name
 cap=4*(1<<30);resource.setrlimit(resource.RLIMIT_AS,(cap,cap));assert sys.version_info[:2]==(3,11)
 out_dir=HERE/'execution';out_dir.mkdir(exist_ok=True);output=out_dir/(a.level+'.json');attempt=out_dir/(a.level+'-attempt.json');assert not output.exists()
 with attempt.open('x') as stream:json.dump({'utc':now(),'PID':os.getpid(),'plan_sha256':sha(HERE/'plan.json'),'allocation_sha256':a.allocation_sha256,'level':a.level,'attempt':1},stream,indent=2)
 signal.signal(signal.SIGTERM,terminate)
 record={'started_utc':now(),'PID':os.getpid(),'level':a.level,'status':'STARTED','plan_sha256':sha(HERE/'plan.json'),'allocation_sha256':a.allocation_sha256,'targets_read':False,'WP_queries':0,'attempt':1,'rows_completed':0}
 try:
  cg_rel=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2];cg=Path('/sys/fs/cgroup')/cg_rel.lstrip('/')
  limits={name:(cg/name).read_text().strip() for name in ('cpu.max','memory.max','memory.swap.max')}
  assert limits=={'cpu.max':'100000 100000','memory.max':str(cap),'memory.swap.max':'0'}
  record.update(cgroup=cg_rel,cgroup_limits=limits,python=sys.version,numpy=np.__version__)
  assert np.__version__=='2.2.6'
  package=ROOT/plan['zip'];raw_path=ROOT/plan['raw'];ref_path=ROOT/plan['reference'][a.level]['path']
  assert sha(package)==plan['zip_sha256'] and sha(raw_path)==plan['raw_sha256'] and sha(ref_path)==plan['reference'][a.level]['sha256']
  class NoTorch:
   def find_spec(self,fullname,path=None,target=None):
    if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
  sys.meta_path.insert(0,NoTorch());b=counters();time.sleep(2);record['precheck']=delta(b,counters())
  if record['precheck']['cpu58']['busy_fraction']>=.05:
   record['status']='NOISY_PRECHECK_NO_RETRY';return
  with tempfile.TemporaryDirectory(prefix='baseline3-fallback-isa-') as tmp:
   d=Path(tmp)/'package';cwd=Path(tmp)/'empty';d.mkdir();cwd.mkdir();os.chdir(cwd);t=time.perf_counter()
   with zipfile.ZipFile(package) as z:
    assert set(z.namelist())==set(plan['zip_files'])
    for name in z.namelist():
     body=z.read(name);assert hashlib.sha256(body).hexdigest()==plan['zip_files'][name]['sha256'];dest=d/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
   record['extract_seconds']=time.perf_counter()-t;sys.path.insert(0,str(d));original_open=open
   flags={'v2':'','avx2':'avx avx2 fma'}[a.level]
   def forced_open(path,*args,**kwargs):return io.StringIO('flags : '+flags+'\n') if str(path)=='/proc/cpuinfo' else original_open(path,*args,**kwargs)
   t=time.perf_counter()
   with patch('builtins.open',forced_open):solution=importlib.import_module('solution')
   model=solution.PredictionModel();record['startup_seconds']=time.perf_counter()-t
   assert solution.LEVEL==a.level and model.member_count==3 and len(model.model)==3
   assert sha(solution.wnn_cpp.__file__)==plan['zip_files'][a.level+'/wnn_cpp.cpython-311-x86_64-linux-gnu.so']['sha256']
   raw=np.load(raw_path,mmap_mode='r',allow_pickle=False);reference=np.load(ref_path,allow_pickle=False)
   assert raw.shape==(5,20000,112) and raw.dtype==np.float32 and reference.shape==(5,512,2) and reference.dtype==np.float32
   assert np.isfinite(raw).all()
   sequence=[];record['per_sequence']=sequence;digest=hashlib.sha256();whole=counters();max_threads=threads()
   for k,seq in enumerate(raw):
    y=np.zeros((20000,2),np.float32);point=SimpleNamespace(seq_ix=10000+k,step_in_seq=0,need_prediction=False,state=None)
    b=counters();cpu0=time.process_time();t=time.perf_counter()
    for j,row in enumerate(seq):
     point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
     if j<99:assert value is None
     else:assert value.shape==(2,) and value.dtype==np.float32;y[j]=value
    wall=time.perf_counter()-t;cpu=time.process_time()-cpu0;e=counters()
    assert np.isfinite(y).all();expected=reference[k].copy();expected[:99]=0;assert y[:512].dtype==expected.dtype and y[:512].shape==expected.shape and y[:512].tobytes()==expected.tobytes()
    digest.update(y.tobytes());max_threads=max(max_threads,threads())
    sequence.append({'sequence':k,'rows':20000,'wall_seconds':wall,'cpu_seconds':cpu,'activity':delta(b,e)})
    record['rows_completed']+=20000
    (out_dir/(a.level+'-progress.json')).write_text(json.dumps({'utc':now(),'rows_completed':record['rows_completed'],'sequence':sequence[-1]},indent=2)+'\n')
   whole_activity=delta(whole,counters());wall=sum(s['wall_seconds'] for s in sequence);cpu=sum(s['cpu_seconds'] for s in sequence)
   activity={}
   for key in ('cpu18','cpu58'):
    ticks=sum(s['activity'][key]['ticks'] for s in sequence);busy=sum(s['activity'][key]['busy_ticks'] for s in sequence)
    activity[key]={'ticks':ticks,'busy_ticks':busy,'busy_fraction':busy/ticks}
   assert sha(package)==plan['zip_sha256'] and sha(raw_path)==plan['raw_sha256'] and sha(ref_path)==plan['reference'][a.level]['sha256']
   for name,expected in plan['source_sha256'].items():assert sha(ROOT/name)==expected,name
   assert sha(a.allocation)==a.allocation_sha256 and sha(HERE/'plan.json')==record['plan_sha256']
   record.update(status='COMPLETE',callback_rows=100000,emitted_rows=99505,sequence_count=5,wall_seconds=wall,cpu_seconds=cpu,
    us_per_callback=wall/100000*1e6,projected_callback_minutes_39_4M=wall/100000*39400000/60,
    projection_at_most_50min=wall/100000*39400000/60<=50,projection_at_most_60min=wall/100000*39400000/60<=60,
    callback_activity=activity,whole_trace_activity=whole_activity,clean=activity['cpu58']['busy_fraction']<.05 and whole_activity['cpu58']['busy_fraction']<.05,
    per_sequence=sequence,archived_prefix_callback_slots_bitexact=2560,archived_emitted_prediction_rows_bitexact=2065,warmup_zero_placeholder_slots=495,all100k_finite=True,full100k_archived_parity_claimed=False,
    prediction_sha256=digest.hexdigest(),max_observed_threads=max_threads,native_sha256=sha(solution.wnn_cpp.__file__),
    forced_isa_harness_only=True,zip_sha256=sha(package),raw_sha256=sha(raw_path),reference_sha256=sha(ref_path))
   assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
 except BaseException as exc:record.update(status='FAILED',exception=type(exc).__name__+': '+str(exc));raise
 finally:
  record.update(ended_utc=now(),worker_main_seconds=time.perf_counter()-start,script_entry_seconds=time.perf_counter()-ENTRY_START,max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
