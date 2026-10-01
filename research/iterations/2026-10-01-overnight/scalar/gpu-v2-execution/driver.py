"""Transparent generated-probe controller; no model or recipe changes."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def nvml():
 gpu=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=2).strip().split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=2)
 entries=[]
 for line in apps.splitlines():
  fields=[v.strip() for v in line.split(',')]
  if len(fields)==3 and fields[0]==gpu[0].strip():entries.append({'pid':int(fields[1]),'MiB':int(fields[2])})
 return {'uuid':gpu[0].strip(),'free_MiB':int(gpu[1]),'processes':entries}
def ident(pid):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();tail=raw[raw.rfind(')')+2:].split()
  return {'pid':pid,'start_ticks':int(tail[19]),'pgrp':int(tail[2]),'state':tail[0]}
 except FileNotFoundError:return None
def rss(pid):
 try:
  status=Path(f'/proc/{pid}/status').read_text().splitlines()
  return {line.split(':')[0]:int(line.split()[1]) for line in status if line.startswith(('VmRSS:','VmHWM:','Threads:'))}
 except FileNotFoundError:return {}
def group_memory(identity):
 members=[]
 for entry in Path('/proc').iterdir():
  if entry.name.isdigit():
   item=ident(int(entry.name))
   if item and item['pgrp']==identity['pgrp']:
    members.append(dict(item,**rss(item['pid'])))
 return {'members':members,'aggregate_RSS_KiB':sum(v.get('VmRSS',0) for v in members)}
def kill_owned(proc,identity):
 current=ident(proc.pid)
 if current is not None and all(current[k]==identity[k] for k in ['pid','start_ticks','pgrp']):os.killpg(proc.pid,signal.SIGKILL)
 elif proc.poll() is None:raise RuntimeError('refuse signal: owned process identity changed')
ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text())
assert alloc.get('SCALARPLE_generated_GPU_probe_authorized') is True and alloc.get('SCALARPLE_generated_GPU_probe_v2_authorized') is True
assert not alloc.get('competition_inputs_authorized',False)
for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
assert sha(P/'generated/gpu_fixture.npz')==alloc['fixture_sha256']
plan=json.loads((P/'gpu-v2-proposal.json').read_text())
assert alloc['proposal_sha256']==sha(P/'gpu-v2-proposal.json') and alloc['freeze_sha256']==sha(P/'GPU-V2-PREP-FREEZE.json')
assert alloc.get('physical_GPU',6)==6
out=P/'gpu-v2-execution';receipt=out/'execution.json';assert not receipt.exists()
start=time.monotonic();started=now();jobs=[];status='RUNNING';failure=None
try:
 for arm in ['control','ple']:
  for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
  assert sha(P/'generated/gpu_fixture.npz')==alloc['fixture_sha256']
  pre=nvml();assert pre['free_MiB']>=6144 and not pre['processes'],('GPU launch precondition',pre)
  output=P/f'gpu-v2-{arm}-attempt01.json';assert not output.exists()
  log=out/f'{arm}-attempt01.log';samples=out/f'{arm}-attempt01-samples.jsonl';assert not log.exists() and not samples.exists()
  argv=['taskset','-c','24-29',str(ROOT/'.venv/bin/python'),str(P/'gpu_probe_v2.py'),'--arm',arm,'--allocation',str(a.allocation.resolve()),'--output',str(output)]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',WNN_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
  begin=time.monotonic();f=log.open('w');proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);identity=ident(proc.pid);assert identity and identity['pgrp']==proc.pid
  job={'arm':arm,'started_UTC':now(),'argv':argv,'identity':identity,'prelaunch':pre,'resource_peaks':{'GPU_MiB':0,'RSS_KiB':0,'HWM_KiB':0,'Threads':0},'termination':None};jobs.append(job)
  # SIGALRM deadline interrupts the short resource query too. Signal only a
  # verified still-owned process group; no unrelated GPU process is signaled.
  def deadline_handler(signum,frame):
   job['termination']='deadline';kill_owned(proc,identity);raise TimeoutError('owned generated GPU arm/stage deadline')
  signal.signal(signal.SIGALRM,deadline_handler);signal.setitimer(signal.ITIMER_REAL,min(240.,600.-(begin-start)))
  try:
   with samples.open('w') as stream:
    while proc.poll() is None:
     state=nvml();memory=rss(proc.pid);group=group_memory(identity);own=[v for v in state['processes'] if v['pid']==proc.pid];foreign=[v for v in state['processes'] if v['pid']!=proc.pid]
     used=max([v['MiB'] for v in own],default=0)
     row={'UTC':now(),'elapsed':time.monotonic()-begin,'NVML':state,'process_status':memory,'process_group':group};stream.write(json.dumps(row)+'\n');stream.flush()
     peaks=job['resource_peaks'];peaks['GPU_MiB']=max(peaks['GPU_MiB'],used);peaks['RSS_KiB']=max(peaks['RSS_KiB'],group['aggregate_RSS_KiB']);peaks['HWM_KiB']=max(peaks['HWM_KiB'],memory.get('VmHWM',0));peaks['Threads']=max(peaks['Threads'],memory.get('Threads',0))
     if foreign or used>5632 or group['aggregate_RSS_KiB']*1024>8*(1<<30):
      job['termination']='foreign occupancy or resource bound';kill_owned(proc,identity);raise RuntimeError(job['termination'])
     time.sleep(.5)
   job['exit_code']=proc.wait();job['seconds']=time.monotonic()-begin
   assert job['exit_code']==0 and output.exists(),(arm,job['exit_code'])
   job['result_sha256']=sha(output);assert json.loads(output.read_text())['status']=='PASS'
  finally:
   signal.setitimer(signal.ITIMER_REAL,0)
   if proc.poll() is None:kill_owned(proc,identity)
   proc.wait();f.close();job['exit_code']=proc.returncode;job['seconds']=time.monotonic()-begin;job['finished_UTC']=now()
 control=json.loads((P/'gpu-v2-control-attempt01.json').read_text());ple=json.loads((P/'gpu-v2-ple-attempt01.json').read_text())
 estimate=1.25*15000*(2*control['median_wall_seconds']+3*ple['median_wall_seconds'])+5400
 at=datetime.datetime.now(datetime.timezone.utc);remaining=(datetime.datetime(2026,10,1,8,23,30,tzinfo=datetime.timezone.utc)-at).total_seconds()
 cost={'evaluated_UTC':at.isoformat(),'formula':plan['gate']['remaining_time_upper_bound_seconds'],'estimate_seconds':estimate,'remaining_seconds':remaining,'PASS':estimate<=remaining,'does_not_waive_parent_start_cutoffs':True}
 status='PASS' if cost['PASS'] else 'FAIL_COST_GATE'
except BaseException as exc:
 status='FAIL';failure=repr(exc);cost=None
finally:
 receipt.write_text(json.dumps({'status':status,'failure':failure,'started_UTC':started,'finished_UTC':now(),'seconds':time.monotonic()-start,'driver_sha256':sha(__file__),'allocation_sha256':sha(a.allocation),'proposal_sha256':sha(P/'gpu-v2-proposal.json'),'freeze_sha256':sha(P/'GPU-V2-PREP-FREEZE.json'),'jobs':jobs,'cost':cost,'monitor_limitations':'0.5s target sampling plus bounded nvidia-smi latency; sampled GPU/RSS can miss between-sample peaks. Torch allocator fraction enforced internally; SIGALRM deadline is OS-scheduled, not a cgroup hard real-time guarantee. Only owned PID/start/pgrp signaled.','competition_inputs_read':False},indent=2)+'\n')
 print(receipt.read_text(),flush=True)
if status!='PASS':sys.exit(1)
