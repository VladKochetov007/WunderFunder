"""Root-allocated two-lane fixed SCALARPLE screen. No metrics, retries or model selection."""
from pathlib import Path
import argparse,concurrent.futures,copy,datetime,json,os,signal,subprocess,threading,time
from training_recipe import sha,enforce_cutoff
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def now():return datetime.datetime.now(datetime.timezone.utc)
def ident(pid):
 try:
  text=Path(f'/proc/{pid}/stat').read_text();v=text[text.rfind(')')+2:].split();return {'pid':pid,'pgid':int(v[2]),'start_ticks':int(v[19])}
 except (FileNotFoundError,ProcessLookupError):return None
def members(pgid):
 out=[]
 for directory in Path('/proc').iterdir():
  if not directory.name.isdigit():continue
  item=ident(int(directory.name))
  if item and item['pgid']==pgid:
   try:
    lines=(directory/'status').read_text().splitlines();item['rss_KiB']=next((int(x.split()[1]) for x in lines if x.startswith('VmRSS:')),0);out.append(item)
   except FileNotFoundError:pass
 return out
def gpu_state(index):
 values=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(', ')
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
 return {'uuid':values[0],'used_MiB':int(values[1]),'free_MiB':int(values[2]),'compute':[{'pid':int(v[1]),'MiB':int(v[2])} for row in processes if (v:=row.split(', '))[0]==values[0]]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();a.allocation=a.allocation.resolve();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_forecast_fit_authorized') is True and alloc.get('SCALARPLE_OOF_authorized') is True
 planpath=ROOT/alloc['training_plan_path'];assert sha(planpath)==alloc['training_plan_sha256'];plan=json.loads(planpath.read_text());assert alloc['sources']==plan['sources'];assert alloc['runs']==plan['runs'];assert alloc['lanes']==plan['lanes'] and alloc['limits']==plan['limits'];assert alloc.get('derived_dependency_allocations_authorized') is True
 def sources():
  for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 sources();out=ROOT/plan['execution_path'];out.mkdir(exist_ok=False);started=time.monotonic();stop=threading.Event();records=[];lock=threading.Lock();limits=plan['limits'];py=str(ROOT/'.venv/bin/python');owned={}
 def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
 t0=plan['cost']['control_median_seconds'];t1=plan['cost']['ple_median_seconds'];cost=1.25*15000*(2*t0+3*t1)+5400;deadline=datetime.datetime.fromisoformat(plan['deadline_UTC']);remaining=(deadline-now()).total_seconds();write(out/'cost-at-launch.json',{'utc':now().isoformat(),'formula':'1.25*15000*(2*control+3*ple)+5400','seconds':cost,'remaining_seconds':remaining,'PASS':cost<=remaining,'separate_final_reserve_seconds':4800});assert cost<=remaining
 def kill_owned(proc,identity):
  if proc.poll() is None:
   assert ident(proc.pid)==identity,'refuse reused group leader';os.killpg(proc.pid,signal.SIGKILL)
 def child(label,script,arguments,lane=None,uses_gpu=False,cap=600):
  sources();assert not stop.is_set();assert time.monotonic()-started<limits['stage_seconds'];assert now()<deadline-datetime.timedelta(seconds=4800),'final verification reserve reached'
  index=lane['gpu'] if lane else None;affinity=lane['cpus'] if lane else '38';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index) if uses_gpu else '',WNN_THREADS='4' if lane else '1',OMP_NUM_THREADS='4' if lane else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4' if lane else '1',NUMEXPR_NUM_THREADS='1',WNN_GPU_FRACTION='0.34375',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='2',PYTHONUNBUFFERED='1')
  pre=gpu_state(index) if uses_gpu else None
  if pre:assert pre['used_MiB']<=128 and pre['free_MiB']>=limits['gpu_MiB'] and not pre['compute'],pre
  command=['taskset','-c',affinity,py,str(P/script),*arguments];log=out/(label+'.log');assert not log.exists();stream=log.open('w');begin=time.monotonic();proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);identity=ident(proc.pid);assert identity and identity['pgid']==proc.pid
  record={'label':label,'argv':command,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','WNN_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','WNN_GPU_FRACTION','WNN_LOADER_WORKERS','WNN_LOADER_RAM_GB']},'identity':identity,'started_UTC':now().isoformat(),'cap_seconds':cap,'GPU_precheck':pre,'peak_sampled_aggregate_RSS_KiB':0,'peak_sampled_GPU_MiB':0};seen={proc.pid:identity};failure=None
  with lock:records.append(record);owned[label]=(proc,identity)
  samples=out/(label+'.samples.jsonl')
  try:
   with samples.open('w') as sf:
    while proc.poll() is None:
     rows=members(proc.pid);seen.update({r['pid']:{k:r[k] for k in ['pid','pgid','start_ticks']} for r in rows});rss=sum(r['rss_KiB'] for r in rows);g=gpu_state(index) if uses_gpu else None;elapsed=time.monotonic()-begin;record['peak_sampled_aggregate_RSS_KiB']=max(record['peak_sampled_aggregate_RSS_KiB'],rss)
     if g:record['peak_sampled_GPU_MiB']=max(record['peak_sampled_GPU_MiB'],g['used_MiB'])
     sf.write(json.dumps({'utc':now().isoformat(),'seconds':elapsed,'processes':rows,'aggregate_RSS_KiB':rss,'GPU':g})+'\n');sf.flush()
     assert rss<=limits['host_RSS_GiB_per_lane']*(1<<20),'sampled aggregate RSS bound'
     if g:assert g['used_MiB']<=limits['gpu_MiB'] and all(x['pid'] in seen for x in g['compute']),'GPU budget or foreign owner'
     assert elapsed<cap and time.monotonic()-started<limits['stage_seconds'],'deadline'
     assert now()<deadline-datetime.timedelta(seconds=4800),'final verification reserve reached'
     assert not stop.is_set(),'peer failed';time.sleep(.5)
   assert proc.wait()==0,(label,proc.returncode)
  except BaseException as exc:failure=repr(exc);stop.set();kill_owned(proc,identity);raise
  finally:
   if proc.poll() is None:kill_owned(proc,identity)
   proc.wait();stream.close()
   # Signal only descendants whose PID start identity was previously observed.
   for pid,item in seen.items():
    if pid!=proc.pid and ident(pid)==item:
     try:os.kill(pid,signal.SIGKILL)
     except ProcessLookupError:pass
   record.update(exit_code=proc.returncode,failure=failure,seconds=time.monotonic()-begin,finished_UTC=now().isoformat(),log_sha256=sha(log),samples_sha256=sha(samples));write(out/(label+'.process.json'),record)
   with lock:owned.pop(label,None)
 def derived(job,identity):
  value=copy.deepcopy(alloc);value['runs']={job['name']:job};value['root_allocation_sha256']=sha(a.allocation);value['input_identity']=identity;path=out/(job['name']+'.allocation.json');assert not path.exists();write(path,value);return path
 status='FAIL';failure=None
 try:
  before=out/'inputs-before.json';child('inputs-before','input_preflight.py',['--allocation',str(a.allocation),'--output',str(before)],cap=300);inputid={'receipt_path':str(before.relative_to(ROOT)),'receipt_sha256':sha(before)}
  def lane_run(lane):
   try:
    completed={};headjobs=[]
    for name in lane['runs']:
     job=copy.deepcopy(alloc['runs'][name]);enforce_cutoff(job);runpath=ROOT/'artifacts/runs'/name;assert not runpath.exists(),str(runpath)
     if job['stage']=='head':
      parent=completed[job['parent_run']];assert parent['status']=='PASS';job['parent']['sha256']=parent['checkpoint_sha256']
     path=derived(job,inputid);child(name,'train.py',['--allocation',str(path),'--run',name],lane,True,limits['parent_seconds'] if job['stage']=='parent' else limits['head_seconds'])
     checked=out/(name+'.integrity.json');child(name+'-check','check_fit.py',['--allocation',str(path),'--run',name,'--output',str(checked)],lane,False,120);completed[name]=json.loads(checked.read_text());assert completed[name]['status']=='PASS'
     if job['stage']=='head':job['endpoint_sha256']=completed[name]['checkpoint_sha256'];headjobs.append(job)
    for job in headjobs:
     path=out/(job['name']+'.oof-allocation.json');value=copy.deepcopy(alloc);value['runs']={job['name']:job};value['root_allocation_sha256']=sha(a.allocation);value['input_identity']=inputid;write(path,value);child(job['name']+'-OOF','predict_scalar.py',['--allocation',str(path),'--run',job['name']],lane,True,limits['prediction_seconds'])
    return completed
   except BaseException:
    stop.set();raise
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(lane_run,lane) for lane in alloc['lanes']]
   for f in futures:f.result()
  after=out/'inputs-after.json';child('inputs-after','input_preflight.py',['--allocation',str(a.allocation),'--output',str(after)],cap=300);assert json.loads(before.read_text())['files']==json.loads(after.read_text())['files'];status='PASS_FIXED_FITS_AND_OOF_PENDING_INDEPENDENT_AUDIT'
 except BaseException as exc:failure=repr(exc);stop.set()
 finally:
  with lock:
   for proc,identity in owned.values():kill_owned(proc,identity)
  write(out/'result.json',{'status':status,'failure':failure,'started_UTC':(now()-datetime.timedelta(seconds=time.monotonic()-started)).isoformat(),'finished_UTC':now().isoformat(),'seconds':time.monotonic()-started,'allocation_sha256':sha(a.allocation),'training_plan_sha256':sha(planpath),'jobs':records,'WP_queries':0,'monitor_limits':'RSS/GPU sampled at roughly0.5s plus query cost; may miss transients. Monotonic deadline sampled, not hard realtime. Own-group leader start identity required; only observed start-bound descendants signaled after leader exits. No unconditional unseen-orphan guarantee.'})
 if status!='PASS_FIXED_FITS_AND_OOF_PENDING_INDEPENDENT_AUDIT':raise SystemExit(1)
if __name__=='__main__':main()
