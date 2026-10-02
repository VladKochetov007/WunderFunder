"""Root-invoked bounded delivery finisher. Stops at READY_RUNTIME_PLAN; never starts runtime."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,math,os,shutil,subprocess,time,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];N=HERE/'native-quality'
FLOOR=3*2**30
GATE=dict(native_incumbent_pooled_min=.0001,native_incumbent_each_fold_strict_min=0.,native_GPU_pooled_min=-.0001,native_GPU_each_fold_min=-.00015)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def read(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).relative_to(ROOT))
def safe(s):
 p=Path(s);assert not p.is_absolute() and '..' not in p.parts;return ROOT/p
def guarded(config,ch):
 assert sha(config)==ch;c=read(config);assert c['execution_authorized'] is True and c['native_gate']==GATE
 assert c['check_CPU']==8 and c['check_RAM_bytes']==4*2**30
 assert c['sources'][rel(Path(__file__))]==sha(Path(__file__))
 for p,h in c['sources'].items():assert sha(safe(p))==h,('source changed',p)
 for key,p in [('training_allocation',HERE/'training-allocation.json'),('native_allocation',N/'allocation.json')]:
  b=c[key];assert safe(b['path']).resolve()==p.resolve() and sha(p)==b['sha256']
 return c

def report_gate(report):
 assert report['status']=='COMPLETE_FIXED_NATIVE_ORDINARY_WARM_COMPARISON'
 assert report['pure_sequences']==860 and report['pure_chains']==250 and report['scored_rows']==1614484
 assert report['models']==['ordinary_warm_native4','current_student_native4','ordinary_warm_GPU4'] and report['coefficients']==[.25]*4
 values={}
 for name in ('primary','native_minus_GPU_warm'):
  x=report['comparisons'][name]['scopes']['all'];values[name]={}
  for label,z in [('pooled',x),*[('fold'+str(f),x['folds'][str(f)]) for f in (0,1)]]:
   delta=float(z['delta_WP']);assert math.isfinite(delta) and all(z['candidate']['valid']) and all(z['control']['valid'])
   assert abs(delta-(z['candidate']['WP']-z['control']['WP']))<2e-14
   values[name][label]=delta
 p=values['primary'];g=values['native_minus_GPU_warm']
 checks=dict(native_incumbent_pooled=p['pooled']>=GATE['native_incumbent_pooled_min'],
  native_incumbent_both_folds=all(p['fold'+str(f)]>0 for f in (0,1)),
  native_GPU_pooled=g['pooled']>=GATE['native_GPU_pooled_min'],native_GPU_both_folds=all(g['fold'+str(f)]>=GATE['native_GPU_each_fold_min'] for f in (0,1)))
 return dict(gate=GATE,values=values,checks=checks,passed=all(checks.values()),CI_exclusion_required=False)

def wait_and_check(config,ch,c,deadline):
 fit=HERE/'execution-01/result.json';sup=N/'supervisor-result.json';report=N/'evaluation/report.json'
 while True:
  assert utc()<deadline,'completion waiting deadline'
  assert shutil.disk_usage(ROOT).free>FLOOR,'disk floor while waiting'
  guarded(config,ch)
  if fit.exists():
   r=read(fit);assert r['status']=='COMPLETED_2_ORDINARY_WARM_FINAL_FITS_NO_METRICS',('final fits failed',r.get('failure'))
  if sup.exists():
   n=read(sup);assert n['status']=='COMPLETE_TWO_OOF_ONE_FIXED_READOUT',('native pipeline failed',n.get('error'))
  if fit.exists() and sup.exists() and report.exists():break
  print(json.dumps(dict(utc=utc().isoformat(),status='WAITING_FINAL_AND_NATIVE',fit=fit.exists(),native=sup.exists())),flush=True);time.sleep(min(20,max(0.01,(deadline-utc()).total_seconds())))
 r=read(fit);n=read(sup);q=read(report);a=read(HERE/'training-allocation.json');na=read(N/'allocation.json');commands=read(HERE/'commands.json')
 assert r['allocation_sha256']==c['training_allocation']['sha256'] and r['fits']==2 and r['OOF_jobs']==0 and r['metric_calls']==0
 assert [j['name'] for j in r['jobs']]==[j['name'] for j in commands['jobs']]
 for j in r['jobs']:assert j['exit_code']==0 and not j.get('cleanup_error') and not j['remaining_scope_pids']
 assert n['allocation_sha256']==c['native_allocation']['sha256'] and [x['name'] for x in n['stages']]==['prepare','fold0','fold1','evaluate']
 for x in n['stages']:assert x['exit_code']==0 and sha(N/(x['name']+'.log'))==x['log_sha256']
 assert q['allocation_sha256']==c['native_allocation']['sha256'] and q['source_sha256']==na['source_sha256'][rel(N/'evaluate.py')]
 for p,h in na['source_sha256'].items():assert sha(safe(p))==h
 provenance=dict(final_result_sha256=sha(fit),native_supervisor_sha256=sha(sup),native_report_sha256=sha(report))
 return provenance,report_gate(q)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--config-sha256',required=True);a=ap.parse_args()
 config=a.config.resolve();c=guarded(config,a.config_sha256);deadline=datetime.fromisoformat(c['deadline_utc']);assert deadline.tzinfo is not None and utc()<deadline
 out=HERE/'finish-delivery-01';out.mkdir(exist_ok=False);started=utc();rec=dict(status='WAITING',started_utc=started.isoformat(),PID=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),config_sha256=a.config_sha256,stages=[],runtime_started=False,retries=0)
 write(out/'started.json',rec);Q=None;active=None
 try:
  assert shutil.disk_usage(ROOT).free>FLOOR+64*2**20,'disk floor'
  provenance,gate=wait_and_check(config,a.config_sha256,c,deadline);rec.update(provenance=provenance,native_gate_result=gate)
  outcome_files={HERE/'execution-01/result.json':provenance['final_result_sha256'],N/'supervisor-result.json':provenance['native_supervisor_sha256'],N/'evaluation/report.json':provenance['native_report_sha256']}
  if not gate['passed']:rec['status']='NATIVE_GATE_FAILED_NO_PACKAGING';return
  spec=importlib.util.spec_from_file_location('owned_delivery_queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
  package=HERE/'package/warm_student_fixed4.zip';check=HERE/'runtime/package-check.json';runtime=HERE/'runtime/runtime-plan.json'
  for p in [package,package.with_suffix('.manifest.json'),check,runtime,HERE/'runtime/execution',HERE/'runtime/supervisor.json']:assert not p.exists() and not p.is_symlink(),('collision',str(p))
  hname='oct02_ow_final_s3h2';base=ROOT/'research/iterations/2026-10-01-continuation/student-runtime/package/student_fixed4.zip'
  stages=[('package',120,['.venv/bin/python',rel(HERE/'package_student.py'),'--base-zip',rel(base),'--base-manifest',rel(base.with_suffix('.manifest.json')),'--checkpoint',f'artifacts/runs/{hname}/model_ep1.pt','--checkpoint-receipt',rel(HERE/'execution-01'/f'{hname}.check.json'),'--completion',rel(HERE/'execution-01/result.json'),'--output',rel(package)]),
   ('check_package',120,['/home/grinchenkoeg/scorer_image/run.sh',rel(HERE/'runtime/check_package.py'),'--package',rel(package),'--cpu','8']),
   ('prepare_runtime',30,['.venv/bin/python',rel(HERE/'runtime/prepare_runtime.py'),'--package',rel(package),'--cpu',str(c['runtime_CPU']),'--smt',str(c['runtime_SMT']),'--hard-seconds',str(c['runtime_hard_seconds'])])]
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
  for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
  for k in ('PYTHONHOME','PYTHONPATH','WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S'):env.pop(k,None)
  for name,cap,argv in stages:
   guarded(config,a.config_sha256);assert utc()<deadline and shutil.disk_usage(ROOT).free>FLOOR+64*2**20
   assert all(sha(p)==h for p,h in outcome_files.items()),'completed outcome changed'
   assert all(sha(safe(k))==v for k,v in c['sources'].items())
   seconds=min(cap,(deadline-utc()).total_seconds()-10);assert seconds>1
   unit='ordinary-warm-delivery-'+name+'-'+uuid.uuid4().hex;assert not Q.H.scope(unit)[1]
   cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c','8','/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',*argv]
   row=dict(name=name,unit=unit,argv=cmd,started_utc=utc().isoformat());rec['stages'].append(row);active=unit;p=None;t=time.monotonic()
   try:
    with (out/(name+'.log')).open('x') as log:
     p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);row['PID']=p.pid
     try:row['start_ticks']=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
     except FileNotFoundError:row['start_ticks']=None
     row['returncode']=p.wait(timeout=seconds+10)
    assert row['returncode']==0,('stage failed',name,row['returncode'])
   finally:
    fields,pids=Q.H.scope(unit)
    if pids or (p is not None and p.poll() is None):
     Q.H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=Q.H.scope(unit)
     if pids:Q.H.command(['systemctl','--user','kill','--signal=KILL',unit+'.scope'])
    if p is not None and p.poll() is None:p.wait(timeout=10)
    fields,pids=Q.H.scope(unit);row.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-t,finished_utc=utc().isoformat(),log_sha256=sha(out/(name+'.log')));active=None
    assert not pids,('owned processes remain',unit)
   guarded(config,a.config_sha256);assert shutil.disk_usage(ROOT).free>FLOOR
   assert all(sha(p)==h for p,h in outcome_files.items()),'completed outcome changed'
  m=read(package.with_suffix('.manifest.json'));k=read(check);rp=read(runtime)
  assert k['status']=='PASS' and k['zip_sha256']==m['zip_sha256']==sha(package)
  assert rp['execution_authorized'] is False and rp['artifact']['sha256']==m['zip_sha256']
  assert not (HERE/'runtime/execution').exists() and not (HERE/'runtime/supervisor.json').exists()
  rec.update(status='READY_RUNTIME_PLAN',package=dict(path=rel(package),sha256=sha(package)),manifest_sha256=sha(package.with_suffix('.manifest.json')),package_check_sha256=sha(check),runtime_plan=dict(path=rel(runtime),sha256=sha(runtime)),runtime_authorization_created=False)
 except BaseException as exc:rec.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
 finally:
  if active and Q is not None:
   try:Q.H.command(['systemctl','--user','stop',active+'.scope'])
   except BaseException as exc:rec['cleanup_failure']=repr(exc)
  rec.update(finished_utc=utc().isoformat(),elapsed_seconds=(utc()-started).total_seconds(),free_disk_bytes=shutil.disk_usage(ROOT).free)
  write(out/'result.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
