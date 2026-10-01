"""Bounded allocated fixed knotfit and native pilot; no model/quality selection."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,resource,signal,statistics,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();a.allocation=a.allocation.resolve();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_train_only_knots_authorized') is True and alloc.get('SCALARPLE_native_pilot_authorized') is True
assert alloc['proposal_sha256']==sha(P/'native-input-proposal.json') and alloc['freeze_sha256']==sha(P/'NATIVE-INPUT-FREEZE.json')
for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
out=P/'native-execution';out.mkdir(exist_ok=False);started=time.monotonic();jobs=[];pairs=[];status='RUNNING';failure=None;active=None;identity=None
limit=900
def ident(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();v=s[s.rfind(')')+2:].split();return (int(v[2]),int(v[19]))
 except FileNotFoundError:return None
def terminate_owned():
 if active is not None and active.poll() is None:
  assert ident(active.pid)==identity,'refuse unbound PID';os.killpg(active.pid,signal.SIGKILL)
def deadline(*args):terminate_owned();raise TimeoutError('fixed native/input stage or child deadline')
signal.signal(signal.SIGALRM,deadline)
def child(label,interpreter,script,args,cap):
 global active,identity
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 elapsed=time.monotonic()-started;assert elapsed<limit
 cmd=['taskset','-c','22',interpreter,str(P/script),*args]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 def bounds():resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
 log=out/f'{label}.log';assert not log.exists();f=log.open('w');begin=time.monotonic();active=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=bounds);identity=ident(active.pid);assert identity and identity[0]==active.pid
 job={'label':label,'argv':cmd,'PID':active.pid,'PGID':identity[0],'start_ticks':identity[1],'started_UTC':now(),'max_sampled_RSS_KiB':0};jobs.append(job);signal.setitimer(signal.ITIMER_REAL,min(cap,limit-elapsed))
 try:
  while active.poll() is None:
   try:
    st=Path(f'/proc/{active.pid}/status').read_text().splitlines();rss=[int(s.split()[1]) for s in st if s.startswith('VmRSS:')];job['max_sampled_RSS_KiB']=max([job['max_sampled_RSS_KiB'],*rss])
   except FileNotFoundError:pass
   time.sleep(.25)
  code=active.wait();assert code==0,(label,code)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);terminate_owned();active.wait();f.close();job.update(exit_code=active.returncode,seconds=time.monotonic()-begin,finished_UTC=now(),log_sha256=sha(log));active=None;identity=None
scorer='/home/grinchenkoeg/scorer_image/run.sh';py=str(ROOT/'.venv/bin/python');allocation=['--allocation',str(a.allocation)]
def pair(index,rows,order):
 for attempt in range(2):
  results={};names={}
  for arm in order:
   name=f'pair{index}-{rows}-{arm}-attempt{attempt}';path=out/(name+'.json');child(name,scorer,'benchmark_native_pilot.py',allocation+['--arm',arm,'--rows',str(rows),'--output',str(path)],150)
   results[arm]=json.loads(path.read_text());names[arm]=str(path.relative_to(ROOT))
   assert results[arm]['status']!='FAIL_RUNTIME',f'quiet absolute50min failure {name}'
  row={'index':index,'rows':rows,'attempt':attempt,'order':order,'receipts':names,'quiet':all(r['quiet_SMT_below_5pct'] for r in results.values()),'overhead_ratio':results['prototype']['wall_seconds']/results['baseline']['wall_seconds']};pairs.append(row)
  if row['quiet']:return row
 raise RuntimeError('declared noisy pair retry exhausted')
try:
 child('fit-knots',py,'fit_knots.py',allocation,480)
 child('prepare-package',scorer,'prepare_native_pilot.py',allocation,60)
 for level in ['v2','avx2','v3']:child('correctness-'+level,scorer,'prepare_native_reference.py',allocation+['--level',level],90)
 short=[pair(i,100000,order) for i,order in enumerate([['baseline','prototype'],['prototype','baseline'],['baseline','prototype']])]
 overhead=statistics.median(r['overhead_ratio'] for r in short);assert overhead<=1.05,('median_overhead_gate',overhead)
 long=pair(3,1000000,['prototype','baseline'])
 status='PASS_INPUT_AND_NATIVE_PILOT_NO_QUALITY_CLAIM'
except BaseException as exc:failure=repr(exc);status='FAIL'
finally:
 signal.setitimer(signal.ITIMER_REAL,0);terminate_owned()
 result={'status':status,'failure':failure,'finished_UTC':now(),'elapsed_seconds':time.monotonic()-started,'allocation_sha256':sha(a.allocation),'source_sha256':sha(__file__),'jobs':jobs,'pairs':pairs,'quiet_short_pair_median_ratio':statistics.median(r['overhead_ratio'] for r in pairs if r['quiet'] and r['rows']==100000) if any(r['quiet'] and r['rows']==100000 for r in pairs) else None,'new_forecast_fits':0,'WP_queries':0,'target_reads':0,'resource_limits':'CPU22 only child,4GiB perprocess addressspace,900s stage; identity-bound own-group signal, sampledRSS maymisspeaks; no GPU.','noisy_policy':'Only one whole-pair retry on prospectively defined SMTcontamination; allattempts retained. No performancefailure retries.'}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if status!='PASS_INPUT_AND_NATIVE_PILOT_NO_QUALITY_CLAIM':raise SystemExit(1)
