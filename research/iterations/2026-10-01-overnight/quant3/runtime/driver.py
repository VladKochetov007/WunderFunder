"""Execution ledger only: invoke frozen Q3 harnesses under root runtime allocation."""
from pathlib import Path
import datetime,hashlib,json,os,resource,subprocess,sys,time,traceback
ROOT=Path('/home/grinchenkoeg/WunderFunder-creative')
P=ROOT/'research/iterations/2026-10-01-overnight/quant3';OUT=P/'runtime'
assert sorted(os.sched_getaffinity(0))==[18]
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
start=time.monotonic();deadline=start+895
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
allocation_sha='3aea8e74620e07b10f920174bc139a959246cdff48c40b3f511cb45a9949517c'
assert sha(P/'runtime-allocation.json')==allocation_sha
freeze=json.loads((P/'FREEZE.json').read_text())
for group in ('source_and_binaries','actor_receipts','independent_receipts'):
 for path,expected in freeze[group].items():assert sha(ROOT/path)==expected,(group,path)
assert sha(ROOT/freeze['zip'])==freeze['zip_sha256']
assert not (OUT/'execution-ledger.jsonl').exists() and not (OUT/'summary.json').exists()
def log(event):
 event={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**event}
 with (OUT/'execution-ledger.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 print(json.dumps(event),flush=True)
def run(name,args,allow_three=False):
 command=['/home/grinchenkoeg/scorer_image/run.sh',str(P/args[0]),*map(str,args[1:])]
 log({'event':'started','name':name,'command':command,'remaining_seconds':deadline-time.monotonic()})
 began=time.monotonic()
 with (OUT/f'{name}.log').open('x') as f:
  result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()))
 log({'event':'finished','name':name,'returncode':result.returncode,'seconds':time.monotonic()-began,'log_sha256':sha(OUT/f'{name}.log')})
 if result.returncode!=0 and not(allow_three and result.returncode==3):raise RuntimeError(f'{name} failed with exit {result.returncode}')
def bench(backend,rows,repeat):
 for attempt in range(2):
  name=f'{backend}-{rows}-r{repeat}-attempt{attempt}';path=OUT/f'{name}.json'
  run(name,['benchmark_zip.py','--zip',ROOT/freeze['zip'],'--raw',raw,'--reference',OUT/'reference/quant.npy','--rows',rows,'--backend',backend,'--output',path],allow_three=True)
  receipt=json.loads(path.read_text());log({'event':'runtime_result','name':name,'status':receipt['status'],'projected_minutes':receipt.get('projected_minutes_39_4M'),'receipt_sha256':sha(path)})
  if receipt['status'] in ('NOISY','NOISY_PRECHECK'):
   if attempt==0:time.sleep(5);continue
   return {'status':'UNQUALIFIED_NOISY_RETRIES_EXHAUSTED','receipt':str(path)}
  assert receipt['status'] in ('PASS','FAIL_RUNTIME')
  return {'status':receipt['status'],'receipt':str(path),'projected_minutes':receipt['projected_minutes_39_4M']}
summary={'allocation_sha256':allocation_sha,'freeze_sha256':sha(P/'FREEZE.json'),'zip_sha256':freeze['zip_sha256'],'forecasting_fits':0,'targets_read':False,'WP_queries':0,'vnni':[],'avx2':[]}
raw=ROOT/freeze['empirical_fixture']['path']
try:
 log({'event':'precheck_pass','driver_sha256':sha(__file__),'allocation_sha256':allocation_sha,'source_zip_receipts_verified':True})
 run('reference',['make_runtime_reference.py','--original-zip','/home/grinchenkoeg/WunderFunder-cpp/submissions/candV1_ft2/candV1_ft2.zip','--old-native',ROOT/'research/iterations/2026-10-01-overnight/runtime-vnni-opt/native_block128/build/v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so','--raw',raw,'--output-dir',OUT/'reference'])
 for level in ('v3','avx2','v2'):
  run(f'correctness-{level}',['check_runtime_zip.py','--zip',ROOT/freeze['zip'],'--raw',raw,'--reference',OUT/'reference/quant.npy','--fp16-reference',OUT/'reference/fp16.npy','--level',level,'--output',OUT/f'correctness-{level}.json'])
  assert json.loads((OUT/f'correctness-{level}.json').read_text())['status']=='PASS'
 summary['correctness']='PASS'
 for rows,repeat in [(100000,0),(100000,1),(100000,2),(1000000,0)]:
  result=bench('vnni',rows,repeat);summary['vnni'].append(result)
  if result['status']!='PASS':raise RuntimeError('VNNI primary gate failed: '+result['status'])
 summary['vnni_runtime_gate']='PASS'
 for rows,repeat in [(100000,0),(100000,1),(100000,2),(1000000,0)]:
  result=bench('avx2',rows,repeat);summary['avx2'].append(result)
  if result['status']!='PASS':summary['avx2_runtime_gate']='UNQUALIFIED';break
 else:summary['avx2_runtime_gate']='PASS'
 summary['status']='VNNI_RUNTIME_PASS_NO_QUALITY_READOUT'
except BaseException as exc:
 summary['status']='STOPPED';summary['error']=type(exc).__name__+': '+str(exc);summary['traceback']=traceback.format_exc();log({'event':'stopped','error':summary['error']})
finally:
 summary['seconds']=time.monotonic()-start;summary['driver_sha256']=sha(__file__);summary['CPU18_released_after_process_exit']=True
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log({'event':'summary','status':summary['status'],'summary_sha256':sha(OUT/'summary.json'),'seconds':summary['seconds']})
if summary['status']=='STOPPED':sys.exit(1)
