"""Allocation-gated orchestration only; frozen exporter/predictor are unchanged."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, resource, subprocess, sys, time

D=Path(__file__).resolve().parent
Q=D.parent
ROOT=Q.parents[3]
EXPECTED='a98412b6903dd027acdcd92b4b1dbdfcb147347ed0d00a26b7c6532409d3cedf'
ALLOC=Q/'oof-allocation.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def record(path,value):
 with path.open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')

def identity(label,a):
 entries={}
 for rel,expected in a['input_identity'].items():
  path=ROOT/rel;st=path.stat()
  observed={'sha256':sha(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'resolved_path':str(path.resolve())}
  entries[rel]={'expected':expected,'observed':observed,'match':expected==observed}
 result={'utc':now(),'allocation_sha256':sha(ALLOC),'inputs':entries,'targets_or_masks_decoded':False}
 (D/f'input-identity-{label}.json').write_text(json.dumps(result,indent=2)+'\n')
 assert all(v['match'] for v in entries.values()),'Input identity mismatch'

def source_identity(a):
 assert sha(ALLOC)==EXPECTED
 freeze=Q/'FREEZE.json';assert sha(freeze)==a['freeze_sha256'];f=json.loads(freeze.read_text())
 entries={}
 for group in ['source_and_binaries','actor_receipts','independent_receipts']:
  for rel,expected in f[group].items():
   observed=sha(ROOT/rel);entries[rel]={'expected':expected,'observed':observed};assert observed==expected,rel
 assert sha(ROOT/f['zip'])==a['zip_sha256']==f['zip_sha256']
 for name,expected in a['sources'].items():assert sha(Q/name)==expected
 (D/'source-identity-before.json').write_text(json.dumps({'utc':now(),'allocation_sha256':EXPECTED,'freeze_sha256':sha(freeze),'entries':entries,'zip_sha256':a['zip_sha256'],'orchestrator_sha256':sha(__file__)},indent=2)+'\n')

def lane(fold,a):
 os.sched_setaffinity(0,a['resources']['fold_CPU_affinity'][str(fold)])
 cap=a['resources']['memory_GiB_per_lane_max']*(1<<30)
 resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 deadline=time.monotonic()+a['resources']['wall_seconds_per_lane_max']
 env=os.environ.copy()
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='1'
 env['CUDA_VISIBLE_DEVICES']='';env['PYTHONUNBUFFERED']='1'
 commands=[('prepare',[str(ROOT/'.venv/bin/python'),str(Q/'prepare_oof.py'),'--fold',str(fold),'--allocation',str(ALLOC)])]
 for mode in a['modes']:
  commands.append((mode,['/home/grinchenkoeg/scorer_image/run.sh',str(Q/'predict_oof.py'),'--manifest',str(Q/'oof'/f'fold{fold}'/f'{mode}.checkpoint.json')]))
 ledger=D/f'lane{fold}.jsonl'
 for mode,argv in commands:
  log=D/f'fold{fold}-{mode}.log';usage=D/f'fold{fold}-{mode}.time.txt'
  actual=['/usr/bin/time','-v','-o',str(usage),*argv]
  start=time.monotonic();record(ledger,{'event':'start','utc':now(),'mode':mode,'fold':fold,'argv':actual,'CPU_affinity':sorted(os.sched_getaffinity(0)),'RLIMIT_AS':resource.getrlimit(resource.RLIMIT_AS),'threads_environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}})
  with log.open('x') as f:
   try:r=subprocess.run(actual,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.monotonic()));code=r.returncode
   except subprocess.TimeoutExpired:code=124
  record(ledger,{'event':'end','utc':now(),'mode':mode,'fold':fold,'exit_code':code,'wall_seconds':time.monotonic()-start,'log_sha256':sha(log),'resource_receipt_sha256':sha(usage) if usage.exists() else None})
  assert code==0,f'{mode} fold{fold} failed: {code}'
 print(json.dumps({'fold':fold,'status':'COMPLETE','utc':now()}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--lane',type=int,choices=[0,1]);args=p.parse_args()
 assert sha(ALLOC)==EXPECTED;a=json.loads(ALLOC.read_text())
 assert a['Q3_heldfold_predictions_authorized'] and a['runtime_gate']=='PASS' and a['modes']==['quant','fp16'] and not a['avx2_identity_audit_authorized']
 if args.lane is not None:return lane(args.lane,a)
 os.sched_setaffinity(0,{36});start=time.monotonic();source_identity(a);identity('before',a)
 jobs=[]
 for fold in a['folds']:
  log=(D/f'lane{fold}-driver.log').open('x');argv=[sys.executable,str(Path(__file__).resolve()),'--lane',str(fold)]
  proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  jobs.append((fold,proc,log));record(D/'execution-ledger.jsonl',{'event':'lane_start','utc':now(),'fold':fold,'pid':proc.pid,'argv':argv})
 failed=False
 for fold,proc,log in jobs:
  try:code=proc.wait(timeout=max(.1,start+a['resources']['wall_seconds_per_lane_max']+30-time.monotonic()))
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,15);code=124
  log.close();record(D/'execution-ledger.jsonl',{'event':'lane_end','utc':now(),'fold':fold,'exit_code':code});failed|=code!=0
 identity('after',a)
 summary={'status':'FAIL' if failed else 'COMPLETE_PENDING_ROOT_PROVENANCE_AUDIT','utc':now(),'seconds':time.monotonic()-start,'allocation_sha256':EXPECTED,'orchestrator_sha256':sha(__file__),'targets_read':False,'masks_read':False,'metrics_computed':0,'AVX2_audit_performed':False}
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
 assert not failed

if __name__=='__main__':main()
