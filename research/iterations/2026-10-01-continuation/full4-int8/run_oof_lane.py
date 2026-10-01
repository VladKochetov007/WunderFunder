"""One allocated CPU lane: export its fixed fold then quant and FP16, no fitting."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);a=ap.parse_args()
 allocation=json.loads((P/'allocation.json').read_text());assert allocation['Q4_heldfold_predictions_authorized'] is True
 cpu=allocation['OOF_lanes'][str(a.fold)];assert sorted(os.sched_getaffinity(0))==[cpu]
 rel=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2];cg=Path('/sys/fs/cgroup')/rel.lstrip('/');limits={k:(cg/k).read_text().strip()for k in ['cpu.max','memory.max','memory.swap.max']};assert limits=={'cpu.max':'100000 100000','memory.max':str(4*(1<<30)),'memory.swap.max':'0'}
 out=P/f'lane{a.fold}';out.mkdir(exist_ok=False);start=time.monotonic();result={'status':'STARTED','fold':a.fold,'cpu':cpu,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limits':limits,'allocation_sha256':sha(P/'allocation.json'),'jobs':[]}
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',ARROW_NUM_THREADS='1',MALLOC_CONF='background_thread:false',PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
 try:
  for name,digest in allocation['source_sha256'].items():assert sha(ROOT/name)==digest,name
  identity={}
  for name in ['datasets/valid.parquet','cache/chain_folds.npz','cache/valid_targets.npz']:
   identity[name]={'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size}
  expected=json.loads((P/'runtime-plan.json').read_text())['input'];assert identity['datasets/valid.parquet']['sha256']==expected['sha256']
  result['inputs_before']=identity
  jobs=[('prepare',[str(ROOT/'.venv/bin/python'),str(P/'prepare_oof.py'),'--fold',str(a.fold),'--allocation',str(P/'allocation.json')])]
  for mode in ['quant','fp16']:jobs.append((mode,['/home/grinchenkoeg/scorer_image/run.sh',str(P/'predict_oof.py'),'--manifest',str(P/f'oof/fold{a.fold}/{mode}.checkpoint.json')]))
  for name,argv in jobs:
   t=time.monotonic()
   with (out/f'{name}.log').open('x')as log:r=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,5350-(time.monotonic()-start)))
   result['jobs'].append({'name':name,'argv':argv,'seconds':time.monotonic()-t,'exit_code':r.returncode,'log_sha256':sha(out/f'{name}.log')});assert r.returncode==0
  result['inputs_after']={name:{'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size}for name in identity};assert result['inputs_after']==identity
  assert sha(P/'allocation.json')==result['allocation_sha256']
  result['status']='COMPLETE_TWO_FIXED_OOF_MODES_NO_METRIC'
 except BaseException as exc:result['status']='FAIL_NO_RETRY';result['error']=repr(exc);raise
 finally:
  result['seconds']=time.monotonic()-start;result['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
