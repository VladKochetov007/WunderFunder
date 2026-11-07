from __future__ import annotations
import concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
Q=Path('/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1');H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');PY='/data/epanov_s/tabdata/.venv/bin/python'
sys.path.insert(0,str(H/'operator-r26/src'))
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import process_identity
P=json.loads((Q/'inputs/plan-r2.json').read_text());C=json.loads((Q/'CONTRACT_R2.json').read_text())
def sha(p:Path)->str:
 return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,v:object)->None:
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
assert time.time()<C['fit_deadline_epoch'] and len(P['plans'])==5
for p in P['plans']:
 name=p['id'];d=Q/'profiles-r2'/name;r=json.loads((d/'engineering_review_r2.json').read_text());native=json.loads((d/'profile_receipt.json').read_text());assert r['status']=='engineering_checks_passed' and not r['failed_checks'];assert r['native_profile_sha256']==sha(d/'profile_receipt.json') and native['approved'] is False
 files=native['identity']['source_files'];assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==C['source_sha256']
 for n,digest in files.items():assert sha(Path(P['source_root'])/n)==digest,n
 approved=dict(native);approved['approved']=True;approved['external_review']={'path':str(d/'engineering_review_r2.json'),'sha256':sha(d/'engineering_review_r2.json'),'native_profile_sha256':sha(d/'profile_receipt.json')};write(d/'approved_profile_receipt.json',approved)
 write(Q/'receipts'/f'{name}.root-approval.json',{'schema':'wnn.quality-horizon-root-fit-approval.v1','user_authority':'explicit approval 2026-10-31','profile_sha256':sha(d/'profile_receipt.json'),'review_sha256':sha(d/'engineering_review_r2.json'),'config_sha256':sha(Path(p['config'])),'source_sha256':C['source_sha256'],'foreign_process_stop':False,'four_fixed_epochs':True})
owner=process_identity(os.getpid());write(Q/'receipts/fit-director.owner.json',owner)
def run(plan:dict[str,object])->dict[str,object]:
 name=plan['id'];req=json.loads((Q/'inputs'/f'{name}.profile-r2-request.json').read_text());env=os.environ.copy();env.update(req['environment']);output=Q/'runs'/name
 assert not output.exists();seconds=min(C['max_fit_wall_seconds'],int(C['fit_deadline_epoch']-time.time()-10));assert seconds>600
 args=[PY,P['source_root']+'/scripts/train_sequence.py','--config',plan['config'],'--data-root',P['data_root'],'--split-manifest',P['split'],'--output',str(output),'--profile-receipt',str(Q/'profiles-r2'/name/'approved_profile_receipt.json')];argv=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds-5)+'s',*args]
 write(Q/'inputs'/f'{name}.fit-request.json',{'argv':argv,'native_argv':args,'env':req['environment'],'source_sha256':C['source_sha256'],'config_sha256':sha(Path(plan['config'])),'id':name,'scientific_attempt':True,'max_seconds':seconds,'fixed_epochs':4})
 code,timed,record=run_owned_command(argv,cwd=Path(P['source_root']),env=env,log_path=Q/'logs'/f'{name}.fit.log',receipt_path=Q/'receipts'/f'{name}.fit.child.json',owner=owner,stage=1,trial_number=plan['gpu_index'],mode='quality_horizon_scientific_fit',gpu_key=req['gpu_uuid'],output_dir=output,timeout_seconds=seconds)
 result={'id':name,'exit_code':code,'timed_out':timed,'status':record['status'],'output':str(output)};write(Q/'receipts'/f'{name}.fit-terminal.json',result);print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(run,P['plans']))
write(Q/'receipts/fit-wave.completed.json',{'finished_epoch':time.time(),'fits':results});print('FIVE_FITS_TERMINAL',flush=True)
