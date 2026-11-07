from __future__ import annotations
import concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
Q=Path('/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1');H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');PY='/data/epanov_s/tabdata/.venv/bin/python'
sys.path.insert(0,str(H/'operator-r26/src'))
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import process_identity
P=json.loads((Q/'inputs/plan-r2.json').read_text())
def write(path:Path,value:object)->None:
 path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
def sha(path:Path)->str:
 return hashlib.sha256(path.read_bytes()).hexdigest()
now=time.time();contract={'schema':'wnn.quality-horizon-execution.v1','user_authority':'2026-10-31 explicit approval of quality-diagnosis-r1','root':str(Q),'start_epoch':now,'hard_deadline_epoch':now+14400,'fit_deadline_epoch':now+13500,'max_new_fits':5,'max_profile_attempts':5,'max_profile_wall_seconds':180,'profile_updates':50,'max_fit_wall_seconds':7200,'configured_epochs':4,'primary_split_sha256':'0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c','source_sha256':'c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81','coexistence_authorized':True,'foreign_process_stop':False,'cpu_threads_per_worker':4}
assert sha(Path(P['split']))==contract['primary_split_sha256'];contract=json.loads((Q/'CONTRACT_R2.json').read_text())
old=json.loads((H/'requests/stage4-primary-refit-primary-refit-01-s1t001002-gru.json').read_text());uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True);write(Q/'inputs/device_inventory.json',{'created_epoch':now,'gpus_csv':inventory,'uuid_by_index':uuids})
owner=process_identity(os.getpid());write(Q/'receipts/profile-r2-director.owner.json',owner)
def run(plan:dict[str,object])->dict[str,object]:
 name=plan['id'];env=os.environ.copy();env.update(old['child_environment']);env['CUDA_VISIBLE_DEVICES']=uuids[plan['gpu_index']]
 output=Q/'profiles-r2'/name;args=[PY,P['source_root']+'/scripts/train_sequence.py','--config',plan['config'],'--data-root',P['data_root'],'--split-manifest',P['split'],'--output',str(output),'--profile-updates','50'];argv=['/usr/bin/timeout','--signal=TERM','--kill-after=5s','175s',*args]
 req={'id':name,'gpu_uuid':env['CUDA_VISIBLE_DEVICES'],'config_sha256':sha(Path(plan['config'])),'argv':argv,'native_argv':args,'environment':{k:env[k] for k in old['child_environment']},'source_root':P['source_root'],'created_at_epoch':time.time()};write(Q/'inputs'/f'{name}.profile-r2-request.json',req)
 code,timed,record=run_owned_command(argv,cwd=Path(P['source_root']),env=env,log_path=Q/'logs'/f'{name}.profile-r2.log',receipt_path=Q/'receipts'/f'{name}.profile-r2.child.json',owner=owner,stage=1,trial_number=plan['gpu_index'],mode='quality_horizon_engineering_profile',gpu_key=env['CUDA_VISIBLE_DEVICES'],output_dir=output,timeout_seconds=180)
 result={'id':name,'exit_code':code,'timed_out':timed,'receipt':str(Q/'receipts'/f'{name}.profile-r2.child.json'),'profile':str(output/'profile_receipt.json'),'config':plan['config']};print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(run,P['plans']))
write(Q/'receipts/profile-wave-r2.completed.json',{'finished_epoch':time.time(),'profiles':results});print('PROFILE_WAVE_TERMINAL',flush=True)
