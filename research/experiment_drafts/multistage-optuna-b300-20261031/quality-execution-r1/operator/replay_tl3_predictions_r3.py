from __future__ import annotations
import concurrent.futures,json,os,sys
from pathlib import Path
Q=Path('/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1');H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');sys.path.insert(0,str(H/'operator-r26/src'))
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import process_identity
owner=process_identity(os.getpid());uuids=json.loads((Q/'inputs/device_inventory.json').read_text())['uuid_by_index'];old=json.loads((H/'requests/stage4-primary-refit-primary-refit-01-s1t001002-gru.json').read_text());root=Q/'legacy-replay';jobs=[('tl3c_w18final_distilled_s5',3),('tl3c_fttall_tlead_s5',4),('tl3c_fttall_tlead_s6',5)]
def run(job:tuple[str,int])->dict[str,object]:
 name,gpu=job;env=os.environ.copy();env.update(old['child_environment']);env.update(CUDA_VISIBLE_DEVICES=uuids[gpu],PYTHONPATH=str(H/'operator-native4/src')+':'+str(H/'python-deps'),WNN_PROJECT_ROOT=str(root/'data'));argv=['/usr/bin/timeout','--signal=TERM','--kill-after=5s','890s','/data/epanov_s/tabdata/.venv/bin/python',str(root/'source/scripts/predict_run.py'),name,'valid','--campaign-split-manifest',str(H/'data/chronological_primary.json'),'--campaign-evaluation-only','--inference-precision','fp32'];code,timed,record=run_owned_command(argv,cwd=root/'source',env=env,log_path=Q/'logs'/(name+'.predict-r3.log'),receipt_path=Q/'receipts'/(name+'.predict-r3.child.json'),owner=owner,stage=0,trial_number=gpu,mode='fixed_tl3_replay_inference_only',gpu_key=uuids[gpu],output_dir=root/'data/cache',timeout_seconds=900);r={'run':name,'exit_code':code,'timed_out':timed,'status':record['status']};print(json.dumps(r),flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,jobs))
(Q/'receipts/tl3-replay-predictions-r3.completed.json').write_text(json.dumps({'jobs':results,'new_fits':0},indent=2));print('TL3_REPLAY_TERMINAL',flush=True)
