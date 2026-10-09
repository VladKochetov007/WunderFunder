from pathlib import Path
import importlib.util,sys,json,hashlib,shutil
import numpy as np
import torch
work=Path('/home/vlad/development/wunder/worktrees/sol-diversity-delivery-20261009')
base=work/'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/tpu_gpu_benchmark'
producer=Path('/tmp/wnn-sol-frozen-gpu-r2/loss_pair_gpu_worker.py')
producer_sha=hashlib.sha256(producer.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('actual_gpu_writer',producer); writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
sys.path.insert(0,str(base));import tpu_packet_runtime as consumer
from run_device_parity import validate_profile_receipt
from wnn.nn import RNNModel
tmp=Path('/tmp/wnn-sol-actual-gpu-writer-fixture-r4');tmp.mkdir(exist_ok=False)
writer.OUT=tmp/'out';writer.OUT.mkdir()
manifest=json.loads((base/'fixtures/actual_cpu_stage_r1/manifest.json').read_text())
cache=tmp/'producer_cache';cache.mkdir()
for name in ('manifest.json','reader_validation.json','source_stage_receipt.json'): shutil.copyfile(base/'fixtures/actual_cpu_stage_r1'/name,cache/name)
shutil.copyfile(cache/'source_stage_receipt.json',tmp/'source_stage_receipt.json')
(cache/'source_verification.json').write_text('{}')
profile_path=tmp/'runtime_profile.json';profile_path.write_text('{}')
torch.manual_seed(20261009);model=RNNModel(n_in=211,hidden=256,layers=2,n_out=2,dropout=0)
state={k:v.numpy() for k,v in model.state_dict().items()}
orders={'schema':'wnn.loss-pair.sample-orders.v1','epochs':{'0':[f'datasets/train.parquet::rg={i}' for i in range(350)]}}
schedule={'schema':'wnn.loss-pair.onecycle-trace.v1','total_steps':220,'rows':[]}
arms={}
for name in ('wmse','huber_d1'):
 arm=tmp/name;prefit=arm/'prefit';prefit.mkdir(parents=True);arms[name]=arm
 np.savez(prefit/'initial_state.npz',**state)
 np.savez(prefit/'normalization.npz',mean=state['mean'],std=state['std'])
 (prefit/'normalization.json').write_text('{}');(prefit/'sample_orders.json').write_text(json.dumps(orders));(arm/'schedule_trace.json').write_text(json.dumps(schedule))
 np.savez(prefit/'profile_inputs.npz',inputs=np.zeros((32,2000,211),np.float32),targets=np.zeros((32,2000,2),np.float32),row_mask=np.broadcast_to(np.arange(2000)[None,:]>=99,(32,2000)))
bootstrap={'source_commit':'fixture_not_fit','asset_manifest_sha256':'fixture','archive_sha256':'fixture','code_manifest_sha256':'fixture'}
shared=writer.OUT/'shared';shared.mkdir();(shared/'device_profile_gate.json').write_text('{"fixture":true}');(shared/'fold0_assets_validation.json').write_text('{"fixture":true}')
for name in arms:
 out_arm=writer.OUT/'arms'/name;(out_arm/'model').mkdir(parents=True)
 (out_arm/'update_trace.jsonl').write_text('{"elapsed_seconds":1.0}\n')
 (out_arm/'model/checkpoint_timing.jsonl').write_text('{"fixture":true}\n')
contract,profile=writer._collect_shared(arms=arms,arm_run_names={k:k for k in arms},runtime_validation=cache/'reader_validation.json',runtime_profile=profile_path,source_receipt=json.loads((cache/'source_stage_receipt.json').read_text()),bootstrap_receipt=bootstrap,manifest_path=cache/'manifest.json',fit_results=[{'label':f'fit_{k}','wall_seconds':1.0} for k in arms],prediction_results={k:{'wall_seconds':1.0} for k in arms})
contract_path=writer.OUT/'training_contract.json'
admission={'gpu_training_contract_sha256':consumer.digest(contract_path)}
_,resolved=consumer.verify_shared_contract(contract_path,admission)
validate_profile_receipt(resolved['profile_inputs'],resolved['profile_inputs_receipt'],resolved['initial_state'])
consumer.check_profile_binding(json.loads(resolved['profile_inputs_receipt'].read_text()), orders['epochs']['0'][:32], consumer.digest(resolved['normalization']))
assert producer_sha==hashlib.sha256(producer.read_bytes()).hexdigest(),'producer changed during fixture check'
receipt={'status':'pass','producer_commit':'8924deec1eaf1a50cb31d4c42e042d24093b4be6','method':'invoked actual GPU worker _collect_shared then TPU verify_shared_contract/validate_profile_receipt','producer_path':str(producer),'producer_sha256':producer_sha,'writer_function':'_collect_shared','contract_sha256':consumer.digest(contract_path),'profile_receipt_sha256':consumer.digest(resolved['profile_inputs_receipt']),'roles_verified':sorted(resolved),'fixture_root':str(tmp),'limits':'Synthetic arrays and source placeholders; checks real writer schema/binary interface, not fitting/producer execution/actual device or data parity.'}
(base/'ACTUAL_GPU_WRITER_CONSUMER_CHECK_R2.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(contract_path,base/'fixtures/actual_gpu_writer_contract_r2.json')
shutil.copyfile(resolved['profile_inputs_receipt'],base/'fixtures/actual_gpu_writer_profile_receipt_r2.json')
print(json.dumps({'status':'pass','writer_sha256':producer_sha,'roles':len(resolved)}))
