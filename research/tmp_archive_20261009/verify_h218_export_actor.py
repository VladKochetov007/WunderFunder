import hashlib,json
from pathlib import Path
import numpy as np
root=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
base=root/'research/campaign_2026-10-06/bandit_continuation'
attempt='h218-block-memory-cleanfold0-eval468-fp32-export-v2-20261008'
receipt_path=base/f'runtime/attempts/{attempt}.json'; receipt_raw=receipt_path.read_bytes(); receipt=json.loads(receipt_raw)
pred=root/'cache/pred_h218_block_memory_cleanfold0_seed7_ep3_fp32_eval468_v1_valid.npy'
side=pred.with_suffix('.json'); side_raw=side.read_bytes(); meta=json.loads(side_raw)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
assert receipt['status']=='finished' and receipt['phase']=='terminal' and receipt['exit_status']==0
assert receipt['contract_sha256']==sha(base/'attention_training_H218/H218_FOLD0_EVAL_EXPORT_CONTRACT_V2_20261008.json')
assert meta['prediction_path']==str(pred.relative_to(root))
assert meta['prediction_shape']==[468,20000,2] and meta['prediction_dtype']=='float32'
assert meta['checkpoint_sha256']=='ab9c7ae1ef415de54dcc56bcfe1c47239e742d12e4413a88c9303a1500075700'
assert meta['evaluation_group_ids_sha256']=='2aac564e257cf5857d20f3a54a44de52e23ba5c9fa79c01df4f59e2d5cdd1ec1'
prereg=json.loads((base/'attention_training_H218/H218_FOLD0_REFERENCE_BLEND_PREREG_20261008_R1.json').read_text())
fold=json.loads((root/'research/campaign_2026-09-29/artifacts/splits/fold0.json').read_text())
assert meta['evaluation_row_groups']==fold['evaluation']['valid']
arr=np.load(pred,mmap_mode='r',allow_pickle=False)
assert arr.shape==(468,20000,2) and arr.dtype==np.float32
for lo in range(0,468,8):
 if not np.isfinite(arr[lo:lo+8]).all():raise ValueError(f'nonfinite rows {lo}:{lo+8}')
ph=sha(pred)
assert ph==meta['prediction_sha256']
assert pred.stat().st_size==74_880_128 and pred.stat().st_size==meta['prediction_bytes']
print(json.dumps({
 'attempt_id':attempt,'attempt_receipt_path':str(receipt_path.relative_to(root)),'attempt_receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),
 'attempt_status':receipt['status'],'exit_status':receipt['exit_status'],'elapsed_seconds':receipt['elapsed_seconds'],
 'contract_path':receipt['contract'],'contract_sha256':receipt['contract_sha256'],'log_path':str(receipt['log']),
 'prediction_path':str(pred.relative_to(root)),'prediction_sha256':ph,'prediction_bytes':pred.stat().st_size,'prediction_shape':list(arr.shape),'prediction_dtype':str(arr.dtype),'finite_rows_checked':468,
 'sidecar_path':str(side.relative_to(root)),'sidecar_sha256':hashlib.sha256(side_raw).hexdigest(),'row_group_count':len(meta['evaluation_row_groups']),'row_group_sha256':meta['evaluation_group_ids_sha256'],
 'checkpoint_sha256':meta['checkpoint_sha256'],'source_lock_sha256':meta['source_sha256']['source_lock'],'split_manifest_sha256':meta['split_manifest_sha256'],
 'timing':meta['timing'],'official_score_computed':meta['official_score_computed'],'labels_read':meta['labels_read'],'targets_read':meta['targets_read'],
 'peak_worker_ram_bytes':receipt['peak_ram_bytes'],'peak_observed_vram_bytes':receipt['peak_observed_vram_bytes'],'worker_oom_deltas':receipt['final_memory_event_deltas'],'parent_local_oom_deltas':receipt['final_parent_memory_event_deltas']
},sort_keys=True))
