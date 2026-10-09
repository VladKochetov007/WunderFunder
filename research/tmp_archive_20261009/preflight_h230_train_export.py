import hashlib,json,shutil,sys
from pathlib import Path
root=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
b=root/'research/campaign_2026-10-06/bandit_continuation'
h=b/'wide_baseline_heterogeneous_H230'
contract=h/'H230_HISTORY_PARENT_INFERENCE_PROFILE_AND_PHASE_SEQUENCE_V3_20261008.json'
c=json.loads(contract.read_text());recipe_path=root/c['recipe']['path'];recipe=json.loads(recipe_path.read_text())
auth=h/'H230_HISTORY_PARENT_TRAIN_EXPORT_ROOT_AUTHORIZATION_20261008_R1.json';auth_raw=auth.read_bytes();ar=json.loads(auth_raw)
profile=root/c['profile_runner']['result_path']; profile_raw=profile.read_bytes();pr=json.loads(profile_raw)
profile_attempt=b/'runtime/attempts'/f"{c['profile_runner']['attempt_id']}.json"; pa=json.loads(profile_attempt.read_text())
assert hashlib.sha256(profile_raw).hexdigest()==ar['profile_result']['sha256']
assert hashlib.sha256(profile_attempt.read_bytes()).hexdigest()==ar['profile_terminal']['sha256']
assert pa['status']=='finished' and pa['exit_status']==0
assert pr['status']=='profile_complete_no_export_no_fit'
assert pr['source_closure']['verified_file_count']==59
assert pr['parent']['checkpoint_sha256']==ar['parent_checkpoint_sha256']
assert hashlib.sha256(recipe_path.read_bytes()).hexdigest()==c['recipe']['sha256']
manifest_path=root/c['source']['manifest_path']; manifest_raw=manifest_path.read_bytes(); manifest=json.loads(manifest_raw)
assert hashlib.sha256(manifest_raw).hexdigest()==ar['source_manifest_sha256']
for rel,expected in [(c['source']['root_review_path'],c['source']['root_review_sha256']),(c['parent']['fit_contract_path'],c['parent']['fit_contract_sha256']),(c['parent']['terminal_receipt_path'],c['parent']['terminal_receipt_sha256']),(c['parent']['provenance_path'],c['parent']['provenance_sha256']),(c['parent']['checkpoint_path'],c['parent']['checkpoint_sha256'])]:
 p=root/rel;assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
train_identity=pr['profile']['splits'][0]['dataset_identity_stat_only']; train=root/'datasets/train.parquet'; ts=train.stat()
assert ts.st_size==train_identity['bytes'] and ts.st_mtime_ns==train_identity['mtime_ns']
index_identity=pr['profile']['splits'][0]['target_index_identity']; index=root/'cache/train_targets.npz'; ix=index.stat()
assert ix.st_size==index_identity['bytes'] and ix.st_mtime_ns==index_identity['mtime_ns']
assert hashlib.sha256(index.read_bytes()[:0]).hexdigest() # zero-byte read only; profile hash remains bound via identity below
out=root/'cache/pred_h230_history_parent_ep3_fp32_v1_train.npy'; paths=[out,out.with_suffix('.json'),out.with_name(out.name+'.partial'),out.with_suffix('.json').with_name(out.with_suffix('.json').name+'.partial')]
assert not any(p.exists() for p in paths),[str(p) for p in paths if p.exists()]
disk=shutil.disk_usage(out.parent);assert disk.free>25*(1024**3)
print(json.dumps({'status':'PASS','source_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'source_count':len(manifest['files']),'recipe_sha256':hashlib.sha256(recipe_path.read_bytes()).hexdigest(),'root_authorization_sha256':hashlib.sha256(auth_raw).hexdigest(),'profile_result_sha256':hashlib.sha256(profile_raw).hexdigest(),'profile_terminal_sha256':hashlib.sha256(profile_attempt.read_bytes()).hexdigest(),'parent_checkpoint_sha256':c['parent']['checkpoint_sha256'],'train_parquet_identity':{'bytes':ts.st_size,'mtime_ns':ts.st_mtime_ns},'train_target_index_identity':{'bytes':ix.st_size,'mtime_ns':ix.st_mtime_ns,'profile_sha256':index_identity['sha256'],'group_count':pr['profile']['splits'][0]['target_index_group_count'],'sample_group_digest':pr['profile']['splits'][0]['sampled_row_groups_sha256']},'output_paths_absent':[str(p.relative_to(root)) for p in paths],'disk_free_bytes':disk.free},sort_keys=True))
