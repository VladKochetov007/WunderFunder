from pathlib import Path
import hashlib, json

MAIN=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
OPS=MAIN/'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011'
R2=OPS/'oldz4_f1_core_hft_r2'
R3=OPS/'oldz4_f1_hft_resource_fit_r3'
if R3.exists(): raise FileExistsError(R3)
R3.mkdir()

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(p.read_text())

profile=read(R2/'HFT_PROFILE_CONTRACT_R2.json')
fit=read(R2/'HFT_FIT_CONTRACT_R2.json')
attempt_path=MAIN/'research/campaign_2026-10-06/bandit_continuation/runtime/attempts/oldz4-r2-f1-s3-hft-profile.json'
log_path=MAIN/'research/campaign_2026-10-06/bandit_continuation/runtime/logs/oldz4-r2-f1-s3-hft-profile.log'
launch_path=R2/'ROOT_F1_HFT_PROFILE_LAUNCH_R2.json'
attempt=read(attempt_path); log=log_path.read_text()
line=next((x for x in log.splitlines() if x.startswith('PROFILE_UPDATE_EVIDENCE ')),None)
if line is None: raise ValueError('PROFILE_UPDATE_EVIDENCE missing from actual terminal log')
evidence=json.loads(line[len('PROFILE_UPDATE_EVIDENCE '):])
if not (attempt['id']=='oldz4-r2-f1-s3-hft-profile' and attempt['status']=='failed' and attempt['phase']=='terminal' and attempt['exit_status']==1): raise ValueError('unexpected profile terminal')
if not (evidence['optimizer_attempts']==270 and evidence['optimizer_completed_updates']==268 and evidence['amp_skipped_updates']==2 and evidence['gradient_stats_nonfinite_steps']==2 and evidence['loss_finite'] is True and evidence['gradient_finite'] is False and evidence['counts_consistent'] is True and evidence['valid_for_throughput'] is False): raise ValueError('actual profile evidence changed')
if fit['run']!='oldz4_h230_hft_f1_s3_r2' or fit['attempt']!='oldz4-r2-f1-s3-hft-fit': raise ValueError('unexpected R2 fit contract')
output_dir=MAIN/'artifacts/runs'/fit['run']
if output_dir.exists(): raise FileExistsError(f'fit output already exists: {output_dir}')

bindings_path=R2/'SOURCE_INPUT_BINDINGS_R2.json'; bindings=read(bindings_path)
r4source_path=Path(bindings['r4']['source_bindings_path']); r4sources=read(r4source_path)
snapshot_path=Path(r4sources['r3_source_snapshot_bindings_path'])
snapshot=read(snapshot_path)
crop_receipt=R2/'CORE_CROP_PRODUCER_RECEIPT_R2.json'
core=read(crop_receipt)
launch=read(launch_path)

hashes={
 'r2_packet_manifest_sha256':sha(R2/'PACKET_FILE_MANIFEST_R2B.json'),
 'r2_runner_sha256':sha(R2/'run_f1_core_hft_stage_r2.py'),
 'source_input_bindings_sha256':sha(bindings_path),
 'r4_input_bindings_sha256':sha(Path(bindings['r4']['input_bindings_path'])),
 'r4_source_bindings_sha256':sha(r4source_path),
 'source_snapshot_bindings_sha256':sha(snapshot_path),
 'trainer_sha256':snapshot['hft']['files']['scripts/train_rnn.py'],
 'profile_contract_sha256':sha(R2/'HFT_PROFILE_CONTRACT_R2.json'),
 'fit_contract_sha256':sha(R2/'HFT_FIT_CONTRACT_R2.json'),
 'export_contract_sha256':sha(R2/'HFT_EXPORT_CONTRACT_R2.json'),
 'core_crop_contract_sha256':sha(R2/'CORE_CROP_CONTRACT_R2.json'),
 'core_crop_receipt_sha256':sha(crop_receipt),
 'profile_launch_receipt_sha256':sha(launch_path),
 'profile_attempt_receipt_sha256':sha(attempt_path),
 'profile_log_sha256':sha(log_path),
}
resource={
 'schema':'wnn.oldz4-f1-hft-profile-resource-evidence.r3',
 'profile_attempt_id':attempt['id'],'profile_run':profile['run'],
 'profile_attempt_path':str(attempt_path),'profile_log_path':str(log_path),
 'attempt_status':attempt['status'],'attempt_phase':attempt['phase'],'attempt_exit_status':attempt['exit_status'],
 'failure_class':'strict numerical profile guard rejected nonfinite gradients; resource evidence remains separately inspectable',
 'profile_update_evidence':evidence,
 'profile_resource_observations':{
  'peak_worker_ram_bytes':attempt['peak_ram_bytes'],
  'peak_observed_vram_bytes':attempt['peak_observed_vram_bytes'],
  'worker_reservation':attempt['reservation'],
  'memory_event_deltas':attempt['memory_event_deltas'],
  'parent_memory_event_deltas':attempt['final_parent_memory_event_deltas'],
  'parent_hierarchical_memory_event_deltas':attempt['final_parent_hierarchical_memory_event_deltas'],
  'resource_exit_status':attempt['exit_status']
 },
 'hashes':hashes,
 'root_profile_launch_claims_success_or_fit_admission':launch['profile_success_or_fit_admission_claim'],
 'prepared_utc':'2026-10-12T02:20:00+00:00',
 'qualification':'resource-only evidence; not finite-gradient profile success and not throughput evidence'
}
evidence_path=R3/'F1_HFT_PROFILE_RESOURCE_EVIDENCE_R3.json'
evidence_path.write_text(json.dumps(resource,indent=2,sort_keys=True)+'\n')
evidence_sha=sha(evidence_path)
argv_sha=hashlib.sha256(json.dumps(fit['trainer_argv'],separators=(',',':')).encode()).hexdigest()
contract={
 'schema':'wnn.oldz4-f1-hft-resource-only-fit-contract.r3',
 'packet_id':'oldz4_f1_hft_resource_fit_r3',
 'status':'prepared_pending_root_resource_only_admission',
 'attempt':'oldz4-r3-f1-s3-hft-fit-resource',
 'fit_run':fit['run'],
 'original_fit_attempt':fit['attempt'],
 'scope':'One original fixed-epoch HFT fit using the unchanged R2 fit contract and trainer argv. The failed R2 profile remains failed; no throughput or clean-gradient claim.',
 'no_scientific_change':True,
 'strict_r2_gate_unchanged':True,
 'root_release':{
  'path':'ROOT_F1_HFT_RESOURCE_ONLY_FIT_ADMISSION_R3.json',
  'schema':'wnn.oldz4.f1-hft-resource-only-fit-admission.r3',
  'decision':'PASS_FOR_ONE_BOUNDED_ORIGINAL_FIT',
  'required_cli_binding':'--root-release-sha256 <sha256 of exact root release file>'
 },
 'required_root_release_fields':[
  'decision','attempt','fit_run','fit_contract_sha256','profile_contract_sha256','core_crop_contract_sha256','core_crop_receipt_sha256','source_input_bindings_sha256','source_snapshot_bindings_sha256','r2_runner_sha256','resource_contract_sha256','resource_adapter_sha256','profile_launch_receipt_sha256','profile_attempt_receipt_sha256','profile_log_sha256','profile_resource_evidence_sha256','fit_trainer_argv_sha256','fit_argv_unchanged','profile_strict_gate_passed','profile_valid_for_throughput','resource_limits'
 ],
 'hashes':{**hashes,'profile_resource_evidence_sha256':evidence_sha,'fit_trainer_argv_sha256':argv_sha},
 'profile_evidence_expected':{
  'optimizer_attempts':270,'optimizer_completed_updates':268,'amp_skipped_updates':2,
  'loss_checks':270,'loss_finite':True,'gradient_finite':False,
  'gradient_stats_steps':270,'gradient_stats_nonfinite_steps':2,
  'counts_consistent':True,'valid_for_throughput':False
 },
 'resource_limits':{
  'cpu_cores':1,'worker_memory_max_bytes':6800000000,
  'allocator_vram_bytes':4300000000,'runtime_max_seconds':1800,
  'loader_workers':0,'threads':1,'swap_bytes':0
 },
 'profile_resource_ceiling_checks':{
  'peak_worker_ram_bytes_max':6800000000,
  'peak_observed_vram_bytes_max':4300000000,
  'memory_event_deltas_must_all_equal_zero':True
 },
 'fit':{
  'contract_path_relative_to_r2':'HFT_FIT_CONTRACT_R2.json',
  'trainer_argv_unchanged':True,
  'trainer_argv_sha256':argv_sha,
  'trainer_sha256':hashes['trainer_sha256'],
  'output_run_directory':str(output_dir),
  'output_directory_absent_at_freeze':True,
  'root_crop_receipt_path':str(crop_receipt)
 },
 'profile_evidence_path':'F1_HFT_PROFILE_RESOURCE_EVIDENCE_R3.json',
 'profile_evidence_sha256':evidence_sha,
 'root_profile_launch_receipt_path':str(launch_path),
 'original_r2_packet_path':str(R2),
 'prepared_utc':'2026-10-12T02:20:00+00:00'
}
contract_path=R3/'HFT_RESOURCE_ONLY_FIT_CONTRACT_R3.json'
contract_path.write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
for name,path in [('profile_attempt_receipt',attempt_path),('profile_log',log_path),('profile_launch_receipt',launch_path),('core_crop_receipt',crop_receipt)]: print(name,sha(path))
print('hashes',json.dumps(hashes,sort_keys=True))
print('evidence_sha',evidence_sha,'argv_sha',argv_sha,'contract_sha',sha(contract_path))
print('profile counters',json.dumps(evidence,sort_keys=True))
