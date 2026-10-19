import hashlib, json, math
from pathlib import Path
base = Path('research/campaign_2026-10-19/idea_screens_r13'); cap = base/'records/terminal_capture'
def raw(p): return p.read_bytes()
def load(p): return json.loads(raw(p))
def sha(data): return hashlib.sha256(data).hexdigest()
result_b=raw(cap/'REMOTE_RESULT.json'); source_b=raw(cap/'REMOTE_SOURCE.json'); done_b=raw(cap/'REMOTE_DONE.json')
result=json.loads(result_b); source=json.loads(source_b); done=json.loads(done_b)
manifest_b=raw(base/'PREPARATION_MANIFEST.json'); manifest=json.loads(manifest_b)
smoke=load(cap/'REMOTE_ROOT_IMPORT_SMOKE.json'); admission=load(cap/'REMOTE_ROOT_ADMISSION.json'); launch=load(cap/'REMOTE_ROOT_LAUNCH.json')
validation=load(base/'records/ROOT_CAPTURE_VALIDATION.json'); stop_receipt=load(base/'records/STOP_RECEIPT.json')
stop_b=raw(base/'records/stop.stdout'); stop=json.loads(stop_b)
result_sha=sha(result_b); source_sha=sha(source_b); manifest_sha=sha(manifest_b)
assert result_sha=='7219d98d0856e399dcde390f08b19cbc1cc89e675e75eb6c4fee7c58cf3802dd'
assert done['result_sha256']==result_sha and done['result_bytes']==len(result_b)
assert result['source_sha256']==source_sha
assert source['preparation_manifest_sha256']==manifest_sha=='8504aedca37a20d76ae31d1b2fcd7498676c956f70e3c7e69017a8be13213861'
assert source['source_pins']==manifest['files'] and len(source['source_pins'])==36
assert source['runtime_imports'].keys()==manifest['runtime_py'].keys() and len(source['runtime_imports'])==30
assert smoke['preparation_manifest_sha256']==manifest_sha and smoke['status']=='PASS' and len(smoke['runtime_import_paths'])==30
assert set(smoke['runtime_source_sha256'].values())=={manifest['files'][p] for p in manifest['runtime_py'].values()}
assert smoke['reader_sentinel_calls']==[] and all(smoke[k]==0 for k in ('array_reads','parquet_reads','checkpoint_reads')) and smoke['cuda_initialized'] is False
scores=result['scores']; arms=['zero','current_bypass','warmup99']; components=[str(c) for c in scores['eval_components']]
assert list(scores['arms'])==arms and len(components)==len(set(components))==32
def metric(six):
 corr=[]
 for row in six:
  sw,swy,swp,swyy,swpp,swyp=map(float,row); cov=swyp-swy*swp/sw; vy=swyy-swy*swy/sw; vp=swpp-swp*swp/sw; corr.append(cov/math.sqrt(vy*vp))
 return {'t0':corr[0],'t1':corr[1],'wp':math.fsum(corr)/2.0}
def add_six(items): return [[math.fsum(float(item[t][j]) for item in items) for j in range(6)] for t in range(2)]
errors={'pooled':0.0,'component':0.0,'drop_one':0.0,'contrasts':0.0,'component_contrasts':0.0,'drop_one_contrasts':0.0}
recon={}; comp_n=drop_n=0
for arm in arms:
 data=scores['arms'][arm]; suff=data['official_clipped_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp']
 assert set(suff)==set(data['component_scores'])==set(data['component_drop_one_scores'])==set(data['component_rows'])==set(components)
 pooled=metric(add_six([suff[c] for c in components])); official=data['official_metric']
 for k,v in pooled.items(): errors['pooled']=max(errors['pooled'],abs(v-official[k]))
 rows=sum(data['component_rows'].values()); assert rows==scores['eval_scored_rows']==59436
 recon[arm]=pooled
 for c in components:
  cm=metric(suff[c])
  for k,v in cm.items(): errors['component']=max(errors['component'],abs(v-data['component_scores'][c][k]))
  dm=metric(add_six([suff[x] for x in components if x!=c]))
  assert data['component_drop_one_scores'][c]['selected_rows']==rows-data['component_rows'][c]
  for k,v in dm.items(): errors['drop_one']=max(errors['drop_one'],abs(v-data['component_drop_one_scores'][c][k]))
  comp_n+=1; drop_n+=1
pairs={'warmup99_minus_current_bypass_wp':('warmup99','current_bypass'),'warmup99_minus_zero_wp':('warmup99','zero'),'current_bypass_minus_zero_wp':('current_bypass','zero')}
comp_deltas={}; drop_deltas={}
for label,(a,b) in pairs.items():
 d=recon[a]['wp']-recon[b]['wp']; errors['contrasts']=max(errors['contrasts'],abs(d-scores['deltas_wp'][label]))
 assert abs(d-scores['deltas_wp'][label])<1e-12
 cv=[]; dv=[]
 for c in components:
  cval=scores['arms'][a]['component_scores'][c]['wp']-scores['arms'][b]['component_scores'][c]['wp']
  dval=scores['arms'][a]['component_drop_one_scores'][c]['wp']-scores['arms'][b]['component_drop_one_scores'][c]['wp']; cv.append(cval); dv.append(dval)
  errors['component_contrasts']=max(errors['component_contrasts'],abs(cval-scores['component_deltas_wp'][c][label]))
  errors['drop_one_contrasts']=max(errors['drop_one_contrasts'],abs(dval-scores['component_drop_one_deltas_wp'][c][label]))
 comp_deltas[label]={'positive':sum(x>0 for x in cv),'negative':sum(x<0 for x in cv),'min':min(cv),'max':max(cv)}
 drop_deltas[label]={'positive':sum(x>0 for x in dv),'negative':sum(x<0 for x in dv),'min':min(dv),'max':max(dv)}
assert len(scores['component_deltas_wp'])==len(scores['component_drop_one_deltas_wp'])==32 and comp_n==drop_n==96 and max(errors.values())<2e-12
assert result['fit_components']==source['panel_stats']['fit_components']==64 and result['evaluation_components']==source['panel_stats']['eval_components']==32
assert result['fit_rows_per_arm']==source['panel_stats']['fit_stride_rows']==31808
assert result['evaluation_scored_rows']==source['panel_stats']['eval_original_scored_rows']==59436 and source['panel_stats']['fit_original_scored_rows']==1273664
assert source['sample_records']==96 and source['sample_arrays_hashed_and_mapped_for_integrity']==288 and source['evaluation_records']==32
assert source['evaluation_target_files_hashed_and_mapped_for_integrity'] is True
assert source['evaluation_target_values_used_for_fit_normalization_or_selection'] is False and source['evaluation_target_values_used_for_final_scoring'] is True
assert source['sample_sha256']==admission['sample']['sample_sha256']=='b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64'
assert source['membership_sha256']==admission['sample']['membership_sha256'] and source['split_sha256']==admission['sample']['split_sha256']=='c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7'
init=result['initialization']; assert len(set(init['state_fingerprints'].values()))==1 and init['initial_prediction_parity']['initial_prediction_max_abs_difference']==0.0
assert init['added_side_weights_zero'] and init['logvariance_heads_zero_and_unused']
for arm in arms:
 t=result['training'][arm]; assert t['epochs_completed']==12 and t['attempted_updates']==t['successful_updates']==756 and t['nonfinite_count']==0 and t['row_exposures']==381696
 assert t['variance_head_unchanged'] and t['variance_head_grad_none']
assert result['total_updates']==done['successful_updates']==2268 and result['total_row_exposures']==1145088 and done['nonfinite_count']==0
unit=admission['unit']; inv=launch['invocation_id']
assert unit==launch['unit']==validation['unit']==stop['unit']=='computer-vision-homework-eda19-r13-v1.service'
assert launch['systemd_run_returncode']==0 and stop['before']['InvocationID']==inv==validation['invocation_id']
assert stop['before']['Result']=='success' and stop['before']['ExecMainStatus']=='0'
assert stop['after']['LoadState']=='not-found' and stop['after']['ActiveState']=='inactive' and stop['after']['SubState']=='dead'
assert stop['returncode']==stop_receipt['ssh_returncode']==0 and stop_receipt['stdout']['bytes']==len(stop_b) and stop_receipt['stdout']['sha256']==sha(stop_b) and stop_receipt['stderr']['bytes']==0
assert validation['status']=='PASS_BEFORE_EXACT_STOP' and validation['result_sha256']==result_sha and validation['source_sha256']==source_sha and validation['manifest_sha256']==manifest_sha
assert validation['successful_updates_total']==2268 and validation['validated_transported_payloads']==9 and validation['model_payload_opened_or_copied'] is False
caps=admission['caps']; assert caps['cpu_cores']==0.5 and caps['blas_threads']==1 and caps['memory_high_bytes']==1400000000 and caps['memory_max_bytes']==1500000000 and caps['memory_swap_max_bytes']==0 and caps['runtime_max_s']==600 and caps['cuda_visible_devices']==''
assert stop['before']['MemoryHigh']==str(caps['memory_high_bytes']) and stop['before']['MemoryMax']==str(caps['memory_max_bytes']) and stop['before']['MemorySwapMax']==str(caps['memory_swap_max_bytes']) and stop['before']['CPUQuotaPerSecUSec']=='500ms'
archive_match=result['model_archive']['sha256']==done['model_archive']['sha256'] and result['model_archive']['bytes']==done['model_archive']['bytes']; assert archive_match
memguards=[e['memory_guard'] for arm in result['training'].values() for e in arm['epochs'] if isinstance(e.get('memory_guard'),dict)]
max_cgroup=max(g['own_cgroup_memory_current_bytes'] for g in memguards if 'own_cgroup_memory_current_bytes' in g)
max_rss=max(g['process_rss_bytes'] for g in memguards if 'process_rss_bytes' in g)
audit={
 'schema':'nf19.r13.independent_result_audit.v1','status':'PASS_SCALAR_RECONSTRUCTION_AND_RECEIPT_BINDINGS',
 'scope':'Parsed compact JSON receipts and sufficient statistics only; no NPY/NPZ/Parquet payloads, model-weight loads, or training/scoring replay.',
 'bindings':{'result_sha256':result_sha,'source_json_sha256':source_sha,'done_result_sha256':done['result_sha256'],'manifest_sha256':manifest_sha,'manifest_file_count':36,'source_pin_count':36,'runtime_import_count':30,'sample_sha256':source['sample_sha256'],'membership_sha256':source['membership_sha256'],'split_sha256':source['split_sha256'],'unit':unit,'invocation_id':inv,'capture_validation_status':validation['status'],'import_smoke_status':smoke['status'],'stop_ssh_returncode':stop_receipt['ssh_returncode'],'post_stop_state':{'load_state':stop['after']['LoadState'],'active_state':stop['after']['ActiveState'],'sub_state':stop['after']['SubState']}},
 'score_reconstruction':{'metric_id':scores['metric_id'],'eval_scored_rows':59436,'eval_components':32,'component_scores_reconstructed':comp_n,'drop_one_scores_reconstructed':drop_n,'pooled_by_arm':recon,'max_abs_errors':errors,'reported_wp_deltas':scores['deltas_wp'],'primary_warmup99_minus_current_bypass_wp':scores['primary_delta_warmup99_minus_current_bypass_wp'],'component_delta_sign_counts_ranges':comp_deltas,'drop_one_delta_sign_counts_ranges':drop_deltas,'component_sensitivity_is_not_independent_replication':True,'exploratory_gate_margin_wp':scores['prospective_exploratory_gate_margin_wp'],'exploratory_gate_passed':scores['prospective_exploratory_gate_passed']},
 'protocol':{'arms':arms,'fit_components':64,'evaluation_components':32,'fit_stride_rows_per_arm':31808,'fit_original_scored_rows':1273664,'eval_scored_rows':59436,'sample_records':96,'sample_arrays_hashed_and_mapped_for_integrity':288,'eval_target_values_used_for_fit_normalization_or_selection':False,'eval_target_values_used_for_final_scoring':True,'same_initial_state':True,'initial_prediction_max_abs_difference':0.0,'updates_per_arm':756,'row_exposures_per_arm':381696,'total_updates':2268,'total_row_exposures':1145088,'nonfinite_count':0,'variance_heads_unchanged_and_grad_none':True,'zero_arm_added_side_weights_unchanged':result['training']['zero']['zero_side_weights_unchanged'],'archive_metadata_matches_done':archive_match,'archive_payload_read_by_auditor':False},
 'resource_measurements':{'admitted_caps':caps,'systemd_memory_peak_bytes_before_stop':stop['before'].get('MemoryPeak'),'maximum_recorded_inprocess_cgroup_memory_current_bytes':max_cgroup,'maximum_recorded_process_rss_bytes':max_rss,'measurements_kept_separate':True},
 'interpretation_limit':'One fixed R3 exposed-panel comparison at one initialization/training recipe. Component and drop-one contrasts are panel sensitivities, not independent replications; this audit does not establish GRU memory/state behavior or population generality.'}
out=base/'RESULT_AUDIT.json'
with out.open('x',encoding='utf-8') as f: json.dump(audit,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print('WROTE',out,'SHA256',sha(raw(out)))
print(json.dumps({'errors':errors,'pooled':recon,'component_deltas':comp_deltas,'drop_one_deltas':drop_deltas,'resource_measurements':audit['resource_measurements']},indent=2))
