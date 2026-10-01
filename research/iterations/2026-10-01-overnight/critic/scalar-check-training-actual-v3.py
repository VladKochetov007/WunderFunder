"""Independent fixed-fit/OOF integrity audit. No targets, model forward or quality metrics.

--prepare-only checks frozen sources and exact declared jobs without reading any
empirical checkpoint, prediction, fitted knot array or dataset metadata array.
"""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,math,sys
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2];S=P/'scalar';E=S/'training-seed3-execution'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(1<<20):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def process_contracts(plan,allocation):
 """Literal reconstruction of the frozen driver's22 subprocess contracts."""
 contracts={}
 def add(label,script,arguments,lane,uses_gpu,cap):
  threads='4' if lane else '1';affinity=lane['cpus'] if lane else '38'
  environment={'CUDA_VISIBLE_DEVICES':str(lane['gpu']) if uses_gpu else '',
   'WNN_THREADS':threads,'OMP_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':'1',
   'MKL_NUM_THREADS':threads,'NUMEXPR_NUM_THREADS':'1','WNN_GPU_FRACTION':'0.34375',
   'WNN_LOADER_WORKERS':'2','WNN_LOADER_RAM_GB':'2'}
  contracts[label]={'argv':['taskset','-c',affinity,str(ROOT/'.venv/bin/python'),str(S/script),*arguments],
   'environment':environment,'cap_seconds':cap,'uses_gpu':uses_gpu}
 for part in ['before','after']:
  add('inputs-'+part,'input_preflight.py',['--allocation',str(allocation),'--output',str(E/f'inputs-{part}.json')],None,False,300)
 for lane in plan['lanes']:
  for name in lane['runs']:
   job=plan['runs'][name];derived=E/f'{name}.allocation.json'
   common=['--allocation',str(derived),'--run',name]
   add(name,'train.py',common,lane,True,plan['limits']['parent_seconds' if job['stage']=='parent' else 'head_seconds'])
   add(name+'-check','check_fit.py',common+['--output',str(E/f'{name}.integrity.json')],lane,False,120)
   if job['stage']=='head':add(name+'-OOF','predict_scalar.py',['--allocation',str(E/f'{name}.oof-allocation.json'),'--run',name],lane,True,plan['limits']['prediction_seconds'])
 assert len(contracts)==22
 return contracts
def verify_execution_order(records,plan,allowance_seconds=12):
 """UTC receipt reconstruction, with the pre-existing sampled-monitor allowance."""
 stamp=lambda label,key:datetime.datetime.fromisoformat(records[label][key])
 reserve=datetime.datetime.fromisoformat(plan['deadline_UTC'])-datetime.timedelta(seconds=4800)
 bound=reserve+datetime.timedelta(seconds=allowance_seconds)
 for name in records:
  assert stamp(name,'started_UTC')<=stamp(name,'finished_UTC')<=bound,(name,'final-reserve')
  if name not in ['inputs-before','inputs-after']:
   assert stamp('inputs-before','finished_UTC')<=stamp(name,'started_UTC'),(name,'before-inputs-finished')
   assert stamp(name,'finished_UTC')<=stamp('inputs-after','started_UTC'),(name,'after-inputs-started')
 for lane in plan['lanes']:
  previous=None;heads=[]
  for name in lane['runs']:
   assert stamp(name,'finished_UTC')<=stamp(name+'-check','started_UTC'),(name,'check-before-fit-end')
   if previous:assert stamp(previous+'-check','finished_UTC')<=stamp(name,'started_UTC'),(name,'lane-overlap')
   previous=name
   if plan['runs'][name]['stage']=='head':heads.append(name)
  previous_end=max(stamp(name+'-check','finished_UTC') for name in lane['runs'])
  for name in heads:
   assert previous_end<=stamp(name+'-OOF','started_UTC'),(name,'OOF-order')
   previous_end=stamp(name+'-OOF','finished_UTC')
 return {'global_input_order_verified':True,'per_lane_train_check_OOF_order_verified':True,
  'final_reserve_cutoff_UTC':reserve.isoformat(),'receipt_sampling_allowance_seconds':allowance_seconds,
  'latest_job_finish_UTC':max(stamp(name,'finished_UTC') for name in records).isoformat()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 allocation=S/'training-seed3-allocation.json';planpath=S/'training-plan.seed3.v1.json';alloc=read(allocation);plan=read(planpath)
 assert sha(allocation)=='ae7cab221632bf76e0bb9e70b96b686ac9739b5b971f368c250dbe168df2d60c'
 assert sha(planpath)==alloc['training_plan_sha256']=='8b3b0f25bb22ecab73857aed7f75d8afae7c92c39e51c9a76b7189a9985a17a1'
 assert sha(S/'TRAINING-SEED3-FREEZE.json')==alloc['freeze_sha256']=='9ff40fcba6c7e36d57f9ea14527efab77620947450ac3a5578a0833373101658'
 for key in ['runs','sources','lanes','limits','approved_scalar_fit','historical_input_identity']:assert alloc[key]==plan[key],key
 assert alloc['maximum_fits_authorized_now']==8 and alloc['maximum_OOF_authorized_now']==4 and alloc['WP_queries_authorized_now']==0
 for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
 for path,h in plan['review_evidence'].items():assert sha(ROOT/path)==h,path
 jobs=plan['runs'];assert len(jobs)==8
 contracts=process_contracts(plan,allocation)
 for fold in [0,1]:
  for stage,suffix in [('parent','_parent'),('head','')]:
   x=jobs[f'on_plectl_s3_f{fold}{suffix}'];y=jobs[f'on_ple_s3_f{fold}{suffix}']
   assert len(x['parsed_args'])==len(y['parsed_args'])==103
   assert {k for k in x['parsed_args'] if x['parsed_args'][k]!=y['parsed_args'][k]}==({'name'} if stage=='parent' else {'name','init','freeze'})
 result={'status':'PASS_SOURCE_ONLY_PREPARATION' if a.prepare_only else 'PASS_INDEPENDENT_SCALAR8_FIT4_OOF_INTEGRITY','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checker_sha256':sha(__file__),'allocation_sha256':sha(allocation),'plan_sha256':sha(planpath),'sources_checked':len(plan['sources']),'targets_decoded':False,'model_forward_executed':False,'GPU_used':False,'WP_queries':0}
 if a.prepare_only:
  result.update(empirical_checkpoints_read=False,prediction_arrays_read=False,dataset_arrays_read=False,
   process_contracts_prepared=len(contracts),v1_checker_sha256='7095ec7acf397ec321069f567d32388db0e542c7efe75c88565fb6661b155121')
  a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
 import numpy as np
 import torch
 torch.set_num_threads(2)
 sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(S))
 from wnn.nn import RNNModel
 from scalar_model import SCHEMA,TABLE_SEED
 same=lambda x,y:x.dtype==y.dtype and x.shape==y.shape and x.detach().cpu().numpy().tobytes()==y.detach().cpu().numpy().tobytes()
 final=read(E/'result.json');assert final['status']=='PASS_FIXED_FITS_AND_OOF_PENDING_INDEPENDENT_AUDIT' and final['failure'] is None and final['WP_queries']==0
 assert final['allocation_sha256']==sha(allocation) and final['training_plan_sha256']==sha(planpath) and final['seconds']<=plan['limits']['stage_seconds']+12
 before=read(E/'inputs-before.json');after=read(E/'inputs-after.json');assert before['status']==after['status']=='PASS' and before['files']==after['files']
 historical=read(ROOT/plan['historical_input_identity']['path']);assert sha(ROOT/plan['historical_input_identity']['path'])==plan['historical_input_identity']['sha256']
 assert set(before['files'])==set(historical['files'])
 for name,f in before['files'].items():assert f['sha256']==historical['files'][name]['sha256'] and f['bytes']==historical['files'][name]['bytes']
 for receipt in [before,after]:assert receipt['allocation_sha256']==sha(allocation) and receipt['script_sha256']==sha(S/'input_preflight.py') and not receipt['decoded_competition_arrays']
 identity={'receipt_path':str((E/'inputs-before.json').relative_to(ROOT)),'receipt_sha256':sha(E/'inputs-before.json')}
 cost=read(E/'cost-at-launch.json');expected_cost=1.25*15000*(2*plan['cost']['control_median_seconds']+3*plan['cost']['ple_median_seconds'])+5400
 assert cost['seconds']==expected_cost and cost['PASS'] and cost['seconds']<=cost['remaining_seconds'] and cost['separate_final_reserve_seconds']==4800
 expected_labels=['inputs-before','inputs-after']+list(jobs)+[name+'-check' for name in jobs]+[name+'-OOF' for name,j in jobs.items() if j['stage']=='head']
 ledger=final['jobs'];assert len(ledger)==22 and sorted(j['label'] for j in ledger)==sorted(expected_labels)
 by_label={j['label']:j for j in ledger};assert len(by_label)==22
 assert set(contracts)==set(by_label)
 ordering=verify_execution_order(by_label,plan)
 for label,r in by_label.items():
  contract=contracts[label]
  assert r['argv']==contract['argv'] and r['environment']==contract['environment'] and r['cap_seconds']==contract['cap_seconds'],(label,'process-contract')
  if contract['uses_gpu']:
   assert r['GPU_precheck']['used_MiB']<=128 and r['GPU_precheck']['free_MiB']>=plan['limits']['gpu_MiB'] and not r['GPU_precheck']['compute'],label
  else:assert r['GPU_precheck'] is None and r['peak_sampled_GPU_MiB']==0,label
  assert r==read(E/f'{label}.process.json') and r['exit_code']==0 and r['failure'] is None
  assert r['log_sha256']==sha(E/f'{label}.log') and r['samples_sha256']==sha(E/f'{label}.samples.jsonl')
  assert r['identity']['pid']==r['identity']['pgid'] and r['identity']['start_ticks']>0
  assert r['peak_sampled_aggregate_RSS_KiB']<=plan['limits']['host_RSS_GiB_per_lane']*(1<<20) and r['peak_sampled_GPU_MiB']<=plan['limits']['gpu_MiB']
  samples=[json.loads(line) for line in (E/f'{label}.samples.jsonl').read_text().splitlines()]
  if samples:
   assert max(s['aggregate_RSS_KiB'] for s in samples)==r['peak_sampled_aggregate_RSS_KiB']
   for sample in samples:
    assert sum(p['rss_KiB'] for p in sample['processes'])==sample['aggregate_RSS_KiB']
    assert all(p['pgid']==r['identity']['pgid'] for p in sample['processes'])
    if sample['GPU']:assert contract['uses_gpu'] and sample['GPU']['used_MiB']<=plan['limits']['gpu_MiB']
    else:assert not contract['uses_gpu']
  assert r['seconds']<=r['cap_seconds']+12
  if label in jobs and jobs[label]['stage']=='parent':assert datetime.datetime.fromisoformat(r['started_UTC'])<datetime.datetime(2026,10,1,6,tzinfo=datetime.timezone.utc)
 fitted=plan['approved_scalar_fit'];fit=read(ROOT/fitted['receipt_path']);assert sha(ROOT/fitted['receipt_path'])==fitted['receipt_sha256']
 assert sha(ROOT/fitted['knots_path'])==fitted['knots_sha256']
 with np.load(ROOT/fitted['knots_path'],allow_pickle=False) as z:knots=z['knots'].copy();counts=z['counts'].copy()
 base_template=RNNModel(211,128,2,0,n_out=23).state_dict();shape_base={k:tuple(v.shape) for k,v in base_template.items()}
 extras={'scalar.knots':(44,16),'scalar.counts':(44,),'scalar.raw_indices':(44,),'scalar.table':(44,16,16),'scalar.projection':(128,16)}
 generator=torch.Generator(device='cpu').manual_seed(TABLE_SEED);initial_table=torch.randn((44,16,16),generator=generator,dtype=torch.float32)
 endpoints={};checkpoint_evidence={};prediction_evidence={};learned={}
 def declared_derived(job,path):
  actual=read(path);expected=copy.deepcopy(alloc);expected['runs']={job['name']:job};expected['root_allocation_sha256']=sha(allocation);expected['input_identity']=identity
  assert actual==expected,job['name'];return actual
 for lane in plan['lanes']:
  previous=None
  for name in lane['runs']:
   job=copy.deepcopy(jobs[name]);stage=job['stage'];record=by_label[name]
   if previous:assert record['started_UTC']>=by_label[previous+'-check']['finished_UTC']
   previous=name
   if stage=='head':job['parent']['sha256']=checkpoint_evidence[job['parent_run']]['endpoint_sha256']
   derived_path=E/f'{name}.allocation.json';declared_derived(job,derived_path)
   expected_argv=['taskset','-c',lane['cpus'],str(ROOT/'.venv/bin/python'),str(S/'train.py'),'--allocation',str(derived_path),'--run',name]
   assert record['argv']==expected_argv and record['environment']['CUDA_VISIBLE_DEVICES']==str(lane['gpu']) and record['environment']['WNN_GPU_FRACTION']=='0.34375'
   assert record['GPU_precheck']['used_MiB']<=128 and not record['GPU_precheck']['compute'] and record['GPU_precheck']['free_MiB']>=5632
   assert record['cap_seconds']==plan['limits']['parent_seconds' if stage=='parent' else 'head_seconds']
   args=copy.deepcopy(job['parsed_args']);args.update(scalar_arm=job['arm'],scalar_stage=stage,scalar_allocation_sha256=sha(derived_path),scalar_training_plan_sha256=sha(planpath),scalar_parent_sha256=job.get('parent',{}).get('sha256'))
   if job['arm']=='ple':args.update(scalar_schema=SCHEMA,scalar_fit_receipt_sha256=fitted['receipt_sha256'],scalar_knots_array_sha256=fit['knots_array_sha256'],scalar_counts_array_sha256=fit['counts_array_sha256'])
   path=ROOT/job['endpoint'];epoch=3 if stage=='parent' else 1;assert sorted(f.name for f in path.parent.glob('model_ep*.pt'))==[f'model_ep{i}.pt' for i in range(epoch+1)]
   expected_shapes={**shape_base,**(extras if job['arm']=='ple' else {})};hashes={}
   for ep in range(epoch+1):
    cp=path.with_name(f'model_ep{ep}.pt');ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==ep and ck['args']==args and ck['n_in']==211 and ck['readout'] is None and not ck['readout_tilted']
    state=ck['model'];assert set(state)==set(expected_shapes)
    for key,t in state.items():assert tuple(t.shape)==expected_shapes[key] and t.dtype==(torch.int64 if key in ['scalar.counts','scalar.raw_indices'] else torch.float32) and torch.isfinite(t).all(),(name,ep,key)
    if job['arm']=='ple':
     assert state['scalar.knots'].numpy().tobytes()==knots.tobytes() and state['scalar.counts'].numpy().tobytes()==counts.tobytes()
     assert state['scalar.raw_indices'].tolist()==list(range(22))+list(range(52,74))
    if stage=='head':
     prior=endpoints[job['parent_run']]
     assert all(same(v,state[k]) for k,v in prior.items() if not k.startswith('head.'))
    hashes[cp.name]=sha(cp)
   endpoint=ck['model'];endpoints[name]=endpoint
   latest=torch.load(path.with_name('model.pt'),map_location='cpu',weights_only=False);assert set(latest)=={'model','args','n_in','readout','readout_tilted'} and latest['n_in']==211 and latest['readout'] is None and not latest['readout_tilted'] and latest['args']==args and set(latest['model'])==set(endpoint) and all(same(v,latest['model'][k]) for k,v in endpoint.items())
   history=read(path.with_name('history.json'));assert len(history)==epoch+1 and all(r['epoch']==i and r['evaluation_skipped'] and r['metric_placeholder_not_a_readout'] and r['weighted_pearson']==0 for i,r in enumerate(history))
   if stage=='parent' and job['arm']=='ple':
    assert torch.count_nonzero(endpoint['scalar.projection'])>0 and not same(endpoint['scalar.table'],initial_table)
    learned[name]={'projection_norm':float(torch.linalg.vector_norm(endpoint['scalar.projection'])),'table_change_norm':float(torch.linalg.vector_norm(endpoint['scalar.table']-initial_table))}
   if stage=='head':assert any(not same(v[:2],endpoint[k][:2]) for k,v in endpoints[job['parent_run']].items() if k.startswith('head.2.'))
   author=read(E/f'{name}.integrity.json');assert author['status']=='PASS' and author['checkpoint_sha256']==sha(path) and author['allocation_sha256']==sha(derived_path) and author['script_sha256']==sha(S/'check_fit.py')
   checkpoint_evidence[name]={'endpoint_sha256':sha(path),'saved_epoch_hashes':hashes,'args_exact':True,'all_saved_states_finite':True,'author_receipt_sha256':sha(E/f'{name}.integrity.json')}
 for fold in [0,1]:
  ctl=endpoints[f'on_plectl_s3_f{fold}_parent'];ple=endpoints[f'on_ple_s3_f{fold}_parent']
  for key in ctl:
   if 'mean' in key or 'std' in key:assert same(ctl[key],ple[key]),key
  assert any(not same(v,ple[k]) for k,v in ctl.items() if k.startswith(('inp.','gru.')))
 assert sha(ROOT/'cache/valid_targets.npz')==before['files']['cache/valid_targets.npz']['sha256'] and sha(ROOT/'cache/chain_folds.npz')==before['files']['cache/chain_folds.npz']['sha256']
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group']
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:valid=z['split']=='valid';fold_map=z['fold'];row_map=z['row_group']
 assert len(groups)==1873 and len(np.unique(groups))==1873
 for name,decl in jobs.items():
  if decl['stage']!='head':continue
  job=copy.deepcopy(decl);job['parent']['sha256']=checkpoint_evidence[job['parent_run']]['endpoint_sha256'];job['endpoint_sha256']=checkpoint_evidence[name]['endpoint_sha256'];path=E/f'{name}.oof-allocation.json';declared_derived(job,path)
  r=read((ROOT/job['prediction']).with_suffix('.inputs.json'));output=ROOT/job['prediction'];assert r['checkpoint_sha256']==job['endpoint_sha256'] and r['prediction_sha256']==sha(output)
  assert r['allocation_sha256']==sha(path) and r['training_plan_sha256']==sha(planpath) and r['script_sha256']==sha(S/'predict_scalar.py') and r['epoch']==1 and r['fold']==job['fold'] and r['seed']==3 and r['arm']==job['arm']
  assert r['fit_receipt_sha256']==fitted['receipt_sha256'] and r['targets_read'] is False and r['masks_read'] is False and r['WP_queries']==r['metrics_computed']==0
  assert r['input_hashes_before']==r['input_hashes_after']=={n:before['files'][n]['sha256'] for n in ['datasets/valid.parquet','cache/valid_targets.npz','cache/chain_folds.npz']}
  positions=np.flatnonzero(np.isin(groups,row_map[valid&(fold_map==job['fold'])]));assert len(positions)==(467 if job['fold']==0 else 469) and positions.tolist()==r['rows'] and groups[positions].tolist()==r['row_groups']
  assert hashlib.sha256(groups.tobytes()).hexdigest()==r['source_row_group_order_sha256'] and r['chain_folds_sha256']==before['files']['cache/chain_folds.npz']['sha256']
  pred=np.load(output,mmap_mode='r',allow_pickle=False);assert pred.dtype==np.float32 and pred.shape==(1873,20000,2);selected=set(positions.tolist())
  for i in range(1873):assert (np.isfinite(pred[i]).all() if i in selected else np.isnan(pred[i]).all()),(name,i)
  del pred
  lane=next(l for l in plan['lanes'] if l['fold']==job['fold']);process=by_label[name+'-OOF'];assert process['argv']==['taskset','-c',lane['cpus'],str(ROOT/'.venv/bin/python'),str(S/'predict_scalar.py'),'--allocation',str(path),'--run',name]
  assert process['started_UTC']>=max(by_label[j+'-check']['finished_UTC'] for j in lane['runs'])
  prediction_evidence[name]={'sha256':r['prediction_sha256'],'receipt_sha256':sha(output.with_suffix('.inputs.json')),'selected_rows':len(positions),'finite_selected_NaN_outside':True}
 assert len(list((P/'cache').glob('pred_on_ple*_s3_f[01]_valid.npy')))==4
 result.update(fit_count=8,OOF_count=4,saved_epoch_files_checked=24,checkpoints=checkpoint_evidence,predictions=prediction_evidence,scalar_learning=learned,root_result_sha256=sha(E/'result.json'),input_before_sha256=sha(E/'inputs-before.json'),input_after_sha256=sha(E/'inputs-after.json'),complete_job_receipts=22,exact_process_contracts_verified=22,execution_order=ordering,empirical_checkpoints_read=True,prediction_arrays_read=True,metadata_arrays_read=['valid_targets.row_group','chain_folds.split','chain_folds.fold','chain_folds.row_group'],limitations=['Opaque targetfile bytes hashed for provenance; no targets/masks decoded.','Resource monitoring sampled; no instantaneous aggregatecgroup claim.','Final-reserve timestamp comparison retains12-second monitor/query scheduling allowance, not an infinitely precise hard cutoff.','Finite selected arrays and NaNoutside confirmed, but no independent inference replay or quality calculation.'])
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checkpoints','predictions','scalar_learning']}))
if __name__=='__main__':main()
