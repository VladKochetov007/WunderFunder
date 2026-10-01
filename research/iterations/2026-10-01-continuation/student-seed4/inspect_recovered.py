"""Postcompletion seed4 state/lineage checks only; no inference, targets, or metric."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'commands.json').read_text());a=json.loads((HERE/'allocation-01.json').read_text())
assert sha(HERE/'commands.json')==a['commands_sha256']
r=json.loads((HERE/'execution-01/result.json').read_text())
assert r['status']=='FAILED_STOP_NO_RETRY' and len(r['jobs'])==5
assert r['allocation_sha256']==sha(HERE/'allocation-01.json')
assert all(x['exit_code']==0 and x['cleanup_error'] is None and x['scope_state'] in ('','inactive') for x in r['jobs'][:4])
assert [(x['name'],x['stage']) for x in r['jobs'][:4]]==[('cont_sh256_s4_f0','parent'),('cont_sh256_s4h2_f0','H2'),('cont_sh256_s4h2_f0','OOF'),('cont_sh256_s4_f1','parent')]
assert r['jobs'][4]['exit_code']==-15 and r['jobs'][4]['stage']=='H2' and r['jobs'][4]['scope_state']=='inactive'
recovery=json.loads((HERE/'recovery-01/result.json').read_text());ra=json.loads((HERE/'recovery-allocation-01.json').read_text())
assert recovery['status']=='PASS_INFRASTRUCTURE_RECOVERY_H2_F1_OOF_F1_NO_METRICS' and recovery['metrics_computed']==0
assert recovery['allocation_sha256']==sha(HERE/'recovery-allocation-01.json') and recovery['runner_sha256']==sha(HERE/'run_recovery.py')==ra['runner_sha256']
assert recovery['original_result_sha256']==ra['original_result_sha256']==sha(HERE/'execution-01/result.json')
assert ra['commands_sha256']==sha(HERE/'commands.json') and ra['execution_authorized']
assert len(recovery['jobs'])==2 and [x['stage'] for x in recovery['jobs']]==['H2','OOF']
assert all(x['exit_code']==0 and x['failure'] is None and x['scope_state'] in ('','inactive') for x in recovery['jobs'])
for rec,argv in zip(recovery['jobs'],[plan['jobs'][-1]['argv'],plan['jobs'][-1]['OOF_argv']]):assert rec['argv'][-len(argv):]==argv
assert recovery['parent_epoch3_sha256']==ra['parent_epoch3_sha256']==sha(ROOT/'artifacts/runs/cont_sh256_s4_f1/model_ep3.pt')
endpoints={};files={};pairs={};predictions={}
for job in plan['jobs']:
 run=ROOT/'artifacts/runs'/job['name'];epochs=job['resolved_args']['epochs']
 for epoch in range(epochs):
  p=run/f'model_ep{epoch}.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['epoch']==epoch and ck['n_in']==211 and ck['args']==job['resolved_args']
  assert ck['args']['seed']==4 and ck['args']['layers']==1
  assert all(torch.isfinite(v).all().item() for v in ck['model'].values())
  assert ck['model']['gru.weight_ih_l0'].shape==(768,256) and ck['model']['head.2.weight'].shape==(23,256)
  assert not any(k.endswith('_l1') for k in ck['model'])
  files[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'epoch':epoch,'finite':True}
  if epoch==epochs-1:endpoints[job['name']]=ck
 p=run/'model.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
 assert ck['args']==job['resolved_args'] and ck['n_in']==211
 last=endpoints[job['name']]['model'];assert ck['model'].keys()==last.keys()
 assert all(torch.equal(v,last[k]) for k,v in ck['model'].items())
 files[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'state_equals_fixed_endpoint':True}
 history=json.loads((run/'history.json').read_text())
 assert len(history)==epochs and all(x['epoch']==i and x['evaluation_skipped'] is True and x['metric_placeholder_not_a_readout'] is True for i,x in enumerate(history))
 files[str((run/'history.json').relative_to(ROOT))]={'sha256':sha(run/'history.json'),'bytes':(run/'history.json').stat().st_size}
for fold in (0,1):
 pn=f'cont_sh256_s4_f{fold}';hn=f'cont_sh256_s4h2_f{fold}'
 parent=endpoints[pn]['model'];head=endpoints[hn]['model'];assert parent.keys()==head.keys()
 frozen=[k for k in parent if k in ('mean','std') or k.startswith(('inp.','gru.'))]
 assert all(torch.equal(parent[k],head[k]) for k in frozen)
 changed=[k for k in parent if k.startswith('head.') and not torch.equal(parent[k],head[k])]
 assert changed and all(k in frozen or k.startswith('head.') for k in parent)
 pairs[str(fold)]={'parent_epoch3_sha256':files[f'artifacts/runs/{pn}/model_ep3.pt']['sha256'],'H2_epoch1_sha256':files[f'artifacts/runs/{hn}/model_ep1.pt']['sha256'],'unchanged_trunk_and_normalization':frozen,'changed_head_tensors':changed}
 p=HERE.parent/'cache'/f'pred_{hn}_valid.npy';q=p.with_suffix('.inputs.json');m=json.loads(q.read_text())
 assert m['run']==hn and m['fold']==fold and m['epoch']==1 and m['metrics_computed']==0
 assert Path(m['checkpoint'])==ROOT/f'artifacts/runs/{hn}/model_ep1.pt' and m['checkpoint_sha256']==pairs[str(fold)]['H2_epoch1_sha256']
 assert m['script_sha256']==sha(HERE/'predict_fixed.py') and Path(m['prediction_path'])==p
 assert len(m['rows'])==len(m['row_groups'])==len(set(m['rows']))
 assert p.is_file() and p.stat().st_size==299680128
 predictions[hn]={'path':str(p.relative_to(ROOT)),'receipt':str(q.relative_to(ROOT)),'receipt_sha256':sha(q),'producer_reported_prediction_sha256':m['prediction_sha256'],'checkpoint_sha256':m['checkpoint_sha256'],'fold_rows':len(m['rows']),'array_body_not_read_by_this_checker':True,'full_array_hash_and_exact_rowmap_will_be_verified_by_frozen_evaluator':True}
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_SEED4_CV4_AND_OOF2_WITH_EXPLICIT_INFRASTRUCTURE_RECOVERY','allocation_sha256':sha(HERE/'allocation-01.json'),'commands_sha256':sha(HERE/'commands.json'),'root_result_sha256':sha(HERE/'execution-01/result.json'),'original_result_status':r['status'],'recovery_result_sha256':sha(HERE/'recovery-01/result.json'),'recovery_allocation_sha256':sha(HERE/'recovery-allocation-01.json'),'original_successful_steps':4,'original_interrupted_H2_attempts':1,'recovery_successful_steps':2,'source_sha256':sha(Path(__file__)),'all_new_run_files':files,'paired_folds':pairs,'OOF':predictions,'all_model_states_finite':True,'all103args_match':True,'epoch_history_metrics_skipped':True,'WP_queries':0,'targets_decoded':False,'model_forward_called':False}
with (HERE/'fit-inspection-recovery.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'status':out['status'],'sha256':sha(HERE/'fit-inspection-recovery.json')}))
