"""Target-free postcompletion checkpoint inspection; no forward, dataset, or metric."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'commands.json').read_text())
a=json.loads((HERE/'allocation-01.json').read_text())
assert sha(HERE/'commands.json')==a['commands_sha256']
result=json.loads((HERE/'execution-01/result.json').read_text())
assert result['status']=='COMPLETED_2_FINAL_FITS_NO_METRICS' and len(result['jobs'])==2
assert result['allocation_sha256']==sha(HERE/'allocation-01.json')
assert all(r['exit_code']==0 and r['cleanup_error'] is None and r['scope_state'] in ('','inactive') for r in result['jobs'])
checkpoints={}; endpoints={}; all_files={}
for job in plan['jobs']:
    run=ROOT/'artifacts/runs'/job['name']; epochs=job['resolved_args']['epochs']
    for epoch in range(epochs):
        p=run/f'model_ep{epoch}.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['epoch']==epoch and ck['n_in']==211 and ck['args']==job['resolved_args']
        assert all(torch.isfinite(v).all().item() for v in ck['model'].values())
        assert ck['model']['gru.weight_ih_l0'].shape==(768,256) and ck['model']['head.2.weight'].shape==(23,256)
        assert not any(k.endswith('_l1') for k in ck['model'])
        key=str(p.relative_to(ROOT));all_files[key]={'sha256':sha(p),'bytes':p.stat().st_size,'epoch':epoch,'finite':True}
        if epoch==epochs-1:endpoints[job['stage']]=ck
    p=run/'model.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
    assert ck['args']==job['resolved_args'] and ck['n_in']==211
    last=endpoints[job['stage']]['model'];assert ck['model'].keys()==last.keys()
    assert all(torch.equal(v,last[k]) for k,v in ck['model'].items())
    all_files[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'state_equals_fixed_endpoint':True}
    history=json.loads((run/'history.json').read_text())
    assert len(history)==epochs and all(x['epoch']==i and x['evaluation_skipped'] is True and x['metric_placeholder_not_a_readout'] is True for i,x in enumerate(history))
    all_files[str((run/'history.json').relative_to(ROOT))]={'sha256':sha(run/'history.json'),'bytes':(run/'history.json').stat().st_size}
    checkpoints[job['stage']]={'path':job['endpoint'],'sha256':sha(ROOT/job['endpoint']),'epoch':epochs-1,'args':job['resolved_args'],'all_state_values_finite':True,'saved_EMA':job['resolved_args']['eval_ema'],'source_of_EMA_identity':'Frozen trainer eval_model=ema_model.module when eval_ema; checkpoint args and source hash, not reconstructed EMA trajectory.'}
parent=endpoints['parent']['model'];head=endpoints['H2']['model']
assert parent.keys()==head.keys()
frozen=[k for k in parent if k in ('mean','std') or k.startswith(('inp.','gru.'))]
assert all(torch.equal(parent[k],head[k]) for k in frozen)
changed=[k for k in parent if k.startswith('head.') and not torch.equal(parent[k],head[k])]
assert changed and all(k in frozen or k.startswith('head.') for k in parent)
r={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_FINAL2_FIXED_ENDPOINTS_FINITE_AND_FROZEN_TRUNK','allocation_sha256':sha(HERE/'allocation-01.json'),'commands_sha256':sha(HERE/'commands.json'),'result_sha256':sha(HERE/'execution-01/result.json'),'inspection_source_sha256':sha(Path(__file__)),'checkpoints':checkpoints,'all_new_run_files':all_files,'unchanged_trunk_and_normalization':frozen,'changed_head_tensors':changed,'timings_seconds':{x['stage']:x['seconds'] for x in result['jobs']},'scope_release':{x['stage']:x['scope_state'] for x in result['jobs']},'WP_queries':0,'data_or_targets_read':False,'model_forward_called':False}
with (HERE/'fit-inspection.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'status':r['status'],'sha256':sha(HERE/'fit-inspection.json'),'endpoints':{k:v['sha256'] for k,v in checkpoints.items()}},indent=2))
