"""All-eight fit integrity after all sixteen stages; never reads predictions/targets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import os

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def expected_args(argv):
    tree=ast.parse((HERE/'train_ctx.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[]
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
        nodes.append(node)
    namespace={'argparse':argparse};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<parser-only>','exec'),namespace)
    args=namespace['ap'].parse_args(argv[2:])
    from recipe import validate_recipe
    validate_recipe(args,namespace['ap'])
    if args.fast:args.sync_free=args.fused_adam=args.pin_memory=args.prop_cache=True
    return vars(args)


def audit(prepare_only):
    path=HERE/'execution-plan.json';plan=json.loads(path.read_text());plan_sha=sha(path)
    for name,expected in plan['code_sha256'].items():assert sha(HERE/name)==expected,name
    assert len(plan['jobs'])==8
    expected={job['name']:expected_args(job['train_argv']) for job in plan['jobs']}
    for fold in (0,1):
        for seed in (3,4):
            pair=[j for j in plan['jobs'] if j['fold']==fold and j['seed']==seed];assert {j['arm'] for j in pair}=={'current','memory'}
            a,b=(expected[j['name']] for j in pair)
            assert {key for key in a if a[key]!=b[key]}=={'name','ctx_arm','context_memory'}
    if prepare_only:return {'status':'SOURCE_PREPARATION_PASS','plan_sha256':plan_sha,'jobs':list(expected),'actual_data_read':False}
    ledger=[json.loads(line) for line in (HERE.parent/'job-ledger.jsonl').read_text().splitlines() if line.strip()]
    for job in plan['jobs']:
        for stage in ('train','predict'):
            events=[r for r in ledger if r.get('job')==job['name'] and r.get('stage')==stage]
            assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='completed'
            assert events[0]['plan_sha256']==plan_sha and events[0]['argv'][-len(job[stage+'_argv']):]==job[stage+'_argv']
            assert events[1]['exit_code']==0 and events[1]['timed_out'] is False
            assert sha(HERE.parent/'logs'/f"{job['name']}.{stage}.log")==events[1]['log_sha256']
    import torch
    import numpy as np
    from memory import fixed_projection
    torch.set_num_threads(2);records=[];states={}
    assert sha(ROOT/'cache/chain_folds.npz')==plan['chain_folds_sha256']
    with np.load(ROOT/'cache/chain_folds.npz') as cf:
        cf_group,cf_split,cf_fold=cf['row_group'],cf['split'],cf['fold']
    with np.load(ROOT/'cache/valid_targets.npz') as archive:cache_groups=archive['row_group']
    for job in plan['jobs']:
        parent_path=ROOT/job['parent_checkpoint'];assert sha(parent_path)==job['parent_sha256']
        parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert parent['epoch']==3
        assert parent['n_in']==211 and parent['args']['seed']==job['seed'] and parent['args']['valid_fold']==job['fold']
        assert parent['args']['hidden']==128 and parent['args']['layers']==2
        directory=ROOT/'artifacts/runs'/job['name'];history=json.loads((directory/'history.json').read_text())
        assert len(history)==2 and sorted(f.name for f in directory.glob('model_ep*.pt'))==['model_ep0.pt','model_ep1.pt']
        for epoch,row in enumerate(history):
            assert set(row)=={'evaluation_skipped','policy','epoch','eval_seconds','split'}
            assert row['evaluation_skipped'] is True and row['epoch']==epoch and row['policy']=='fixed endpoint; no WP computed'
        log=(HERE.parent/'logs'/f"{job['name']}.train.log").read_text()
        assert 'FIXED_ENDPOINT 1 validation metrics suppressed' in log and 'weighted_pearson' not in log
        files=[];endpoint=None
        for filename,epoch in [('model_ep0.pt',0),('model_ep1.pt',1),('model.pt',1)]:
            checkpoint=directory/filename;ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert set(ck)=={'model','args','n_in','epoch','readout','readout_tilted'}
            assert ck['args']==expected[job['name']] and ck['epoch']==epoch and ck['n_in']==211
            assert ck['readout'] is None and ck['readout_tilted'] is False
            weights=ck['model'];assert set(weights)==set(parent['model'])|{'ctx_projection','head_extra.weight'}
            assert all(v.dtype==torch.float32 and torch.isfinite(v).all().item() for v in weights.values())
            assert all(v.shape==weights[key].shape for key,v in parent['model'].items())
            assert all(torch.equal(v,weights[key]) for key,v in parent['model'].items() if not key.startswith('head.'))
            assert torch.equal(weights['ctx_projection'],torch.from_numpy(fixed_projection()))
            assert weights['head_extra.weight'].shape==(128,32) and torch.count_nonzero(weights['head_extra.weight']).item()>0
            assert any(not torch.equal(v,weights[key]) for key,v in parent['model'].items() if key.startswith('head.'))
            if filename=='model.pt':assert all(torch.equal(value,endpoint[key]) for key,value in weights.items())
            if filename=='model_ep1.pt':endpoint={k:v.clone() for k,v in weights.items()}
            files.append({'file':filename,'sha256':sha(checkpoint),'epoch':epoch})
        states[job['name']]=endpoint
        receipt_path=(ROOT/job['prediction']).with_suffix('.inputs.json');r=json.loads(receipt_path.read_text())
        assert r['checkpoint_sha256']==files[1]['sha256'] and r['plan_sha256']==plan_sha
        assert r['script_sha256']==plan['code_sha256']['predict_ctx.py']
        assert all(r[k]==job[k] for k in ('fold','seed','arm')) and r['epoch']==1
        assert r['targets_read'] is False and r['masks_read'] is False and r['WP_queries']==0
        assert r['feature_metadata']==expected[job['name']]['context_memory']
        assert r['chain_folds_sha256']==plan['chain_folds_sha256']
        assert r['input_hashes_before']==r['input_hashes_after']==plan['input_sha256']
        assert len(r['rows'])==len(r['row_groups'])>0 and len(set(r['rows']))==len(r['rows'])
        selected=cf_group[(cf_split=='valid')&(cf_fold==job['fold'])]
        positions=np.flatnonzero(np.isin(cache_groups,selected))
        assert r['rows']==positions.tolist() and r['row_groups']==cache_groups[positions].tolist()
        assert r['source_row_group_order_sha256']==hashlib.sha256(cache_groups.tobytes()).hexdigest()
        assert len(r['prediction_sha256'])==64 and (ROOT/job['prediction']).exists()
        records.append({'job':job['name'],'parent_sha256':job['parent_sha256'],'files':files,
                        'backbone_normalization_projection_bitexact':True,'head_changed_extra_nonzero':True,
                        'all_finite':True,'prediction_receipt_sha256':sha(receipt_path),
                        'prediction_content_hash_from_receipt_only':r['prediction_sha256']})
    for fold in (0,1):
        for seed in (3,4):
            names=[j['name'] for j in plan['jobs'] if j['fold']==fold and j['seed']==seed]
            assert any(not torch.equal(states[names[0]][key],states[names[1]][key]) for key in states[names[0]] if key.startswith(('head.','head_extra.')))
    return {'status':'PASS','plan_sha256':plan_sha,'fits':8,'stages':16,'records':records,
            'prediction_files_opened':False,'target_arrays_decoded':False,
            'metadata_only_arrays':['row_group','split','fold'],'WP_queries':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    output=HERE/('actual-fit-checker-preparation.json' if args.prepare_only else 'actual-fit-integrity.json');assert not output.exists()
    try:result=audit(args.prepare_only)
    except BaseException as error:
        output.write_text(json.dumps({'status':'FAIL','error':repr(error),'checker_sha256':sha(Path(__file__))},indent=2)+'\n');raise
    result['checker_sha256']=sha(Path(__file__));output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'receipt_sha256':sha(output)}))


if __name__=='__main__':main()
