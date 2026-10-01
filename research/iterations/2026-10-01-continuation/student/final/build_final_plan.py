"""Resolve final parent/H2 from exact retained historical log args; source-only."""
from pathlib import Path
import json,hashlib,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE.parent))
from build_plan import parser_from_source,resolve
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=parser_from_source(); defaults=resolve(parser,['--name','unused'])
old=[json.loads(line) for line in (HERE/'historical-args.txt').read_text().splitlines() if line.startswith('{')]
assert len(old)==2 and len(defaults)==103
cv=json.loads((HERE.parent/'commands.json').read_text())
jobs=[]
for ix,(historical,stage,name,epochs,cap) in enumerate(zip(old,['parent','H2'],['cont_sh256_final_s3','cont_sh256_final_s3h2'],[4,2],[2400,600])):
    assert set(historical)<=set(defaults)
    desired={**defaults,**historical,'name':name,'layers':1}
    if stage=='H2': desired['init']='cont_sh256_final_s3@ep3'
    argv=[]
    for key,value in desired.items():
        if key!='name' and value==defaults[key]:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool): assert value; argv.append(flag)
        else:argv.extend([flag,str(value)])
    assert resolve(parser,argv)==desired and len(desired)==103
    assert desired['valid_fold']==-1 and desired['eval_source']=='holdout' and desired['include_heldout_valid'] and desired['epochs']==epochs
    assert desired['hidden']==256 and desired['seed']==3 and desired['full_valid']
    delta={k:[v,desired[k]] for k,v in historical.items() if v!=desired[k]}
    assert set(delta)==({'name','layers'} if stage=='parent' else {'name','layers','init'})
    cvdiff={k:[v,desired[k]] for k,v in cv['jobs'][ix]['resolved_args'].items() if v!=desired[k]}
    assert set(cvdiff)==({'name','distill','valid_fold','eval_source','include_heldout_valid'} if stage=='parent' else {'name','init','valid_fold','eval_source','include_heldout_valid'})
    if stage=='H2':assert desired['freeze']=='inp,gru' and not desired['distill'] and not desired['ema_teacher'] and not desired['eval_ema'] and not desired['since_weight']
    jobs.append({'name':name,'stage':stage,'fold':-1,'argv':['.venv/bin/python','research/iterations/2026-10-01-overnight/training/train_aux.py',*argv],'resolved_args':desired,'historical_log_args_diff':delta,'CV_student_recipe_diff':cvdiff,'endpoint':f'artifacts/runs/{name}/model_ep{epochs-1}.pt','wall_limit_seconds':cap})
paths=[ROOT/p for p in cv['source_sha256'] if not p.endswith('predict_fixed.py')]
paths += [Path(__file__),HERE/'run_plan.py',HERE/'historical-args.txt',HERE/'inputs.json',HERE.parent/'final-teacher-availability.json']
result={'status':'PREPARED_REQUIRES_EXPLICIT_ROOT_ALLOCATION','execution_authorized':False,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'jobs':jobs,'environment':cv['environment'],'budget':{'fits':2,'OOF':0,'WP':0,'retries':0},'input_receipt':str((HERE/'inputs.json').relative_to(ROOT)),'input_receipt_sha256':sha(HERE/'inputs.json'),'recipe':'Exact prior shallow seed3 CV recipe except all-valid coverage and historical fin256_s3,fin256_s4 teachers; fixed parent EMA epoch3 then fresh H2 epoch1. No extra epoch or selection.','limits':{'parent_seconds':2400,'H2_seconds':600,'stage_seconds':2700,'GPU':7,'allocator_fraction':.25,'device_total_envelope_MiB':5632,'CPU_ids':[22,23,24,25,28,32],'host_memory_bytes':20*2**30,'swap':0},'lineage_limitation':'Original teacher cache producer epoch is unknown; root accepted retained historical causal teacher bytes without an ep3-cache assertion.'}
with (HERE/'commands.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS_SOURCE_ONLY_2_JOBS_103_ARGS','commands_sha256':sha(HERE/'commands.json'),'historical_diffs':[j['historical_log_args_diff'] for j in jobs],'CV_diffs':[j['CV_student_recipe_diff'] for j in jobs]},indent=2))
