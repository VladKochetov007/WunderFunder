"""Assemble one fixed study from an interrupted prefix and its unused suffix.

Metadata and byte hashes only. Original builder/grader stay unchanged. Never fit,
load tensors or compute a score. Root executes once after both segments complete.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
B=str(HERE.relative_to(ROOT))
ORIGINAL_BUILDER_SHA='50cc4bdb28812253064be47be3214877f7865feeb84a29ee52b5dc15b4ca54da'
ORIGINAL_FAILURE="AssertionError({'uuid': 'GPU-53eaee12-1803-f8da-7399-ccdbfda91a18', 'total_MiB': 16376, 'used_MiB': 8561, 'free_MiB': 7413})"
SUFFIX_SUCCESS='COMPLETED_REMAINING_4_FITS_3_OOF_NO_METRICS'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

assert sha(HERE/'build_bindings.py')==ORIGINAL_BUILDER_SHA
sp=importlib.util.spec_from_file_location('immutable_original_binding_builder',HERE/'build_bindings.py')
O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O)
relative,path,expected_jobs=O.relative,O.path,O.expected_jobs
PRIOR,PRIOR_SHA,EVALUATOR_SHA,METRIC_SHA=O.PRIOR,O.PRIOR_SHA,O.EVALUATOR_SHA,O.METRIC_SHA
OLD_NAMES,NEW_NAMES,SUCCESS=O.OLD_NAMES,O.NEW_NAMES,O.SUCCESS

def key(j):return j['name'],j['stage']
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def pinned(rec):
    p=path(rec['path']);assert sha(p)==rec['sha256'],rec['path']
    return json.loads(p.read_text())

def remap(value,old,new):
    if isinstance(value,str):return new+value[len(old):] if value==old or value.startswith(old+'/') else value
    if isinstance(value,list):return [remap(v,old,new) for v in value]
    if isinstance(value,dict):return {k:remap(v,old,new) for k,v in value.items()}
    return value

def validate_segments(first,stopped,second,finished):
    """Pure structural checks, also exercised by generated tests."""
    original=first['jobs'];prefix=stopped['jobs'];suffix=finished['jobs']
    assert len(original)==12 and set(map(key,original))==expected_jobs()
    assert len(prefix)==5 and list(map(key,prefix))==list(map(key,original[:5]))
    assert stopped['status']=='FAILED_STOP_NO_RETRY' and stopped['failure']==ORIGINAL_FAILURE
    assert first['success_status']==SUCCESS and second['success_status']==SUFFIX_SUCCESS
    assert finished['status']==SUFFIX_SUCCESS and finished['failure'] is None
    assert len(second['jobs'])==7 and len(suffix)==7
    assert list(map(key,suffix))==list(map(key,second['jobs']))==list(map(key,original[5:]))
    for before,after in zip(original[5:],second['jobs'],strict=True):
        expected=copy.deepcopy(before)
        # Only destination paths for post-fit receipts change. argv (the actual
        # fit/inference command), timeout and every other setting stay literal.
        for field in ('post_argv','required_outputs','require_absent'):
            if field in expected:expected[field]=remap(expected[field],first['execution_directory'],second['execution_directory'])
        assert after==expected,('suffix job settings changed',key(before))
    for done,nfit,noof in ((stopped,4,1),(finished,4,3)):
        assert done['fits']==nfit and done['OOF_jobs']==noof
        assert done['metric_calls']==0 and done['retries']==0
        assert sum(j['stage'] in ('parent','H2') for j in done['jobs'])==nfit
        assert sum(j['stage']=='OOF' for j in done['jobs'])==noof
        for j in done['jobs']:
            assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[]
    joined=prefix+suffix
    assert len(set(map(key,joined)))==12 and list(map(key,joined))==list(map(key,original))
    return joined

def merge_sources(first,second,extra):
    merged=dict(first)
    for collection in (second,extra):
        for name,h in collection.items():
            assert name not in merged or merged[name]==h,('previously frozen source changed',name)
            if name not in merged:assert name.startswith(B+'/'),('new source outside campaign',name)
            merged[name]=h
    return merged

def bind_assets(workers,sources,commands):
    command_jobs={j['name']:j for j in commands['jobs']}
    assert len(commands['jobs'])==8 and set(command_jobs)=={n for n,s in expected_jobs() if s!='OOF'}
    validations={}
    for (name,stage),worker in workers.items():
        if stage=='OOF':continue
        fold=int(name[-1]);epoch=3 if stage=='parent' else 1
        matches=[rel for rel in worker['outputs'] if Path(rel).name==f'{name}.check.json']
        assert len(matches)==1
        rp=path(matches[0])
        assert rp.parent==path(worker['_execution_directory'])
        rh=sha(rp);proof=worker['outputs'][relative(rp)]
        assert proof['sha256']==rh and proof['bytes']==rp.stat().st_size
        rec=json.loads(rp.read_text())
        assert rec['status']=='PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE'
        assert rec['job']==name and rec['fold']==fold and rec['stage']==stage
        assert rec['commands_sha256']==sources[relative(HERE/'commands.json')]
        assert rec['validator_sha256']==sources[relative(HERE/'check_checkpoint.py')]
        cj=command_jobs[name]
        assert cj['stage']==stage and cj['fold']==fold and cj['expected_n_in']==211
        assert rec['resolved_arguments_count']==commands['resolved_arguments_count']==len(cj['resolved_args'])
        assert rec['all_args_match'] and rec['all_state_finite_float32'] and rec['model_pt_state_byte_equals_endpoint']
        endpoint=f'artifacts/runs/{name}/model_ep{epoch}.pt';cp=path(endpoint)
        assert cj['endpoint']==endpoint
        assert rec['checkpoint']['path']==endpoint and rec['checkpoint']['n_in']==211
        assert rec['checkpoint']['epoch']==epoch and rec['checkpoint']['sha256']==sha(cp)
        assert rec['checkpoint']['bytes']==cp.stat().st_size
        if stage=='H2':assert rec['byte_identical_frozen_tensors'] and rec['changed_head_tensors'] and rec['parent']
        validations[name]={'path':relative(rp),'sha256':rh,'checkpoint':rec['checkpoint']}
    prior=json.loads(PRIOR.read_text())
    predictions={f'{name}_f{f}':prior['predictions'][f'{name}_f{f}'] for f in (0,1) for name in OLD_NAMES}
    for f in (0,1):
        for name in NEW_NAMES:
            run=f'{name}_f{f}';worker=workers[run,'OOF']
            p=HERE/'cache'/f'pred_{run}_valid.npy';rp=p.with_suffix('.inputs.json')
            r=json.loads(rp.read_text());ph=sha(p);rh=sha(rp)
            for output,h in ((p,ph),(rp,rh)):
                proof=worker['outputs'][relative(output)]
                assert proof['sha256']==h and proof['bytes']==output.stat().st_size
            assert r['run']==run and r['fold']==f and r['epoch']==1 and r['prediction_sha256']==ph
            assert r['script_sha256']==sources[relative(HERE/'predict_fixed.py')]
            assert r['checkpoint_sha256']==validations[run]['checkpoint']['sha256']
            assert Path(r['checkpoint']).name=='model_ep1.pt' and Path(r['checkpoint']).parent.name==run
            predictions[run]={'path':relative(p),'sha256':ph,'receipt':relative(rp),
                              'receipt_sha256':rh,'checkpoint_sha256':r['checkpoint_sha256']}
    ed=ROOT/'research/iterations/2026-10-01-error-analysis/extraction-01'
    er=ed/'receipt.json';ep=ed/'scored-rows.npz';existing=json.loads(er.read_text())
    eh=existing['outputs'][ep.name]['sha256']
    assert ep.stat().st_size==existing['outputs'][ep.name]['bytes']
    return validations,predictions,prior,ep,eh,er


def assemble(plan,plan_path,plan_sha,out,binding_path):
    assert plan['execution_authorized'] is True
    original=plan['original'];suffix=plan['suffix']
    first=pinned(original['allocation']);stopped=pinned(original['result'])
    second=pinned(suffix['allocation'])
    suffix_result_path=path(suffix['result_path'])
    suffix_result_sha=sha(suffix_result_path)
    finished=json.loads(suffix_result_path.read_text())
    suffix_result={'path':suffix['result_path'],'sha256':suffix_result_sha}
    assert first['execution_authorized'] is True and second['execution_authorized'] is True
    assert first['execution_directory']==relative(path(original['result']['path']).parent)
    assert second['execution_directory']==relative(suffix_result_path.parent)
    assert first['deadline_utc']==second['deadline_utc'],'original campaign deadline must not reset'
    assert first['environment']==second['environment'],'scientific/runtime environment changed'
    a,b=dict(first['resources']),dict(second['resources'])
    for k in ('gpu_id','gpu_uuid'):a.pop(k);b.pop(k)
    assert a==b,'resource settings other than the declared GPU changed'
    assert (first['resources']['gpu_id'],second['resources']['gpu_id'])==(6,0)
    assert first['resources']['gpu_uuid']=='GPU-53eaee12-1803-f8da-7399-ccdbfda91a18'
    assert second['resources']['gpu_uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8'
    joined=validate_segments(first,stopped,second,finished)
    next_job=first['jobs'][5]
    for ending in ('.worker-started.json','.worker-result.json','.log'):
        # Original queue stopped at the pre-launch GPU guard. A suffix job may
        # exist only in its second execution directory, never as a first attempt.
        assert not (path(first['execution_directory'])/(next_job['name']+'.'+next_job['stage']+ending)).exists()
    sources=merge_sources(first['sources'],second['sources'],plan['sources'])
    assert sources[relative(Path(__file__))]==sha(__file__)
    assert sources[relative(HERE/'build_bindings.py')]==ORIGINAL_BUILDER_SHA
    assert sources[relative(HERE/'evaluate.py')]==EVALUATOR_SHA
    assert sources['scripts/fold_eval.py']==METRIC_SHA
    for rel,h in sources.items():assert sha(path(rel))==h,rel
    assert sha(PRIOR)==PRIOR_SHA
    assert set(first['sources']).issubset(second['sources'])
    assert set(second['sources'])-set(first['sources'])<=set(plan['sources'])
    # Extra source pins can only add provenance. Existing scientific sources are
    # required byte-identical above and each remaining scientific argv is exact.
    commands=json.loads((HERE/'commands.json').read_text())
    workers={};worker_provenance={};segments=[]
    for label,allocation,done,arec,rrec in (
            ('original_prefix',first,stopped,original['allocation'],original['result']),
            ('remaining_suffix',second,finished,suffix['allocation'],suffix_result)):
        assert done['allocation_sha256']==arec['sha256']
        assert done['runner_sha256']==allocation['sources'][relative(HERE/'run_queue.py')]==sha(HERE/'run_queue.py')
        for rec in allocation['input_bindings']:pinned(rec)
        allocated={key(j):j for j in allocation['jobs']}
        for job in done['jobs']:
            k=key(job);record=job['worker_result'];wp=path(record['path'])
            assert wp.parent==path(allocation['execution_directory'])
            worker=pinned(record)
            assert worker['status']=='COMPLETE' and worker['allocation_sha256']==arec['sha256']
            assert (worker['name'],worker['stage'])==k and worker['argv']==allocated[k]['argv']
            assert worker['returncode']==0
            if job['stage']!='OOF':assert worker['post_returncode']==0
            assert set(worker['outputs'])==set(allocated[k]['required_outputs'])
            assert k not in workers,'duplicate successful scientific job'
            worker['_execution_directory']=allocation['execution_directory']
            workers[k]=worker
            worker_provenance[k]={'segment':label,'allocation':arec,'worker_receipt':record}
        segments.append({'segment':label,'allocation':arec,'result':rrec,
                         'original_status':done['status'],'original_failure':done['failure'],
                         'successful_jobs':len(done['jobs']),'fits':done['fits'],'OOF_jobs':done['OOF_jobs'],
                         'gpu_id':allocation['resources']['gpu_id'],'gpu_uuid':allocation['resources']['gpu_uuid']})
    assert set(workers)==expected_jobs()
    validations,predictions,prior,ep,eh,er=bind_assets(workers,sources,commands)
    for name,rec in validations.items():
        stage='parent' if '_parent_' in name else 'H2'
        rec['worker_provenance']=worker_provenance[name,stage]
    composite_jobs=[]
    for job in joined:
        record=copy.deepcopy(job)
        record['resumption_provenance']=worker_provenance[key(job)]
        composite_jobs.append(record)
    fix=HERE/'controller-v2-fix.json'
    composite={'status':SUCCESS,'failure':None,
        'completion_type':'COMPOSITE_OF_TWO_IMMUTABLE_ALLOCATION_SEGMENTS',
        'resumption_plan':{'path':relative(plan_path),'sha256':plan_sha},
        'assembly_source_sha256':sha(__file__),'segments':segments,'jobs':composite_jobs,
        'fits':8,'OOF_jobs':4,'metric_calls':0,'retries':0,'resource_resumptions':1,
        'retry_semantics':'No scientific job failed or was rerun. Original resource guard interruption remains in segments[0]; only its never-started seven-job suffix moved GPU.',
        'pretraining_mechanical_CLI_fix':{'path':relative(fix),'sha256':sha(fix),
                                        'scope':'Separate earlier controller path fix; no fit or prediction retry.'},
        'execution_started_utc':stopped['started_utc'],'execution_finished_utc':finished['finished_utc'],
        'assembled_utc':datetime.now(timezone.utc).isoformat()}
    recipes={'candidate4':[NEW_NAMES[0],*OLD_NAMES[1:]],
             'fresh_control4':[NEW_NAMES[1],*OLD_NAMES[1:]],'current_student4':list(OLD_NAMES)}
    completion=out/'result.json'
    # Recheck frozen metadata and sources immediately before writing completion.
    for rec in (original['allocation'],original['result'],suffix['allocation'],suffix_result):
        assert sha(path(rec['path']))==rec['sha256']
    assert sha(plan_path)==plan_sha and sha(PRIOR)==PRIOR_SHA
    for rel,h in sources.items():assert sha(path(rel))==h,rel
    write(completion,composite)
    binding={'status':'ROOT_ALLOCATED_FOR_FIXED_ENSEMBLE_BACKBONE','folds':[0,1],
        'recipes':recipes,'comparisons':{'primary':['candidate4','fresh_control4'],
                                       'operational':['candidate4','current_student4']},
        'sources':sources,'data':prior['data'],'predictions':predictions,
        'root_allocation':{'path':relative(completion),'sha256':sha(completion),
                           'required_field':'status','required_value':SUCCESS},
        'existing_eda':{'path':relative(ep),'sha256':eh,'receipt':relative(er),'receipt_sha256':sha(er)},
        'checkpoint_validation_receipts':validations,
        'provenance':{'resumption_plan_path':relative(plan_path),'resumption_plan_sha256':plan_sha,
                      'segments':segments,'resource_resumptions':1,'scientific_job_retries':0,
                      'prior_bindings_path':relative(PRIOR),'prior_bindings_sha256':PRIOR_SHA,
                      'builder_sha256':sha(__file__),'original_builder_sha256':ORIGINAL_BUILDER_SHA,
                      'arrays_loaded':0,'metrics':0},
        'comparison_scope':'Original frozen recipes/grader/masks/CI unchanged. One primary candidate versus fresh control and one operational contrast versus old student4; no search.'}
    write(binding_path,binding)
    write(out/'receipt.json',{'status':'COMPLETE_RESUMED_BINDINGS_NO_METRICS',
        'outputs':{relative(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (completion,binding_path)},
        'source_sha256':sha(__file__),'plan_sha256':plan_sha,'arrays_loaded':0,'metrics':0})
    print(json.dumps({'status':'COMPLETE_RESUMED_BINDINGS_NO_METRICS',
                      'completion_sha256':sha(completion),'bindings_sha256':sha(binding_path),'metrics':0}))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--plan-sha256',required=True)
    a=ap.parse_args();assert sha(a.plan)==a.plan_sha256
    plan=json.loads(a.plan.read_text())
    out=path(plan['combined_output_directory']);binding=path(plan['binding_output'])
    assert out==HERE/'completed-study-01' and binding==HERE/'evaluation-bindings-01.json'
    assert not out.exists() and not out.is_symlink() and not binding.exists() and not binding.is_symlink()
    assert plan['sources'][relative(Path(__file__))]==sha(__file__)
    out.mkdir();write(out/'started.json',{'plan_sha256':a.plan_sha256,'source_sha256':sha(__file__),
                                        'utc':datetime.now(timezone.utc).isoformat(),'metrics':0})
    try:assemble(plan,a.plan.resolve(),a.plan_sha256,out,binding)
    except BaseException as exc:
        write(out/'failure.json',{'status':'FAILED_STOP_NO_RETRY','exception':repr(exc),'metrics':0})
        raise


if __name__=='__main__':main()
