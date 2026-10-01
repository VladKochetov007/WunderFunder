"""Bind the single completed paired study. Metadata/hashes only, no tensor reads.

Root executes this only after eight fits and four OOF outputs have completed.
The grader independently checks the completed pipeline and all input bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PRIOR=ROOT/'research/iterations/2026-10-01-continuation/student/bindings-01.json'
PRIOR_SHA='ba5aa4cc9aa1364fb22c294abaf7f4cf09b04b1ea264bb80f87adec43f2d3450'
EVALUATOR_SHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
METRIC_SHA='d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'
OLD_NAMES=('cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')
NEW_NAMES=('eb_candidate_s3_h2','eb_control_s3_h2')
SUCCESS='COMPLETED_8_FITS_4_OOF_NO_METRICS'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def relative(path):return str(Path(path).resolve().relative_to(ROOT.resolve()))


def path(rel):
    p=Path(rel)
    assert not p.is_absolute() and '..' not in p.parts and str(p)==rel
    return ROOT/p


def expected_jobs():
    fits={(f'eb_{arm}_s3_parent_f{fold}','parent')
          for arm in ('candidate','control') for fold in (0,1)}
    fits|={(f'eb_{arm}_s3_h2_f{fold}','H2')
           for arm in ('candidate','control') for fold in (0,1)}
    return fits|{(name,'OOF') for name,stage in fits if stage=='H2'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allocation',type=Path,required=True)
    ap.add_argument('--allocation-sha256',required=True)
    ap.add_argument('--source-sha256',required=True)
    ap.add_argument('--completion',type=Path,default=HERE/'execution-01/result.json')
    ap.add_argument('--checkpoint-receipt-dir',type=Path,default=HERE/'execution-01')
    ap.add_argument('--output',type=Path,default=HERE/'evaluation-bindings-01.json')
    a=ap.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(HERE.resolve())
    assert a.completion.resolve().is_relative_to(HERE.resolve())
    assert a.checkpoint_receipt_dir.resolve().is_relative_to(HERE.resolve())
    assert sha(__file__)==a.source_sha256 and sha(a.allocation)==a.allocation_sha256
    assert sha(PRIOR)==PRIOR_SHA and sha(HERE/'evaluate.py')==EVALUATOR_SHA
    allocation=json.loads(a.allocation.read_text())
    assert allocation['execution_authorized'] is True and allocation['success_status']==SUCCESS
    assert allocation['execution_directory']==relative(a.completion.parent)
    sources=dict(allocation['sources'])
    assert sources[relative(HERE/'evaluate.py')]==EVALUATOR_SHA
    assert sources[relative(Path(__file__))]==a.source_sha256
    assert sources['scripts/fold_eval.py']==METRIC_SHA
    assert relative(HERE/'commands.json') in sources
    for rel,h in sources.items():assert sha(path(rel))==h,rel
    commands=json.loads((HERE/'commands.json').read_text())
    command_jobs={j['name']:j for j in commands['jobs']}
    assert len(commands['jobs'])==8 and set(command_jobs)=={n for n,s in expected_jobs() if s!='OOF'}
    for binding in allocation['input_bindings']:
        assert sha(path(binding['path']))==binding['sha256'],binding['path']
    done=json.loads(a.completion.read_text())
    assert done['status']==SUCCESS and done['allocation_sha256']==a.allocation_sha256
    assert done['runner_sha256']==sources[relative(HERE/'run_queue.py')]==sha(HERE/'run_queue.py')
    assert done['fits']==8 and done['OOF_jobs']==4 and done['metric_calls']==0 and done['retries']==0
    assert done['failure'] is None and len(done['jobs'])==12
    allocated={(j['name'],j['stage']):j for j in allocation['jobs']}
    assert len(allocation['jobs'])==12 and set(allocated)==expected_jobs()
    jobs={(j['name'],j['stage']):j for j in done['jobs']}
    assert set(jobs)==expected_jobs()
    workers={}
    for key,j in jobs.items():
        assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[]
        wr=j['worker_result'];wp=path(wr['path']);assert sha(wp)==wr['sha256']
        worker=json.loads(wp.read_text())
        assert worker['status']=='COMPLETE' and worker['allocation_sha256']==a.allocation_sha256
        assert (worker['name'],worker['stage'])==key and worker['argv']==allocated[key]['argv']
        assert worker['returncode']==0
        assert set(worker['outputs'])==set(allocated[key]['required_outputs'])
        workers[key]=worker
    validations={}
    for (name,stage),worker in workers.items():
        if stage=='OOF':continue
        fold=int(name[-1]);epoch=3 if stage=='parent' else 1
        rp=a.checkpoint_receipt_dir/f'{name}.check.json'
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
    recipes={'candidate4':[NEW_NAMES[0],*OLD_NAMES[1:]],
             'fresh_control4':[NEW_NAMES[1],*OLD_NAMES[1:]],'current_student4':list(OLD_NAMES)}
    result={'status':'ROOT_ALLOCATED_FOR_FIXED_ENSEMBLE_BACKBONE','folds':[0,1],
        'recipes':recipes,'comparisons':{'primary':['candidate4','fresh_control4'],
                                       'operational':['candidate4','current_student4']},
        'sources':sources,'data':prior['data'],'predictions':predictions,
        'root_allocation':{'path':relative(a.completion),'sha256':sha(a.completion),
                           'required_field':'status','required_value':SUCCESS},
        'existing_eda':{'path':relative(ep),'sha256':eh,'receipt':relative(er),'receipt_sha256':sha(er)},
        'checkpoint_validation_receipts':validations,
        'provenance':{'allocation_path':relative(a.allocation),'allocation_sha256':a.allocation_sha256,
                      'prior_bindings_path':relative(PRIOR),'prior_bindings_sha256':PRIOR_SHA,
                      'builder_sha256':sha(__file__),'arrays_loaded':0,'metrics':0},
        'comparison_scope':'One primary candidate versus fresh matched control; one operational contrast versus old student4. Fixed equal quarters, no standalone/weight/slot/epoch search.'}
    assert sha(a.allocation)==a.allocation_sha256 and sha(PRIOR)==PRIOR_SHA and sha(__file__)==a.source_sha256
    for rel,h in sources.items():assert sha(path(rel))==h,rel
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'bindings':str(a.output),'sha256':sha(a.output),'metrics':0}))


if __name__=='__main__':main()
