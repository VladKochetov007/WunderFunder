"""Post-completion binding adapter v2; original allocation-bound builder stays frozen.

Reads receipts/metadata and hashes files; never loads prediction/target/model tensors,
computes a metric, or authorizes training. The root supplies its exact success value
and validator receipt directory from the frozen runner interface.
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
EVALUATOR_SHA='c0ea2c2b122f5ab0b758942bc2068cce3089dfa2b8d18295fc6818fc48749d33'
OLD_NAMES=('cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')
NEW='cont_onsh256_s3_h2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def relative(path):
    # Outputs and local receipts use physical campaign paths; registered historical
    # inputs retain their existing lexical repository paths in the prior manifest.
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allocation',type=Path,required=True)
    ap.add_argument('--allocation-sha256',required=True)
    ap.add_argument('--source-sha256',required=True)
    ap.add_argument('--completion',type=Path,default=HERE/'execution-01/result.json')
    ap.add_argument('--completion-success-value',default='COMPLETED_4_FITS_2_OOF_NO_METRICS',
                    choices=['COMPLETED_4_FITS_2_OOF_NO_METRICS'])
    ap.add_argument('--checkpoint-receipt-dir',type=Path,default=HERE/'execution-01')
    ap.add_argument('--output',type=Path,default=HERE/'evaluation-bindings-01.json')
    a=ap.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(HERE.resolve())
    assert sha(__file__)==a.source_sha256
    assert a.completion.resolve().is_relative_to(HERE.resolve())
    assert a.checkpoint_receipt_dir.resolve().is_relative_to(HERE.resolve())
    assert sha(a.allocation)==a.allocation_sha256 and sha(PRIOR)==PRIOR_SHA
    assert sha(HERE/'evaluate.py')==EVALUATOR_SHA
    allocation=json.loads(a.allocation.read_text())
    assert allocation['execution_authorized'] is True
    commands_path=HERE/'commands.json'
    commands=json.loads(commands_path.read_text())
    assert allocation['commands_sha256']==sha(commands_path)
    sources=dict(commands['source_sha256'])
    for name,h in allocation['source_sha256'].items():
        assert name not in sources or sources[name]==h
        sources[name]=h
    sources.setdefault('scripts/fold_eval.py','d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0')
    assert sources[relative(HERE/'evaluate.py')]==EVALUATOR_SHA
    # This pre-readout adapter was prepared separately after the training allocation
    # froze its original builder. Do not mutate that running allocation/source set.
    assert relative(Path(__file__)) not in sources
    sources[relative(Path(__file__))]=a.source_sha256
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    done=json.loads(a.completion.read_text())
    assert a.completion_success_value.startswith('COMPLETE'), 'Explicit successful root status required'
    assert done['status']==a.completion_success_value
    assert done['allocation_sha256']==a.allocation_sha256 and done['commands_sha256']==sha(commands_path)
    assert done['runner_sha256']==allocation['runner_sha256']==sha(HERE/'run.py')
    assert done['fits']==4 and done['OOF_jobs']==2 and done['metric_calls']==0 and done['retries']==0
    assert done['failure'] is None and len(done['jobs'])==6
    assert [j['stage'] for j in done['jobs']]==['parent','H2','OOF','parent','H2','OOF']
    for job in done['jobs']:
        assert job['exit_code']==0 and job['failure'] is None and job['remaining_scope_pids']==[]
        wr=job['worker_result'];wp=ROOT/wr['path']
        assert sha(wp)==wr['sha256']
        worker=json.loads(wp.read_text())
        assert worker['status']=='COMPLETE' and worker['allocation_sha256']==a.allocation_sha256
    assert len(commands['jobs'])==4
    validations={}
    for job in commands['jobs']:
        rp=a.checkpoint_receipt_dir/f"{job['name']}.check.json"
        rec=json.loads(rp.read_text())
        assert rec['status']=='PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE'
        assert rec['job']==job['name'] and rec['fold']==job['fold'] and rec['stage']==job['stage']
        assert rec['commands_sha256']==sha(commands_path)
        assert rec['validator_sha256']==sha(HERE/'check_checkpoint.py')
        assert rec['all103_args_match'] and rec['all_state_finite_float32'] and rec['model_pt_state_byte_equals_endpoint']
        cp=ROOT/job['endpoint']
        assert rec['checkpoint']['path']==job['endpoint'] and rec['checkpoint']['n_in']==323
        assert rec['checkpoint']['sha256']==sha(cp)
        if job['stage']=='H2':assert rec['byte_identical_frozen_tensors'] and rec['changed_head_tensors'] and rec['parent']
        validations[job['name']]={'path':relative(rp),'sha256':sha(rp),'checkpoint':rec['checkpoint']}
    prior=json.loads(PRIOR.read_text())
    predictions={f'{name}_f{f}':prior['predictions'][f'{name}_f{f}'] for f in (0,1) for name in OLD_NAMES}
    for f in (0,1):
        run=f'{NEW}_f{f}'
        p=HERE/'cache'/f'pred_{run}_valid.npy';rp=p.with_suffix('.inputs.json')
        r=json.loads(rp.read_text());ph=sha(p)
        assert r['run']==run and r['fold']==f and r['epoch']==1
        assert r['prediction_sha256']==ph
        assert r['script_sha256']==sha(HERE/'predict_fixed.py')
        assert r['checkpoint_sha256']==validations[run]['checkpoint']['sha256']
        assert Path(r['checkpoint']).name=='model_ep1.pt' and Path(r['checkpoint']).parent.name==run
        predictions[run]={'path':relative(p),'sha256':ph,'receipt':relative(rp),
                          'receipt_sha256':sha(rp),'checkpoint_sha256':r['checkpoint_sha256']}
    ed=ROOT/'research/iterations/2026-10-01-error-analysis/extraction-01'
    er=ed/'receipt.json';existing=json.loads(er.read_text())
    ep=ed/'scored-rows.npz';eh=existing['outputs'][ep.name]['sha256']
    assert ep.stat().st_size==existing['outputs'][ep.name]['bytes']
    # E cache full hash and all old prediction hashes are checked by evaluate.py
    # before scoring. Builder reuses their immutable provenance without tensor reads.
    result={'status':'ROOT_ALLOCATED_FOR_FIXED_ONORM_REPLACEMENT','folds':[0,1],
        'recipes':{'candidate4':[NEW,*OLD_NAMES[1:]],'current_student4':list(OLD_NAMES)},
        'sources':sources,'data':prior['data'],'predictions':predictions,
        'root_allocation':{'path':relative(a.completion),'sha256':sha(a.completion),
                           'required_field':'status','required_value':a.completion_success_value},
        'existing_eda':{'path':relative(ep),'sha256':eh,'receipt':relative(er),'receipt_sha256':sha(er)},
        'checkpoint_validation_receipts':validations,
        'provenance':{'allocation_path':relative(a.allocation),'allocation_sha256':a.allocation_sha256,
                      'prior_bindings_path':relative(PRIOR),'prior_bindings_sha256':PRIOR_SHA,
                      'builder_sha256':sha(__file__),'arrays_loaded':0,'metrics':0},
        'source_preparation_correction':'Separate v2 fixes allocation_sha256 and runner .check.json receipt interface. Original allocation-bound build_bindings.py is unchanged and was never executed. No training recipe, checkpoint, prediction, metric implementation or active allocation changed.',
        'comparison_scope':'One fixed equal-quarter replacement; no standalone gate, fifth member, epoch or weight search.'}
    assert sha(a.allocation)==a.allocation_sha256 and sha(PRIOR)==PRIOR_SHA and sha(__file__)==a.source_sha256
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'bindings':str(a.output),'sha256':sha(a.output),'metrics':0}))


if __name__=='__main__':main()
