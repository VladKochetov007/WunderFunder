"""Strict fixed-screen recipe and identity guards; no metrics or alternative modes."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import torch
from memory import metadata

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def plan():
    return json.loads((HERE/'execution-plan.json').read_text())


def validate_recipe(args,parser):
    """Exact parsed namespace equality catches undeclared flags and changed defaults."""
    frozen=plan();matches=[job for job in frozen['jobs'] if job['name']==args.name]
    if len(matches)!=1:parser.error('Run name is not registered in the frozen eight-fit plan')
    job=matches[0];expected=parser.parse_args(job['train_argv'][2:])
    if vars(args)!=vars(expected):parser.error('Arguments differ from the exact frozen job')
    if args.ctx_arm!=job['arm'] or args.valid_fold!=job['fold'] or args.seed!=job['seed']:
        parser.error('Plan job metadata mismatch')
    args.context_memory=metadata(args.ctx_arm)
    args.context_memory['source_fold']=args.valid_fold
    args.evaluation_policy='none; fixed epoch1; preserve evaluation-loader RNG'
    args.ctx_plan_sha256=sha(HERE/'execution-plan.json')
    args.ctx_parent_sha256=job['parent_sha256']


def verify_inputs(args):
    """Called only inside an allocated actual fit, after argument validation."""
    frozen=plan()
    assert datetime.now(timezone.utc)<datetime.fromisoformat(frozen['budget']['cutoff_no_new_screen_jobs_UTC'].replace('Z','+00:00'))
    for name,expected in frozen['code_sha256'].items():assert sha(HERE/name)==expected,name
    assert sha(HERE/'execution-plan.json')==args.ctx_plan_sha256
    parent,_,epoch=args.init.partition('@ep');assert epoch=='3'
    path=ROOT/'artifacts/runs'/parent/'model_ep3.pt'
    assert sha(path)==args.ctx_parent_sha256,'Parent checkpoint changed'
    assert sha(ROOT/'cache/chain_folds.npz')==frozen['chain_folds_sha256']
    historical=json.loads((HERE.parent/'sam/input-provenance.json').read_text())
    for name in ('datasets/valid.parquet','cache/valid_targets.npz'):
        previous=historical['files'][name];p=ROOT/name;s=p.stat()
        assert s.st_size==previous['bytes'] and s.st_mtime_ns==previous['mtime_ns'],name
        assert str(p.resolve())==previous['resolved_path'],name
    assert not (ROOT/'artifacts/runs'/args.name).exists(),'Immutable output already exists'


def verify_dataset_identity():
    """Byte identities only; caller uses input columns and row-group arrays only."""
    result={}
    for name,expected in plan()['input_sha256'].items():
        actual=sha(ROOT/name);assert actual==expected,name
        result[name]=actual
    return result


def skip_evaluation_rng(loader):
    # Sequential DataLoader iteration consumes exactly one CPU int64 base-seed draw.
    torch.empty((),dtype=torch.int64).random_(generator=loader.generator)
