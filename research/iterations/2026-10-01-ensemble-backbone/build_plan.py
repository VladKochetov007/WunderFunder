"""Resolve the fixed eight recipes using literal argparse AST; no Torch or data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'research/iterations/2026-10-01-continuation/student/commands.json'
TRAINER = HERE / 'training/train_ensemble.py'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parser_from_source():
    p = argparse.ArgumentParser(add_help=False)
    for n in ast.walk(ast.parse(TRAINER.read_text())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument':
            kw = {k.arg: ({'int': int, 'float': float, 'str': str}[k.value.id]
                         if k.arg == 'type' else ast.literal_eval(k.value)) for k in n.keywords}
            p.add_argument(*[ast.literal_eval(x) for x in n.args], **kw)
    return p


def resolved(p, argv):
    d = vars(p.parse_args(argv))
    if d['fast']:
        d.update(sync_free=True, fused_adam=True, pin_memory=True, prop_cache=True)
    return d


def main():
    assert sha(BASE) == 'f0804cda8b2dde75490496e2014898352ce205fe54837f8c7cbcdfa7823ce30a'
    base = json.loads(BASE.read_text()); parser = parser_from_source(); jobs = []
    for fold in (0, 1):
        for arm in ('control', 'candidate'):
            parent = f'eb_{arm}_s3_parent_f{fold}'
            for stage in ('parent', 'H2'):
                old = next(j for j in base['jobs'] if j['fold'] == fold and j['stage'] == stage)
                name = parent if stage == 'parent' else f'eb_{arm}_s3_h2_f{fold}'
                argv = list(old['argv']); argv[1] = str(TRAINER.relative_to(ROOT))
                argv[argv.index('--name') + 1] = name
                if stage == 'H2':
                    argv[argv.index('--init') + 1] = parent + '@ep3'
                argv += ['--ensemble-objective', arm, '--partner-bindings',
                         str(HERE.relative_to(ROOT) / f'partner-bindings_f{fold}.json')]
                args = resolved(parser, argv[2:]); scientific = {k: v for k, v in args.items() if k not in ('ensemble_objective', 'partner_bindings')}
                diff = {k: [v, scientific[k]] for k, v in old['resolved_args'].items() if v != scientific[k]}
                assert set(diff) == ({'name'} if stage == 'parent' else {'name', 'init'})
                assert len(args) == 105 and len(scientific) == len(old['resolved_args']) == 103
                job = dict(name=name, arm=arm, stage=stage, fold=fold, argv=argv, resolved_args=args,
                           reference_job=old['name'], scientific_argument_changes=diff,
                           expected_n_in=211, endpoint=f'artifacts/runs/{name}/model_ep{3 if stage == "parent" else 1}.pt',
                           wall_limit_seconds=3600 if stage == 'parent' else 1200)
                if stage == 'H2':
                    job.update(dependency=parent + '@ep3', OOF_wall_limit_seconds=1200,
                               OOF_argv=['.venv/bin/python', str(HERE.relative_to(ROOT) / 'predict_fixed.py'), name,
                                         '--fold', str(fold), '--epoch', '1', '--batch', '16', '--output-dir',
                                         str(HERE.relative_to(ROOT) / 'cache')])
                job['check_argv'] = ['.venv/bin/python', str(HERE.relative_to(ROOT) / 'check_checkpoint.py'), name,
                                     '--commands', str(HERE.relative_to(ROOT) / 'commands.json'), '--output',
                                     str(HERE.relative_to(ROOT) / f'execution-01/{name}.check.json')]
                jobs.append(job)
    for fold in (0, 1):
        for stage in ('parent', 'H2'):
            c, t = [j['resolved_args'] for j in jobs if j['fold'] == fold and j['stage'] == stage]
            assert {k for k in c if c[k] != t[k]} == ({'name', 'ensemble_objective'} if stage == 'parent' else {'name', 'init', 'ensemble_objective'})
    paths = [BASE, ROOT / 'research/iterations/2026-10-01-overnight/training/train_aux.py',
             TRAINER, HERE / 'training/ensemble_support.py', HERE / 'predict_fixed.py',
             HERE / 'check_checkpoint.py', Path(__file__), ROOT / 'scripts/train_rnn.py',
             *[ROOT / f'src/wnn/{x}.py' for x in ('nn', 'resources', 'data', 'features', 'splits', 'outcomes', 'propensity')]]
    result = dict(status='PREPARED_FIXED_RECIPES_ROOT_ALLOCATION_REQUIRED', resolved_arguments_count=105,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                  budget=dict(fits=8, H2_OOF_jobs=4, retries=0, actor_metrics=0),
                  scientific_change='Candidate supervised main prediction is ordered F32 equal-quarter blend; multiply ONLY main distill-MSE/gcorr by4. Control consumes identical partners but keeps original loss. EMA/aux unchanged.',
                  trajectory_caveat='Same initialized model and pre-iterator CPU RNG/row-list hashes are logged; CUDA kernels need not be bitwise deterministic. H2 starts from its own arm parent, intentionally different after training.',
                  payload='Each batch appends independent float32[T,3,2] partners after optional propensity; no target-column overload.',
                  jobs=jobs)
    (HERE / 'commands.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(jobs=len(jobs), arguments=105, commands_sha256=sha(HERE / 'commands.json'))))


if __name__ == '__main__':
    main()
