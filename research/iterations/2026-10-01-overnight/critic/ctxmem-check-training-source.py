"""Source/argv-only independent CTX checks. Never import trainer or open datasets."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent / 'ctxmem'
ROOT = HERE.parents[3]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
plan = json.loads((P / 'execution-plan.json').read_text())
receipt = json.loads((P / 'trainer-source.json').read_text())
base = ROOT / receipt['base']
assert sha(base) == receipt['base_sha256']
transformed = base.read_text()
for before, after in receipt['changes']:
    assert transformed.count(before) == 1
    transformed = transformed.replace(before, after)
assert transformed == (P / 'train_ctx.py').read_text()
assert sha(P / 'train_ctx.py') == receipt['implementation_sha256']
assert len(receipt['changes']) == 9
tree = ast.parse(transformed)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
prefix = []
for node in main.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args':
        break
    prefix.append(node)
scope = {'argparse': argparse}
exec(compile(ast.Module(body=prefix, type_ignores=[]), '<frozen-parser>', 'exec'), scope)
parser = scope['ap']

recipe_tree = ast.parse((P / 'recipe.py').read_text())
validate = next(n for n in recipe_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_recipe')
validation_scope = {'plan': lambda: plan, 'metadata': lambda arm: {'synthetic_metadata_arm': arm}, 'sha': sha, 'HERE': P}
exec(compile(ast.Module(body=[validate], type_ignores=[]), '<frozen-validator>', 'exec'), validation_scope)
validate_recipe = validation_scope['validate_recipe']
parsed = []
rejections = 0
for job in plan['jobs']:
    args = parser.parse_args(job['train_argv'][2:])
    plain = vars(args).copy()
    assert len(plain) == 103
    parsed.append(plain)
    assert (args.hidden, args.layers, args.dropout, args.epochs, args.seed, args.valid_fold) == (128, 2, 0., 2, job['seed'], job['fold'])
    assert args.ctx_arm == job['arm'] and args.name == job['name']
    assert args.train_source == 'valid' and args.n_valid_folds == 4 and args.full_valid and args.mask_weight
    assert args.freeze == 'inp,gru' and args.loss == 'gcorr' and args.aux_struct == args.aux_next == .3
    assert args.lr == .0005 and args.wd == .01 and args.chunk == 2000 and args.batch == 16
    assert not args.eval_ema and args.ema_teacher == args.swa_frac == args.td_alpha == 0
    assert not args.max_batches and not args.profile_steps and not args.parity_dump
    assert not args.include_heldout_valid and args.precision == 'fp16' and args.fast
    validate_recipe(args, parser)
    assert args.ctx_plan_sha256 == sha(P / 'execution-plan.json')
    assert args.ctx_parent_sha256 == job['parent_sha256']
    assert args.context_memory['source_fold'] == job['fold']
    for key, value in plain.items():
        changed = copy.deepcopy(plain)
        changed[key] = not value if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else value + '_undeclared' if isinstance(value, str) else object()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                validate_recipe(argparse.Namespace(**changed), parser)
        except SystemExit as error:
            assert error.code == 2
            rejections += 1
        else:
            raise AssertionError(('undeclared argument accepted', job['name'], key))
assert len(parsed) == 8 and rejections == 824
for fold in [0, 1]:
    for seed in [3, 4]:
        pair = [dict(a) for a in parsed if a['valid_fold'] == fold and a['seed'] == seed]
        assert len(pair) == 2
        for row in pair:
            row.pop('name')
            row.pop('ctx_arm')
        assert pair[0] == pair[1]
names = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
assert 'evaluate' not in names and names.count('skip_evaluation_rng') == 1
assert names.index('validate_recipe') < names.index('verify_inputs')

manifest = plan['code_sha256']
assert manifest, 'Actor preparation not yet frozen'
for name, expected in manifest.items():
    assert sha(P / name) == expected, name
assert sha(HERE.parent / 'amendment-01.json') == plan['amendment_sha256']
assert sha(P / 'runtime-gate-result.json') == plan['native_cost_result_sha256']
assert sha(P / 'native-runtime-plan.json') == plan['native_runtime_plan_sha256']
assert len(plan['readouts']) == 4
result = {
    'status': 'SOURCE_TRANSFORM_AND_EXACT_ARGV_GUARDS_PASS',
    'checker_sha256': sha(__file__),
    'plan_sha256': sha(P / 'execution-plan.json'),
    'trainer_sha256': sha(P / 'train_ctx.py'),
    'recipe_sha256': sha(P / 'recipe.py'),
    'predictor_sha256': sha(P / 'predict_ctx.py'),
    'source_hashes_verified': len(manifest),
    'exact_base_transforms': 9,
    'jobs': 8,
    'parsed_arguments_per_job': 103,
    'undeclared_argument_rejections': rejections,
    'matched_arms_only_name_and_arm_differ': True,
    'original_loss_optimizer_AMP_scheduler_clip_unchanged': True,
    'evaluation_call_absent_from_actual_main': True,
    'validation_metadata_test_scope': 'Real AST validator with metadata constructor stub; actual memory metadata already covered by independent model/state/CUDA checks.',
    'data_or_checkpoint_arrays_opened': False,
    'GPU_jobs': 0,
    'forecast_fits': 0,
    'WP_queries': 0
}
output = HERE / 'ctxmem-training-source-result.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
