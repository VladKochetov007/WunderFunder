"""Integrity audit only: checkpoints/history/receipts; never opens targets or predictions."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_SHA = '9057f8d60bb8e3641dcd715b8386a8d366e24baea66b44782e3320d1047a3e33'
CHAIN_SHA = '315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def frozen_plan():
    path = HERE / 'execution-plan.seed3.v1.json'
    assert sha(path) == PLAN_SHA, 'Seed3 plan changed'
    plan = json.loads(path.read_text())
    for name, expected in plan['code_sha256'].items():
        assert sha(HERE / name) == expected, f'Frozen source changed: {name}'
    assert len(plan['jobs']) == 8
    return plan


def expected_args(argv):
    """Extract only the frozen CLI parser and guard AST; never execute trainer main."""
    tree = ast.parse((HERE / 'train_sam.py').read_text())
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_sam_recipe')
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    parser_nodes = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            assert isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args'
            break
        parser_nodes.append(node)
    else:
        raise AssertionError('Frozen parser boundary missing')
    namespace = {'argparse': argparse}
    module = ast.fix_missing_locations(ast.Module(body=[guard, *parser_nodes], type_ignores=[]))
    exec(compile(module, '<frozen-parser-only>', 'exec'), namespace)
    parsed = namespace['ap'].parse_args(argv[2:])
    namespace['validate_sam_recipe'](parsed, namespace['ap'])
    if parsed.fast:
        parsed.sync_free = parsed.fused_adam = parsed.pin_memory = parsed.prop_cache = True
    return vars(parsed)


def expected_shapes():
    shapes = {'mean': [211], 'std': [211], 'inp.weight': [128, 211], 'inp.bias': [128],
              'head.0.weight': [128, 256], 'head.0.bias': [128],
              'head.2.weight': [23, 128], 'head.2.bias': [23]}
    for layer in range(2):
        for which in ('ih', 'hh'):
            shapes[f'gru.weight_{which}_l{layer}'] = [384, 128]
            shapes[f'gru.bias_{which}_l{layer}'] = [384]
    return shapes


def audit(prepare_only):
    started = time.monotonic()
    plan = frozen_plan()
    declared = {job['name']: expected_args(job['train_argv']) for job in plan['jobs']}
    for job in plan['jobs']:
        args = declared[job['name']]
        assert args['seed'] == 3 and args['valid_fold'] == job['fold'] and args['sam_stage'] == job['stage']
        assert args['name'] == job['name']
        if job['stage'] == 'head':
            assert len(job['depends_on']) == 1 and args['init'] == job['depends_on'][0] + '@ep3'
        else:
            assert args['init'] == '' and args['eval_ema'] is True
    for fold in (0, 1):
        ctl = declared[f'on_sam_pctl_s3_f{fold}']; sam = declared[f'on_sam_psam_s3_f{fold}']
        assert {k for k in ctl if ctl[k] != sam[k]} == {'name', 'sam_rho'}
        ctl = declared[f'on_sam_hctl_s3_f{fold}']; sam = declared[f'on_sam_hsam_s3_f{fold}']
        assert {k for k in ctl if ctl[k] != sam[k]} == {'name', 'init'}
    if prepare_only:
        return {'status': 'SOURCE_PREPARATION_PASS', 'plan_sha256': PLAN_SHA,
                'checker_sha256': sha(Path(__file__)), 'jobs': list(declared),
                'full_parser_argument_count': len(next(iter(declared.values()))),
                'checkpoint_files_opened': [], 'prediction_files_opened': [], 'target_files_opened': []}

    # Establish that *all* eight fits and post-fit stages have completed before
    # opening even the first checkpoint. No partial-fit audit or metric readout.
    ledger_path = HERE.parent / 'job-ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    events = {}
    for job in plan['jobs']:
        for stage, argvkey in [('train', 'train_argv'), ('predict', 'predict_argv')]:
            records = [r for r in ledger if r.get('job') == job['name'] and r.get('stage') == stage]
            begin = [r for r in records if r['event'] == 'started']
            done = [r for r in records if r['event'] == 'completed']
            assert len(begin) == len(done) == 1 and len(records) == 2, f'Incomplete/retried stage: {job["name"]}/{stage}'
            assert begin[0]['plan_sha256'] == PLAN_SHA
            assert begin[0]['argv'][-len(job[argvkey]):] == job[argvkey]
            assert done[0]['exit_code'] == 0 and not done[0]['timed_out']
            assert begin[0]['utc'] < done[0]['utc']
            log = HERE.parent / 'logs' / f'{job["name"]}.{stage}.log'
            assert sha(log) == done[0]['log_sha256']
            events[job['name'], stage] = {'begin': begin[0], 'done': done[0]}
    for job in plan['jobs']:
        for parent in job['depends_on']:
            assert events[parent, 'predict']['done']['utc'] < events[job['name'], 'train']['begin']['utc']

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    import torch
    torch.set_num_threads(2)
    shape = expected_shapes()
    checkpoints, receipts, per_job = {}, {}, []
    for job in plan['jobs']:
        name, stage = job['name'], job['stage']
        endpoint_epoch = 3 if stage == 'parent' else 1
        run = ROOT / 'artifacts' / 'runs' / name
        assert job['endpoint'] == str((run / f'model_ep{endpoint_epoch}.pt').relative_to(ROOT))
        assert sorted(p.name for p in run.glob('model_ep*.pt')) == [f'model_ep{e}.pt' for e in range(endpoint_epoch + 1)]
        history_path = run / 'history.json'
        history = json.loads(history_path.read_text())
        assert len(history) == endpoint_epoch + 1
        for epoch, entry in enumerate(history):
            assert set(entry) == {'evaluation_skipped', 'policy', 'epoch', 'eval_seconds', 'split'}
            assert entry['epoch'] == epoch and entry['evaluation_skipped'] is True
            assert entry['policy'] == 'fixed endpoint; no WP computed'
            assert str(entry['split']).startswith('vfold')
        train_log = (HERE.parent / 'logs' / f'{name}.train.log').read_text()
        assert f'FIXED_ENDPOINT {endpoint_epoch} validation metrics suppressed' in train_log
        assert 'weighted_pearson' not in train_log and 'SELECTED_FINAL' not in train_log
        saved, files = {}, []
        for epoch in range(endpoint_epoch + 1):
            path = run / f'model_ep{epoch}.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert set(ck) == {'model', 'args', 'n_in', 'epoch', 'readout', 'readout_tilted'}
            assert ck['epoch'] == epoch and ck['args'] == declared[name], f'Argument drift: {name}/ep{epoch}'
            assert ck['n_in'] == 211 and ck['readout'] is None and ck['readout_tilted'] is False
            assert {key: list(value.shape) for key, value in ck['model'].items()} == shape
            assert all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in ck['model'].values())
            assert (ck['model']['std'] > 0).all().item()
            saved[epoch] = ck
            files.append({'epoch': epoch, 'path': str(path), 'sha256': sha(path), 'all_finite': True})
        final_path = run / 'model.pt'
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        assert set(final) == set(saved[endpoint_epoch]) and set(final['model']) == set(saved[endpoint_epoch]['model'])
        assert final['epoch'] == endpoint_epoch and final['args'] == declared[name]
        assert final['n_in'] == 211 and final['readout'] is None and final['readout_tilted'] is False
        assert all(value.dtype == torch.float32 for value in final['model'].values())
        assert all(torch.equal(value, final['model'][key]) for key, value in saved[endpoint_epoch]['model'].items())
        checkpoints[name] = saved
        item = {'name': name, 'stage': stage, 'fold': job['fold'], 'endpoint': endpoint_epoch,
                'args_exact_all_fields': True, 'finite_all_epochs': True, 'history_WP_readouts': 0,
                'model_pt_equals_fixed_endpoint': True, 'model_pt_sha256': sha(final_path),
                'history_sha256': sha(history_path), 'epoch_files': files,
                'saved_weights': 'EMA by exact frozen source and eval_ema=True' if stage == 'parent' else 'ordinary H2 model',
                'training_started_utc': events[name, 'train']['begin']['utc'],
                'training_completed_utc': events[name, 'train']['done']['utc']}
        if stage == 'head':
            parent_name = job['depends_on'][0]
            parent = checkpoints[parent_name][3]['model']
            head_checks = []
            for epoch, ck in saved.items():
                state = ck['model']
                unchanged = [k for k in parent if torch.equal(parent[k], state[k])]
                changed = [k for k in parent if not torch.equal(parent[k], state[k])]
                assert all(k.startswith('head.') for k in changed)
                assert all(k in unchanged for k in parent if not k.startswith('head.'))
                assert not torch.equal(parent['head.0.weight'], state['head.0.weight'])
                assert not torch.equal(parent['head.2.weight'][:2], state['head.2.weight'][:2])
                head_checks.append({'epoch': epoch, 'exact_parent_backbone_and_buffers': True,
                                    'changed_head_keys': changed, 'unchanged_keys': unchanged})
            assert not torch.equal(saved[0]['model']['head.0.weight'], saved[1]['model']['head.0.weight'])
            assert not torch.equal(saved[0]['model']['head.2.weight'][:2], saved[1]['model']['head.2.weight'][:2])
            item['lineage'] = {'parent': parent_name, 'parent_epoch': 3,
                               'parent_checkpoint_sha256': sha(ROOT / 'artifacts/runs' / parent_name / 'model_ep3.pt'),
                               'checks_by_head_epoch': head_checks, 'head_changed_between_epochs': True}
            receipt_path = HERE.parent / 'cache' / f'pred_{name}_valid.inputs.json'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['run'] == name and receipt['fold'] == job['fold'] and receipt['epoch'] == 1
            assert receipt['checkpoint'] == str(run / 'model_ep1.pt')
            assert receipt['checkpoint_sha256'] == files[-1]['sha256']
            assert receipt['script_sha256'] == plan['code_sha256']['predict_fixed.py']
            assert receipt['chain_folds_sha256'] == CHAIN_SHA and receipt['metrics_computed'] == 0
            assert receipt['nonselected_rows'] == 'NaN sentinel'
            expected_prediction = HERE.parent / 'cache' / f'pred_{name}_valid.npy'
            assert receipt['prediction_path'] == str(expected_prediction)
            assert re.fullmatch('[0-9a-f]{64}', receipt['prediction_sha256'])
            assert len(receipt['rows']) == len(receipt['row_groups']) > 0
            assert len(set(receipt['rows'])) == len(receipt['rows'])
            assert len(set(receipt['row_groups'])) == len(receipt['row_groups'])
            assert receipt['rows'] == sorted(receipt['rows'])
            # Only metadata stat; neither prediction bytes nor array headers are opened.
            assert expected_prediction.is_file() and expected_prediction.stat().st_size > 0
            receipts[name] = receipt
            item['prediction_provenance'] = {'receipt_sha256': sha(receipt_path),
                                            'script_sha256': receipt['script_sha256'],
                                            'checkpoint_sha256': receipt['checkpoint_sha256'],
                                            'recorded_prediction_sha256': receipt['prediction_sha256'],
                                            'prediction_bytes_rehashed': False,
                                            'predictions_opened': False, 'metrics_computed': 0,
                                            'source_row_group_order_sha256': receipt['source_row_group_order_sha256'],
                                            'selected_row_count': len(receipt['rows'])}
        per_job.append(item)
    for fold in (0, 1):
        a, b = [receipts[f'on_sam_h{arm}_s3_f{fold}'] for arm in ('ctl', 'sam')]
        for key in ('rows', 'row_groups', 'source_row_group_order_sha256', 'chain_folds_sha256'):
            assert a[key] == b[key]
    assert len({r['source_row_group_order_sha256'] for r in receipts.values()}) == 1
    assert set(receipts['on_sam_hctl_s3_f0']['rows']).isdisjoint(receipts['on_sam_hctl_s3_f1']['rows'])
    assert set(receipts['on_sam_hctl_s3_f0']['row_groups']).isdisjoint(receipts['on_sam_hctl_s3_f1']['row_groups'])
    return {'status': 'PASS', 'plan_sha256': PLAN_SHA, 'checker_sha256': sha(Path(__file__)),
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'jobs': per_job,
            'all_eight_jobs_completed_before_checkpoint_audit': True,
            'checkpoint_files_opened': 32, 'prediction_files_opened': [], 'target_files_opened': [],
            'new_WP_queries': 0, 'wall_seconds': time.monotonic() - started,
            'limitations': ['EMA identity follows frozen source and arguments; no independent raw/EMA trace was saved to reconstruct numerical EMA updates.',
                            'H2 full parent loading follows frozen source and init argument; exact frozen-backbone equality is checked numerically at both saved epochs.',
                            'Prediction SHA256 is checked as recorded provenance only; prediction bytes are intentionally not opened or rehashed.']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    result = audit(args.prepare_only)
    output = HERE / ('actual-fits-preparation.json' if args.prepare_only else 'actual-fits-seed3-integrity.json')
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
