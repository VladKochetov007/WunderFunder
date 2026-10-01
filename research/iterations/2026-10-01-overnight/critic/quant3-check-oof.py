"""Independent allocated Q3 integrity audit; never decode targets or compute scores."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import zipfile

HERE = Path(__file__).resolve().parent
P = HERE.parent
Q = P / 'quant3'
D = Q / 'oof-execution'
ROOT = P.parents[2]
ALLOCATION_SHA = 'a98412b6903dd027acdcd92b4b1dbdfcb147347ed0d00a26b7c6532409d3cedf'
FP16_NATIVE_SHA = 'dd6e59b2d75518ee06fcbe3ba49f8450b3ee83695fa5692e4b8edaf86d7c8bad'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def stamp(s):
    return datetime.datetime.fromisoformat(s).timestamp()


def preflight():
    assert sha(Q / 'oof-allocation.json') == ALLOCATION_SHA
    a = read(Q / 'oof-allocation.json')
    f = read(Q / 'FREEZE.json')
    assert sha(Q / 'FREEZE.json') == a['freeze_sha256']
    assert sha(Q / 'preflight-plan.md') == a['preflight_plan_sha256']
    assert sha(HERE / 'quant3-runtime-independent-review.json') == a['runtime_review_sha256']
    assert a['folds'] == [0, 1] and a['modes'] == ['quant', 'fp16']
    assert a['outputs'] == ['on_q3quant', 'on_q3fp16']
    assert not a['avx2_identity_audit_authorized'] and not a['new_blends_or_compositions_authorized']
    for group in ['source_and_binaries', 'actor_receipts', 'independent_receipts']:
        for path, expected in f[group].items():
            assert sha(ROOT / path) == expected, path
    for path, expected in a['sources'].items():
        assert sha(Q / path) == expected
    assert sha(ROOT / f['zip']) == a['zip_sha256'] == f['zip_sha256']
    assert a['resources']['fold_CPU_affinity'] == {'0': [36], '1': [37]}
    assert a['resources']['forecast_fits'] == a['resources']['GPU_jobs'] == 0
    return a, f


def audit(a, f):
    # No model, prediction or metadata arrays may be opened before all jobs and
    # the controller's final full input-identity pass have completed.
    summary = read(D / 'summary.json')
    assert summary['status'] == 'COMPLETE_PENDING_ROOT_PROVENANCE_AUDIT'
    assert summary['allocation_sha256'] == ALLOCATION_SHA
    assert summary['orchestrator_sha256'] == sha(D / 'driver.py')
    assert not summary['targets_read'] and not summary['masks_read']
    assert summary['metrics_computed'] == 0 and not summary['AVX2_audit_performed']
    identity = {}
    for phase in ['before', 'after']:
        r = read(D / f'input-identity-{phase}.json')
        assert r['allocation_sha256'] == ALLOCATION_SHA and not r['targets_or_masks_decoded']
        assert set(r['inputs']) == set(a['input_identity'])
        for name, record in r['inputs'].items():
            assert record['expected'] == record['observed'] == a['input_identity'][name]
            assert record['match'] is True
            path = ROOT / name
            current = path.stat()
            assert current.st_size == record['observed']['bytes']
            assert current.st_mtime_ns == record['observed']['mtime_ns']
            assert str(path.resolve()) == record['observed']['resolved_path']
        identity[phase] = r
    assert stamp(identity['before']['utc']) < stamp(identity['after']['utc']) <= stamp(summary['utc'])
    source = read(D / 'source-identity-before.json')
    assert source['allocation_sha256'] == ALLOCATION_SHA
    assert source['freeze_sha256'] == a['freeze_sha256'] and source['zip_sha256'] == a['zip_sha256']
    assert source['orchestrator_sha256'] == sha(D / 'driver.py')
    expected_sources = {k: v for group in ['source_and_binaries', 'actor_receipts', 'independent_receipts'] for k, v in f[group].items()}
    assert set(source['entries']) == set(expected_sources)
    for name, r in source['entries'].items():
        assert r['expected'] == r['observed'] == expected_sources[name]

    ledger = lines(D / 'execution-ledger.jsonl')
    assert [r['event'] for r in ledger] == ['lane_start', 'lane_start', 'lane_end', 'lane_end']
    assert [r['fold'] for r in ledger[:2]] == [0, 1] and [r['fold'] for r in ledger[2:]] == [0, 1]
    resource_records = []
    for fold in [0, 1]:
        start, end = ledger[fold], ledger[fold + 2]
        assert end['exit_code'] == 0 and stamp(end['utc']) >= stamp(start['utc'])
        assert stamp(start['utc']) > stamp(identity['before']['utc'])
        assert stamp(end['utc']) < stamp(identity['after']['utc'])
        assert start['argv'] == [str(ROOT / '.venv/bin/python'), str(D / 'driver.py'), '--lane', str(fold)]
        lane = lines(D / f'lane{fold}.jsonl')
        assert [(r['event'], r['mode']) for r in lane] == [(event, mode) for mode in ['prepare', 'quant', 'fp16'] for event in ['start', 'end']]
        for i, mode in enumerate(['prepare', 'quant', 'fp16']):
            s, e = lane[2 * i], lane[2 * i + 1]
            assert s['fold'] == e['fold'] == fold and e['exit_code'] == 0
            assert s['CPU_affinity'] == [36 + fold] and s['RLIMIT_AS'] == [8 << 30, 8 << 30]
            assert s['threads_environment'] == {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
            expected_command = [str(ROOT / '.venv/bin/python'), str(Q / 'prepare_oof.py'), '--fold', str(fold), '--allocation', str(Q / 'oof-allocation.json')] if mode == 'prepare' else ['/home/grinchenkoeg/scorer_image/run.sh', str(Q / 'predict_oof.py'), '--manifest', str(Q / 'oof' / f'fold{fold}' / f'{mode}.checkpoint.json')]
            usage_path = D / f'fold{fold}-{mode}.time.txt'
            assert s['argv'] == ['/usr/bin/time', '-v', '-o', str(usage_path), *expected_command]
            assert sha(D / f'fold{fold}-{mode}.log') == e['log_sha256']
            assert sha(usage_path) == e['resource_receipt_sha256']
            usage = usage_path.read_text()
            assert 'Exit status: 0' in usage
            peak = int(next(line.split(':', 1)[1].strip() for line in usage.splitlines() if 'Maximum resident set size' in line))
            assert peak < 8 * (1 << 20)
            assert stamp(s['utc']) >= stamp(start['utc']) and stamp(e['utc']) >= stamp(s['utc'])
            resource_records.append({'fold': fold, 'mode': mode, 'wall_seconds': e['wall_seconds'], 'maximum_process_RSS_KiB': peak, 'time_receipt_sha256': sha(usage_path)})
        lane_seconds = stamp(lane[-1]['utc']) - stamp(start['utc'])
        assert lane_seconds < 3600
    guards = lines(D / 'watchdog-v2.jsonl')
    assert guards[0]['event'] == 'start' and guards[0]['source_sha256'] == sha(D / 'watchdog_v2.py')
    assert not any(r['event'] in ['guard_kill', 'IDENTITY_MISMATCH_NO_SIGNAL'] for r in guards)
    sampled_RSS = {fold: max(r['aggregate_rss_bytes'] for r in guards if r['event'] == 'sample' and r['fold'] == fold) for fold in [0, 1]}
    assert all(v < 8 << 30 for v in sampled_RSS.values())

    # Dependency/lineage checks above are complete before the first array read.
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    with np.load(ROOT / 'cache/valid_targets.npz', allow_pickle=False) as z:
        groups = z['row_group']  # No target/scoring array is decoded.
    assert groups.shape == (1873,) and len(np.unique(groups)) == 1873
    group_sha = hashlib.sha256(groups.tobytes()).hexdigest()
    assert group_sha == '008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7'
    with np.load(ROOT / 'cache/chain_folds.npz', allow_pickle=False) as z:
        cf_group, cf_split, cf_fold, cf_chain = z['row_group'], z['split'], z['fold'], z['chain']
    for chain in np.unique(cf_chain[cf_split == 'valid']):
        assert len(np.unique(cf_fold[(cf_split == 'valid') & (cf_chain == chain)])) == 1
    old_integrity = read(P / 'training/actual-fit-checks.json')
    old_by_name = {r['name']: r for r in old_integrity['records']}
    old_plan = read(P / 'training/plan.json')
    old_jobs = {r['name']: r for r in old_plan['jobs']}
    parent_copies = read(P / 'parent-copy-receipt.json')
    expected_families = ['on_ctld256', 'on_ctl128s3', 'on_ctlc500']
    output_paths = [P / 'cache' / f'pred_on_q3{mode}_f{fold}_valid.npy' for fold in [0, 1] for mode in ['quant', 'fp16']]
    assert sorted(P.joinpath('cache').glob('pred_on_q3*_valid.npy')) == sorted(output_paths)
    assert not list(P.joinpath('cache').glob('pred_on_q3*.partial'))
    children_checked = []
    jobs_checked = []
    for fold in [0, 1]:
        chosen = cf_group[(cf_split == 'valid') & (cf_fold == fold)]
        positions = np.flatnonzero(np.isin(groups, chosen))
        assert len(positions) == [467, 469][fold]
        quant = read(Q / 'oof' / f'fold{fold}' / 'quant.checkpoint.json')
        fp16 = read(Q / 'oof' / f'fold{fold}' / 'fp16.checkpoint.json')
        assert quant['children'] == fp16['children'] and len(quant['children']) == 3
        for i, child in enumerate(quant['children']):
            name = f'{expected_families[i]}_f{fold}'
            assert child['run'] == name and child['fold'] == fold and child['epoch'] == 1
            assert child['coef_float64'] == 1 / 3 and child['coef_float32_hex'] == np.float32(1 / 3).tobytes().hex()
            checkpoint = ROOT / 'artifacts/runs' / name / 'model_ep1.pt'
            export = Q / 'oof' / f'fold{fold}' / f'weights_{i}.npz'
            assert Path(child['checkpoint']) == checkpoint and Path(child['export']) == export
            assert sha(checkpoint) == child['checkpoint_sha256'] == old_by_name[name]['checkpoint_sha256']
            assert sha(export) == child['export_sha256']
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = ck['args']
            assert ck['epoch'] == 1 and ck['n_in'] == 211 and ck['readout'] is None and not ck['readout_tilted']
            assert cfg['valid_fold'] == fold and cfg['epochs'] == 2 and cfg['seed'] == 3
            assert cfg['hidden'] == [256, 128, 128][i] and cfg['chunk'] == [2000, 2000, 500][i]
            assert cfg['layers'] == 2 and cfg['dropout'] == 0 and cfg['freeze'] == 'inp,gru'
            assert cfg['aux_loss_scale'] == 1 and cfg['aux_struct'] == cfg['aux_next'] == .3
            assert cfg['init'] == old_jobs[name]['parent']
            parent_name = cfg['init'].split('@ep')[0]
            copied = next(r for r in parent_copies if r['copy'].endswith(f'/artifacts/runs/{parent_name}/model_ep3.pt'))
            assert copied['sha256'] == old_by_name[name]['parent_sha256']
            with np.load(export, allow_pickle=False) as z:
                assert set(z.files) == set(ck['model']) == set(child['arrays'])
                for key in z.files:
                    array = z[key]
                    expected = ck['model'][key].detach().cpu().float().numpy()
                    if key in ['head.2.weight', 'head.2.bias']:
                        expected = expected[:2]
                    assert array.dtype == np.float32 and np.isfinite(array).all()
                    assert np.array_equal(array, expected), (name, key)
                    meta = child['arrays'][key]
                    assert meta['dtype'] == str(array.dtype) and meta['shape'] == list(array.shape)
                    assert meta['sha256'] == hashlib.sha256(array.tobytes()).hexdigest()
                    if key == 'std':
                        assert (array > 0).all()
            children_checked.append({'run': name, 'checkpoint_sha256': child['checkpoint_sha256'], 'export_sha256': child['export_sha256'], 'array_count': len(child['arrays']), 'exact_export_vs_checkpoint': True, 'parent_sha256': copied['sha256']})
            del ck, array, expected
        for mode, manifest in [('quant', quant), ('fp16', fp16)]:
            manifest_path = Q / 'oof' / f'fold{fold}' / f'{mode}.checkpoint.json'
            assert manifest['schema'] == 'q3-composite-checkpoint-v1' and manifest['kind'] == 'ensemble_recipe_manifest_not_torch_checkpoint'
            assert manifest['mode'] == mode and manifest['fold'] == fold and manifest['epoch'] == 1
            assert manifest['allocation_sha256'] == ALLOCATION_SHA and Path(manifest['allocation']) == Q / 'oof-allocation.json'
            assert manifest['source_row_group_order_sha256'] == group_sha
            assert manifest['chain_folds_sha256'] == a['input_identity']['cache/chain_folds.npz']['sha256']
            assert manifest['data_reader_source_sha256'] == a['input_identity']['src/wnn/data.py']['sha256']
            assert manifest['predictor_source_sha256'] == a['sources']['predict_oof.py']
            assert manifest['exporter_source_sha256'] == a['sources']['prepare_oof.py']
            assert manifest['zip_sha256'] == a['zip_sha256'] and manifest['native_cpu_level'] == 'v3'
            assert manifest['sum'] == 'original member order float32 fused multiply-add per member; coefficients float32(1/3)'
            assert manifest['same_composition'] and not manifest['targets_read']
            native = Path(manifest['native'])
            assert sha(native) == manifest['native_sha256']
            if mode == 'quant':
                assert manifest['quant_backend'] == 'vnni'
                assert manifest['native_sha256'] == f['source_and_binaries']['research/iterations/2026-10-01-overnight/quant3/native/build/v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so']
                with zipfile.ZipFile(ROOT / f['zip']) as z:
                    assert hashlib.sha256(z.read(manifest['zip_native_entry'])).hexdigest() == manifest['native_sha256']
            else:
                assert manifest['quant_backend'] is None and manifest['zip_native_entry'] is None
                assert manifest['native_sha256'] == FP16_NATIVE_SHA
            output = P / 'cache' / f'pred_on_q3{mode}_f{fold}_valid.npy'
            meta_path = output.with_suffix('.json')
            assert not output.with_suffix('.inputs.json').exists()
            meta = read(meta_path)
            assert meta['run'] == f'on_q3{mode}' and meta['fold'] == fold and meta['epoch'] == 1
            assert Path(meta['checkpoint']) == manifest_path and meta['checkpoint_sha256'] == sha(manifest_path)
            assert meta['checkpoint_kind'] == manifest['kind'] and meta['child_checkpoints'] == manifest['children']
            assert meta['rows'] == positions.tolist() and meta['row_groups'] == groups[positions].tolist()
            assert meta['source_row_group_order_sha256'] == group_sha and meta['chain_folds_sha256'] == manifest['chain_folds_sha256']
            assert meta['native_sha256'] == manifest['native_sha256'] and meta['zip_sha256'] == a['zip_sha256']
            assert meta['source_sha256'] == a['sources']['predict_oof.py'] and meta['sum'] == manifest['sum']
            assert meta['coefficients'] == [1 / 3] * 3 and meta['nonselected_rows'] == 'NaN'
            assert not meta['targets_read'] and not meta['masks_read'] and meta['metrics_computed'] == 0
            assert 'avx2_output_identity_audit' not in meta and Path(meta['prediction_path']) == output
            before = output.stat()
            prediction = np.load(output, mmap_mode='r', allow_pickle=False)
            assert prediction.shape == (1873, 20000, 2) and prediction.dtype == np.float32
            selected_set = set(positions.tolist())
            for row_index in range(1873):
                row = prediction[row_index]
                if row_index in selected_set:
                    assert np.isfinite(row).all(), (mode, fold, row_index, 'nonfinite selected')
                else:
                    assert np.isnan(row).all(), (mode, fold, row_index, 'non-NaN outside fold')
            del row
            prediction._mmap.close()
            del prediction
            gc.collect()
            assert sha(output) == meta['prediction_sha256']
            after = output.stat()
            assert before.st_size == after.st_size == 299680128 and before.st_mtime_ns == after.st_mtime_ns
            jobs_checked.append({'fold': fold, 'mode': mode, 'sequences': len(positions), 'finite_selected_rows': len(positions) * 20000, 'NaN_outside_sequences': 1873 - len(positions), 'prediction_sha256': meta['prediction_sha256'], 'receipt_sha256': sha(meta_path), 'composite_sha256': sha(manifest_path), 'native_sha256': manifest['native_sha256']})
    assert not torch.cuda.is_initialized()
    return {
        'status': 'Q3_FOUR_OUTPUTS_INDEPENDENT_INTEGRITY_PASS',
        'allocation_sha256': ALLOCATION_SHA,
        'freeze_sha256': a['freeze_sha256'],
        'zip_sha256': a['zip_sha256'],
        'input_identity_before_sha256': sha(D / 'input-identity-before.json'),
        'input_identity_after_sha256': sha(D / 'input-identity-after.json'),
        'execution_summary_sha256': sha(D / 'summary.json'),
        'execution_ledger_sha256': sha(D / 'execution-ledger.jsonl'),
        'guard_log_sha256': sha(D / 'watchdog-v2.jsonl'),
        'children': children_checked,
        'jobs': jobs_checked,
        'process_resources': resource_records,
        'sampled_group_RSS_max_bytes': sampled_RSS,
        'root_quality_gate': 'Eligible only for the two exact allocation readouts; no metric computed by this checker.',
        'scope_limits': ['Only metadata NPZ arrays row_group/split/fold/chain decoded; target/mask arrays never decoded.', 'Predictions scanned only for shape/dtype/finiteness/sentinels and hashes, not forecast statistics or metric.', 'Per-process RLIMIT_AS and sampled group RSS are distinct from aggregate cgroup enforcement.', 'Pinned one CPU and thread environment verified from launch receipts; OS thread count was not independently sampled here.', 'AVX2 recipe manifests may exist from frozen exporter, but no AVX2 prediction process/output/audit was authorized or observed.', 'Native VNNI runtime qualification remains hardware-specific; no portable runtime or fresh-holdout claim.'],
        'forecast_fits': 0,
        'WP_queries': 0,
        'targets_read': False,
        'masks_read': False,
        'GPU_initialized': False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    output = HERE / ('quant3-oof-checker-preparation.json' if args.prepare_only else 'quant3-oof-independent-integrity.json')
    assert not output.exists()
    a, f = preflight()
    if args.prepare_only:
        result = {'status': 'CHECKER_SOURCE_DEPENDENCIES_PASS', 'allocation_sha256': ALLOCATION_SHA, 'empirical_arrays_opened': False, 'WP_queries': 0}
    else:
        assert (D / 'summary.json').is_file() and (D / 'input-identity-after.json').is_file(), 'Not all producers/input checks are complete'
        try:
            result = audit(a, f)
        except BaseException as error:
            output.write_text(json.dumps({'status': 'FAIL', 'error': repr(error), 'checker_sha256': sha(__file__), 'WP_queries': 0}, indent=2) + '\n')
            raise
    result.update(checker_sha256=sha(__file__), seconds=time.monotonic() - started, max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
