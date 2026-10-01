"""Completed CTX metadata/source/receipt audit only; no model or prediction decoding."""
from pathlib import Path
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
P = HERE.parent
C = P / 'ctxmem'
ROOT = P.parents[2]
read = lambda p: json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


plan = read(C / 'execution-plan.json')
allocation = read(C / 'training-allocation.json')
result = read(C / 'actual-fit-integrity.json')
review = read(HERE / 'ctxmem-training-preparation-review-v2.json')
assert sha(C / 'execution-plan.json') == allocation['execution_plan_sha256'] == result['plan_sha256'] == review['plan_sha256'] == 'fec1e50f26bd2888c6a70df07e2a4fe999d747f47a1287a2a6dd2eb1fea2988b'
assert sha(P / 'amendment-02.json') == allocation['amendment02_sha256'] == review['prefit_supplement_sha256']
assert sha(HERE / 'ctxmem-training-preparation-review-v2.json') == allocation['critic_training_review_sha256']
assert sha(HERE / 'ctxmem-runtime-independent-review.json') == allocation['critic_runtime_review_sha256']
for name, expected in plan['code_sha256'].items():
    assert sha(C / name) == expected, name
assert result['status'] == 'PASS' and result['fits'] == 8 and result['stages'] == 16
assert result['checker_sha256'] == sha(C / 'check_actual_fits.py')
assert result['WP_queries'] == 0 and not result['prediction_files_opened'] and not result['target_arrays_decoded']
assert plan['readouts'] == review['exact_four_readout_strings']
source_check = read(HERE / 'ctxmem-training-source-result-v2.json')
assert source_check['plan_sha256'] == result['plan_sha256'] and source_check['exact_four_readout_strings_verified']
for anchor in source_check['H2_anchor_lineage']:
    path = P / 'cache' / f"pred_{anchor['run']}_valid.inputs.json"
    assert sha(path) == anchor['receipt_sha256']
    r = read(path)
    assert r['checkpoint_sha256'] == anchor['checkpoint_sha256'] and r['prediction_sha256'] == anchor['prediction_sha256_from_receipt']
    assert sha(r['checkpoint']) == anchor['checkpoint_sha256'] and r['epoch'] == 1

ledger = [json.loads(s) for s in (P / 'job-ledger.jsonl').read_text().splitlines() if s.strip()]
events = [r for r in ledger if r.get('job', '').startswith('on_ctx')]
assert len(events) == 32 and {r['job'] for r in events} == {j['name'] for j in plan['jobs']}
records = {r['job']: r for r in result['records']}
handoff = []
for index, job in enumerate(plan['jobs']):
    r = records[job['name']]
    assert r['parent_sha256'] == job['parent_sha256']
    assert r['backbone_normalization_projection_bitexact'] and r['head_changed_extra_nonzero'] and r['all_finite']
    assert [(x['file'], x['epoch']) for x in r['files']] == [('model_ep0.pt', 0), ('model_ep1.pt', 1), ('model.pt', 1)]
    for item in r['files']:
        assert sha(ROOT / 'artifacts/runs' / job['name'] / item['file']) == item['sha256']
    for stage in ['train', 'predict']:
        own = [e for e in events if e['job'] == job['name'] and e['stage'] == stage]
        assert [e['event'] for e in own] == ['started', 'completed']
        start, end = own
        assert start['plan_sha256'] == result['plan_sha256'] and start['gpu'] == 6 + index % 2
        assert start['free_MiB'] >= 6200
        prefix = ['systemd-run', '--user', '--scope', '--quiet', '--unit', f"wunder-{job['name']}-{stage}", '-p', 'MemoryMax=20G', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=600%', '--', 'nice', '-n', '10', '/usr/bin/time', '-v']
        assert start['argv'] == prefix + job[stage + '_argv']
        assert end['exit_code'] == 0 and not end['timed_out']
        assert sha(P / 'logs' / f"{job['name']}.{stage}.log") == end['log_sha256']
        assert datetime.datetime.fromisoformat(start['utc']) > datetime.datetime.fromisoformat(allocation['created_utc'])
        assert datetime.datetime.fromisoformat(end['utc']) > datetime.datetime.fromisoformat(start['utc'])
    pred_receipt = (ROOT / job['prediction']).with_suffix('.inputs.json')
    meta = read(pred_receipt)
    assert sha(pred_receipt) == r['prediction_receipt_sha256']
    assert meta['checkpoint_sha256'] == r['files'][1]['sha256']
    assert meta['prediction_sha256'] == r['prediction_content_hash_from_receipt_only']
    assert meta['plan_sha256'] == result['plan_sha256'] and meta['script_sha256'] == plan['code_sha256']['predict_ctx.py']
    assert meta['epoch'] == 1 and all(meta[k] == job[k] for k in ['fold', 'seed', 'arm'])
    assert meta['input_hashes_before'] == meta['input_hashes_after'] == plan['input_sha256']
    assert not meta['targets_read'] and not meta['masks_read'] and meta['WP_queries'] == meta['metrics_computed'] == 0
    reference = read(P / 'cache' / f"pred_on_ctl128s3_f{job['fold']}_valid.inputs.json")
    assert meta['rows'] == reference['rows'] and meta['row_groups'] == reference['row_groups']
    assert meta['source_row_group_order_sha256'] == reference['source_row_group_order_sha256']
    assert meta['chain_folds_sha256'] == reference['chain_folds_sha256'] == plan['chain_folds_sha256']
    assert len(meta['rows']) == [467, 469][job['fold']]
    handoff.append({'job': job['name'], 'checkpoint_sha256': meta['checkpoint_sha256'], 'prediction_receipt_sha256': sha(pred_receipt), 'prediction_sha256_from_receipt_only': meta['prediction_sha256'], 'sequences': len(meta['rows'])})
monitor = [json.loads(s) for s in (C / 'gpu-budget-watch.jsonl').read_text().splitlines() if s.strip()]
assert monitor[0]['event'] == 'started' and monitor[0]['plan_sha256'] == result['plan_sha256']
assert monitor[-1]['event'] == 'finished' and not monitor[-1]['stopped']
assert set(monitor[-1]['peaks_MiB']) == set(records)
assert max(monitor[-1]['peaks_MiB'].values()) <= allocation['resources']['GPU_MiB_max']
assert min(monitor[-1]['sample_counts'].values()) > 0
out = {'status': 'CTX_COMPLETED_HANDOFF_METADATA_PASS', 'plan_sha256': result['plan_sha256'], 'allocation_sha256': sha(C / 'training-allocation.json'), 'amendment02_sha256': sha(P / 'amendment-02.json'), 'author_actual_integrity_sha256': sha(C / 'actual-fit-integrity.json'), 'checker_sha256': sha(__file__), 'source_hashes_verified': 33, 'jobs': handoff, 'exact_16_stage_commands_successful': True, 'checkpoint_files_hash_bound': 24, 'H2_reference_checkpoint_hashes_verified': 6, 'corrected_four_reference_strings_verified': True, 'GPU_sampled_peak_MiB': max(monitor[-1]['peaks_MiB'].values()), 'GPU_monitor_sha256': sha(C / 'gpu-budget-watch.jsonl'), 'authority': 'Eligible for root four exact guarded WP comparisons; grader must still check actual prediction bytes/rows. Native conditional audit only if all core gates pass.', 'scope': 'No model re-test/decoding or prediction/target array reads. Bound author model-integrity assertions to exact reviewed checker/source and actual checkpoint bytes. Prediction bytes were not rehashed here; root grader verifies them.', 'WP_queries': 0, 'targets_read': False}
path = HERE / 'ctxmem-actual-handoff-review.json'
assert not path.exists()
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
