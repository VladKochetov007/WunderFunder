"""Independent audit of completed CTX runtime receipts; no model/data arrays opened."""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
P = HERE.parent / 'ctxmem'
D = P / 'runtime-prototype'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
close = lambda x, y: math.isclose(x, y, rel_tol=1e-14, abs_tol=1e-14)

plan = read(P / 'native-runtime-plan.json')
allocation = read(P / 'runtime-allocation.json')
summary = read(P / 'runtime-gate-result.json')
prep = read(D / 'preparation.json')
plan_sha = sha(P / 'native-runtime-plan.json')
assert plan_sha == '6a27499f415f978e453586784f6423c19cbd5f5b392e4de623eb723d013d5f71'
assert allocation['plan_sha256'] == summary['plan_sha256'] == prep['plan_sha256'] == plan_sha
assert allocation['critic_review_sha256'] == sha(HERE / 'ctxmem-native-preparation-review.json')
for name, expected in plan['files_sha256'].items():
    assert sha(P / name) == expected, name
assert sha(HERE.parent / 'amendment-01.json') == plan['amendment_sha256']
assert sha(HERE.parent / 'ctxmem-projection-correction.json') == plan['projection_correction_sha256']
assert prep['status'] == 'PASS' and prep['script_sha256'] == plan['files_sha256']['prepare_runtime.py']
for k in ['baseline_zip_sha256', 'raw_fixture_sha256', 'native_module_sha256']:
    assert prep[k] == plan[k]
assert summary == read(P / 'runtime-controller-attempt01.stdout')
assert not (P / 'runtime-controller-attempt01.stderr').read_text()
assert not (P / 'runtime-preparation.stderr').read_text()
assert read(P / 'runtime-preparation.stdout')['receipt_sha256'] == sha(D / 'preparation.json')
assert summary['seconds'] < allocation['limits']['outer_wall_seconds'] == 900
assert summary['forecast_fits'] == summary['WP_queries'] == 0
assert not summary['targets_read'] and not summary['masks_read']

manifests = prep['package_files_sha256']
for condition in ['memory', 'zero']:
    for name in ['solution.py', 'weights_2.npz', 'avx2/wnn_cpp.cpython-311-x86_64-linux-gnu.so', 'v2/wnn_cpp.cpython-311-x86_64-linux-gnu.so']:
        assert manifests[condition][name] == manifests['base'][name]
    assert manifests[condition]['v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so'] == plan['native_module_sha256']
    assert manifests[condition]['ctx_0.json'] == manifests[condition]['ctx_1.json']
assert manifests['memory']['members.json'] == manifests['zero']['members.json']
assert len(prep['identities']) == 4
assert {(r['condition'], r['member']) for r in prep['identities']} == {(c, i) for c in ['memory', 'zero'] for i in [0, 1]}
for row in prep['identities']:
    assert row['all_original_arrays_bitexact'] and row['added_keys'] == ['ctx_projection', 'head_extra.weight']

expected_order = [('base', 0), ('memory', 0), ('memory', 1), ('base', 1), ('base', 2), ('memory', 2), ('zero', 0)]
assert [(a['condition'], a['trial']) for a in summary['attempts']] == expected_order
assert sorted(p.name for p in D.glob('*-r*.json')) == sorted(f'{c}-r{i}.json' for c, i in expected_order)
timings = []
rows = {}
log_hashes = {}
for a in summary['attempts']:
    condition, trial = a['condition'], a['trial']
    stem = f'{condition}-r{trial}'
    r = read(D / (stem + '.json'))
    rows[condition, trial] = r
    assert sha(D / (stem + '.json')) == a['receipt_sha256']
    assert r['status'] == a['status'] == 'PASS'
    assert r['condition'] == condition and r['trial'] == trial
    assert r['rows'] == 100000 and r['member_count'] == 3 and r['weights'] == 'fixed equal1/3'
    assert [s['sequence'] for s in r['sequences']] == list(range(5))
    wall = sum(s['wall_seconds'] for s in r['sequences'])
    cpu = sum(s['cpu_seconds'] for s in r['sequences'])
    assert close(wall, r['wall_seconds']) and close(cpu, r['cpu_seconds'])
    assert close(cpu / wall, r['cpu_to_wall_ratio'])
    assert close(wall * 10, r['us_per_row'])
    assert close(wall / 100000 * 39400000 / 60, r['projected_minutes_39_4M'])
    for core in ['cpu18', 'cpu58']:
        ticks = sum(s['activity'][core]['ticks'] for s in r['sequences'])
        busy = sum(s['activity'][core]['busy_ticks'] for s in r['sequences'])
        assert ticks == r['activity'][core]['ticks'] and busy == r['activity'][core]['busy_ticks']
        assert close(busy / ticks, r['activity'][core]['busy_fraction'])
        for observation in [r['precheck'][core], r['activity'][core], r['wholetrace_activity'][core]] + [s['activity'][core] for s in r['sequences']]:
            assert 0 <= observation['busy_ticks'] <= observation['ticks']
            assert close(observation['busy_ticks'] / observation['ticks'], observation['busy_fraction'])
    assert r['precheck']['cpu58']['busy_fraction'] < .05
    assert r['activity']['cpu58']['busy_fraction'] < .05 and r['sibling58_below_5pct']
    assert r['affinity'] == [18] and r['threads'] == 1
    assert r['max_rss_KiB'] < 4 * 2**20 and r['wall_seconds'] < 120
    assert r['plan_sha256'] == plan_sha and r['raw_fixture_sha256'] == plan['raw_fixture_sha256']
    assert r['harness_sha256'] == plan['files_sha256']['benchmark_runtime.py']
    assert r['native_sha256'] == manifests[condition]['v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so']
    assert r['solution_sha256'] == manifests[condition]['solution.py']
    assert r['torch_runtime_absent'] and r['python'].startswith('3.11.16 ')
    assert not r['targets_read'] and not r['masks_read'] and r['WP_queries'] == 0
    assert r['qualification_only'] == (condition == 'zero')
    assert not (D / (stem + '.stderr')).read_text()
    log = read(D / (stem + '.stdout'))
    assert all(log[k] == r[k] for k in log)
    for suffix in ['stdout', 'stderr']:
        log_hashes[stem + '.' + suffix] = sha(D / (stem + '.' + suffix))
    timings.append({k: r[k] for k in ['condition', 'trial', 'wall_seconds', 'cpu_seconds', 'us_per_row', 'projected_minutes_39_4M', 'max_rss_KiB']})
    timings[-1]['SMT_busy_fraction'] = r['activity']['cpu58']['busy_fraction']
    timings[-1]['receipt_sha256'] = a['receipt_sha256']

ratios = []
assert len(summary['pairs']) == 3
for i, pair in enumerate(summary['pairs']):
    base, memory = rows['base', i], rows['memory', i]
    ratio = memory['wall_seconds'] / base['wall_seconds']
    ratios.append(ratio)
    assert pair['trial'] == i and pair['clean']
    assert pair['order'] == (['base', 'memory'] if i % 2 == 0 else ['memory', 'base'])
    assert pair['base_seconds'] == base['wall_seconds'] and pair['memory_seconds'] == memory['wall_seconds']
    assert ratio == pair['ratio']
assert statistics.median(ratios) == summary['primary_median_paired_ratio'] <= 1.03
assert summary['status'] == 'PASS'
assert close(summary['primary_overhead_fraction'], statistics.median(ratios) - 1)
for condition in ['base', 'memory']:
    assert len({rows[condition, i]['prediction_digest_for_parity_only'] for i in range(3)}) == 1
assert rows['zero', 0]['prediction_digest_for_parity_only'] == rows['base', 0]['prediction_digest_for_parity_only']
assert rows['zero', 0]['zero_extra_archived_baseline_bitexact'] and summary['zero_extra_archived_baseline_bitexact']
assert rows['memory', 0]['prediction_digest_for_parity_only'] != rows['base', 0]['prediction_digest_for_parity_only']
medbase = statistics.median(rows['base', i]['wall_seconds'] for i in range(3))
medmemory = statistics.median(rows['memory', i]['wall_seconds'] for i in range(3))
assert close(medmemory / medbase, summary['diagnostic_ratio_of_condition_medians'])
for condition, median in [('base', medbase), ('memory', medmemory)]:
    assert close(median / 100000 * 39400000 / 60, summary[f'diagnostic_{condition}_projected_minutes_39_4M'])

result = {
    'status': 'CTX_NATIVE_COST_AND_ZERO_PARITY_PASS',
    'checker_sha256': sha(__file__),
    'plan_sha256': plan_sha,
    'allocation_sha256': sha(P / 'runtime-allocation.json'),
    'summary_sha256': sha(P / 'runtime-gate-result.json'),
    'preparation_receipt_sha256': sha(D / 'preparation.json'),
    'controller_sha256': sha(P / 'run_runtime_gate.py'),
    'timings': timings,
    'paired_ratios': ratios,
    'median_paired_ratio': statistics.median(ratios),
    'all_three_pairs_clean_no_retries': True,
    'zero_extra_baseline_prediction_digest_exact': True,
    'original_array_identity_scope': 'Verified preparation source assertions and hash-bound completed receipt; critic did not reopen weights.',
    'package_integrity_scope': 'Hash-bound complete manifests checked before each trace by frozen harness; actual imported v3 binary and wrapper hashes match plan.',
    'command_scope': 'Per-trace argv reconstructed exactly from frozen controller source plus ordered receipt chain; no separate per-process exec ledger is present.',
    'log_hashes': log_hashes,
    'cost_gate_decision': 'Eligible for separate exact eight-fit training source/plan review and root allocation. This receipt authorizes no fit.',
    'limits': [
        'Three paired 100k-row traces measure prototype overhead on fixed five 20k sequences and synthetic nonzero extra matrices; no forecast quality claim.',
        'Final trained package still needs complete callback correctness and absolute <=50-minute runtime gate.',
        'Only measured v3 floating implementation qualified. Prototype leaves original avx2/v2 entries unchanged, so it is not a portable final CTX package.',
        'Reported startup excludes interpreter and NumPy import; CPU model/frequency are inherited from earlier fixed-host audit, not sampled in these receipts.'
    ],
    'data_arrays_opened_by_critic': False,
    'targets_read': False,
    'forecast_fits': 0,
    'WP_queries': 0
}
out = HERE / 'ctxmem-runtime-independent-review.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
