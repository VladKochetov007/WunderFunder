"""Independent raw TRAIN replay of the frozen joint-origin rank experiment."""
from __future__ import annotations
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
AUTHOR = ROOT / 'research/campaign_2026-09-30/data_reconstruction/joint_origin_clock'
OUT = Path(__file__).with_name('joint_origin_clock_independent_v1.json')
PAIRS = ((5, 6), (0, 5), (0, 6), (1, 5), (1, 6), (4, 5), (4, 6))

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='stable')
    y = x[order]
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    result = np.empty(len(x), dtype=np.float64)
    result[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return result

def describe(a: np.ndarray, eligible: np.ndarray) -> tuple[dict, list[bool]]:
    data = a[eligible]
    correlation, identities = [], []
    for first, second in PAIRS:
        u, v = ranks(data[:, first]), ranks(data[:, second])
        identities.append(bool(np.array_equal(u, v)))
        u -= u.mean() if len(u) else 0
        v -= v.mean() if len(v) else 0
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        correlation.append(float(np.dot(u, v) / denominator) if denominator else 0.0)
    return {'rows': len(data), 'distinct_values': [int(np.unique(data[:, i]).size) for i in range(8)],
            'pair_spearman': correlation}, identities

def main() -> None:
    start = time.monotonic()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    selection = json.loads((AUTHOR / 'selection.json').read_text())
    previous_path = AUTHOR.parent / 'event_rate_features/selection.json'
    manifest_path = ROOT / 'research/campaign_2026-09-29/artifacts/splits/fold0.json'
    for key, path in [('previous_selection_sha256', previous_path), ('manifest_sha256', manifest_path),
                      ('source_sha256', AUTHOR / 'probe.py'), ('preregistration_sha256', AUTHOR / 'preregistration.json')]:
        assert selection[key] == digest(path), key
    prior = json.loads(previous_path.read_text())
    exclude = set(prior['excluded_components']) | {r['component'] for key in ('fit', 'evaluation') for r in prior[key]}
    assert selection['excluded_components'] == sorted(exclude)
    members = json.loads(manifest_path.read_text())['sequences']['train']
    members.sort(key=lambda r: hashlib.sha256(f"joint-origin-clock-v1:{r['row_group']}".encode()).digest())
    chosen = []
    used = set(exclude)
    for row in members:
        if row['component'] not in used:
            chosen.append({key: row[key] for key in ('row_group', 'component')})
            used.add(row['component'])
        if len(chosen) == 16:
            break
    assert chosen == selection['selected']
    columns = [f'i0_p{i}' for i in range(22)] + [f'i0_{kind}{i}' for kind in ('dp', 'dv') for i in range(4)] + [f'a{i}' for i in range(8)]
    source = pq.ParquetFile(ROOT / 'datasets/train.parquet')
    records, rank_identical = [], []
    lower, upper = np.float32(-5.199337482452393), np.float32(5.199337482452393)
    for member in chosen:
        table = source.read_row_group(member['row_group'], columns=columns, use_threads=False)
        raw = np.stack([table[name].to_numpy() for name in columns], axis=1)
        assert raw.shape == (20000, 38) and raw.dtype == np.float32 and np.isfinite(raw).all()
        a = raw[:, 30:]
        step = np.arange(20000) >= 99
        bottom = np.logical_and(a[:, 5] == lower, a[:, 6] == lower)
        predecessor = np.zeros(20000, dtype=bool); predecessor[1:] = bottom[:-1]
        interior = np.logical_and.reduce((a[:, 5] > lower, a[:, 6] > lower, a[:, 5] < upper, a[:, 6] < upper))
        primary = step & predecessor & interior
        price = np.zeros(20000, dtype=bool); price[1:] = (np.diff(raw[:, :22], axis=0) != 0).any(1)
        trades = np.zeros(20000, dtype=bool); trades[1:] = (np.diff(raw[:, 22:30], axis=0) != 0).any(1)
        desc, identity = describe(a, primary)
        rank_identical.append(identity)
        records.append({**member, 'all_evaluation_steps': describe(a, step)[0], 'primary_joint_origin': desc,
                        'endpoint_support': {'previous_both_lower': int((step & predecessor).sum()),
                          'previous_lower_current_lower_or_saturated': int((step & predecessor & ~interior).sum()),
                          'both_lower_current': int((step & bottom).sum()),
                          'both_lower_with_price_and_trade_refresh': int((step & bottom & price & trades).sum()),
                          'both_lower_without_price_or_trade_refresh': int((step & bottom & ~price & ~trades).sum())}})
    expected = json.loads((AUTHOR / 'results.json').read_text())
    max_error = 0.0
    for actual, wanted in zip(records, expected['per_component'], strict=True):
        for key in ('row_group', 'component', 'endpoint_support'):
            assert actual[key] == wanted[key], key
        for key in ('all_evaluation_steps', 'primary_joint_origin'):
            assert actual[key]['rows'] == wanted[key]['rows']
            assert actual[key]['distinct_values'] == wanted[key]['distinct_values']
            error = np.max(np.abs(np.array(actual[key]['pair_spearman']) - wanted[key]['pair_spearman']))
            max_error = max(max_error, float(error)); assert error < 1e-12
    supported = [r for r in records if r['primary_joint_origin']['rows'] >= 100]
    summaries = []
    for index, pair in enumerate(PAIRS):
        pos = sum(r['primary_joint_origin']['pair_spearman'][index] >= .99 for r in supported)
        neg = sum(r['primary_joint_origin']['pair_spearman'][index] <= -.99 for r in supported)
        summaries.append({'pair': [f'a{i}' for i in pair], 'components_with_at_least100_primary_rows': len(supported),
                          'positive_rho_ge0.99_components': pos, 'negative_rho_le_minus0.99_components': neg,
                          'strong_common_order_support': max(pos, neg) >= 8})
    assert summaries == expected['summaries']
    output = {'status': 'pass', 'source_sha256': digest(Path(__file__)),
              'author_source_sha256': digest(AUTHOR/'probe.py'), 'selection_sha256': digest(AUTHOR/'selection.json'),
              'preregistration_sha256': digest(AUTHOR/'preregistration.json'), 'author_results_sha256': digest(AUTHOR/'results.json'),
              'selection_independently_reconstructed': True, 'per_component': records, 'summaries': summaries,
              'primary_exact_average_tie_rank_identity_by_component': rank_identical,
              'maximum_spearman_replay_discrepancy': max_error, 'targets_masks_VALID_external_models_read': False,
              'resources': {'wall_seconds': time.monotonic()-start, 'rlimit_as': list(resource.getrlimit(resource.RLIMIT_AS)),
                            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            'hard_cgroup_claim': False}}
    with OUT.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': output['status'], 'supported': len(supported), 'rank_exact_first_three_all16': bool(np.array(rank_identical)[:, :3].all()), 'max_error': max_error}))

if __name__ == '__main__':
    main()
