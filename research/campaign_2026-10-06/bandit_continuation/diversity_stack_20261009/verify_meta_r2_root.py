"""Independently recompute frozen meta-fit coefficients and saved OOF scores."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
sys.path.insert(0, str(root / 'src'))
from wnn.gmetric import GlobalAccumulator

base = root / 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009'
out = root / 'artifacts/analysis/luna_stacking_meta_r2/r2-2291238a-r4-attempt1'
report = json.loads((out / 'LUNA_META_CROSSFIT_R2_RESULT.json').read_text())
config = json.loads((base / 'LUNA_STACKING_META_CROSSFIT_R2.json').read_text())
assignment = json.loads((base / 'LUNA_STACKING_COMPONENT_FOLDS_R2.json').read_text())
group_fold = {g: c['outer_fold'] for c in assignment['assignment']['components'] for g in c['evaluation_groups']}
members = {
    e['column']: [(m['weight'], np.load(root / m['prediction_path'], mmap_mode='r')) for m in e['members']]
    for e in config['base_prediction_sources']
}
oof = np.load(out / 'meta_oof_ridge.npy', mmap_mode='r')
parquet = pq.ParquetFile(root / 'datasets/valid.parquet')
names = list(members) + ['ridge', 'equal_four_way']
metrics = {name: GlobalAccumulator() for name in names}
# Raw sufficient statistics provide a separate implementation from actor's centered merge.
stats = np.zeros((3, 2, 21), dtype=np.float64)
max_prediction_error = 0.0
selected_rows = 0
for i, group in enumerate(report['evaluation_row_groups_in_prediction_order']):
    table = parquet.read_row_group(group, columns=['t0', 't1', 'is_scored'])
    y = np.column_stack([table[c].to_numpy() for c in ['t0', 't1']])
    mask = table['is_scored'].to_numpy().astype(bool) & (np.arange(len(y)) >= 99)
    selected_rows += int(mask.sum())
    p = {name: sum(w * np.asarray(arr[group], dtype=np.float64) for w, arr in entries).astype(np.float32) for name, entries in members.items()}
    d = np.stack([p[name] - p['P4'] for name in ['P3', 'H3', 'Z4']], axis=-1).astype(np.float64)
    fold = group_fold[group]
    coefficients = report['outer_coefficients'][fold]
    rebuilt = np.column_stack([
        p['P4'][:, t].astype(np.float64) + (d[:, t] - coefficients[f'feature_mean_t{t}']) @ np.array(coefficients[f'theta_t{t}'])
        for t in range(2)
    ]).astype(np.float32)
    max_prediction_error = max(max_prediction_error, float(np.max(np.abs(rebuilt - oof[i]))))
    p['ridge'] = oof[i]
    p['equal_four_way'] = (sum(p[n].astype(np.float64) for n in members) / 4).astype(np.float32)
    for name in names:
        metrics[name].add(y, p[name], mask)
    for t in range(2):
        label = np.clip(y[mask, t], -2, 2).astype(np.float64)
        w = np.abs(label)
        x = d[mask, t]
        residual = label - p['P4'][mask, t]
        stats[fold, t] += np.concatenate([
            [w.sum()], (w[:, None] * x).sum(0), [np.dot(w, residual)],
            (x.T @ (w[:, None] * x)).ravel(), x.T @ (w * residual),
            np.zeros(4),
        ])
max_coefficient_error = 0.0
for fold in range(3):
    train = stats.sum(0) - stats[fold]
    for t in range(2):
        s = train[t]
        mean = s[1:4] / s[0]
        covariance = s[5:14].reshape(3, 3) / s[0] - np.outer(mean, mean)
        cross = s[14:17] / s[0] - mean * s[4] / s[0]
        lam = 0.5 * np.trace(covariance) / 3 + 1e-12
        theta = np.linalg.solve(covariance + lam * np.eye(3), cross)
        old = report['outer_coefficients'][fold]
        max_coefficient_error = max(max_coefficient_error, float(np.max(np.abs(theta - old[f'theta_t{t}']))))
assert max_prediction_error == 0.0, max_prediction_error
assert max_coefficient_error < 1e-10, max_coefficient_error
observed = {name: acc.result() for name, acc in metrics.items()}
score_errors = {name: abs(observed[name]['wp'] - report['metrics'][name]['wp']) for name in names}
assert max(score_errors.values()) < 1e-12, score_errors
assert selected_rows == 808863
receipt = {
    'status': 'pass', 'reviewer': 'root', 'method': 'independent raw sufficient statistics excluding each outer fold, full saved OOF reconstruction, official score recomputation',
    'result_sha256': hashlib.sha256((out / 'LUNA_META_CROSSFIT_R2_RESULT.json').read_bytes()).hexdigest(),
    'max_coefficient_error': max_coefficient_error, 'max_prediction_error': max_prediction_error,
    'score_errors': score_errors, 'metrics': observed, 'selected_rows': selected_rows,
    'limits': 'Same exposed fold and fixed predictions; computational verification, not independent training or public transfer confirmation.',
}
(base / 'LUNA_STACKING_ROOT_INDEPENDENT_CHECK_R1.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
