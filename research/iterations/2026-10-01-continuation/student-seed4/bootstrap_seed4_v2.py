"""Fixed paired whole-chain bootstrap of existing six-sum statistics; no prediction/target reads."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METRIC_SHA = 'd05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'
RECIPES = ('double_shallow', 'current_student4', 'strong4', 'single_seed4', 'duplicate_seed3')
CONTRASTS = ('double_shallow-current_student4', 'double_shallow-strong4', 'single_seed4-current_student4', 'double_vs_duplicate_seed3')
REPLICATES = 2000
SEED = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_module():
    path = ROOT / 'scripts/fold_eval.py'
    assert sha(path) == METRIC_SHA
    spec = importlib.util.spec_from_file_location('student_bootstrap_frozen_metric', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_stats(arrays):
    assert set(arrays) == set(RECIPES) | {'positions', 'row_groups', 'fold_id', 'chains'}
    fold, chain = arrays['fold_id'], arrays['chains']
    n = len(fold)
    for key in ('positions', 'row_groups', 'fold_id', 'chains'):
        assert arrays[key].shape == (n,) and np.issubdtype(arrays[key].dtype, np.integer)
    assert set(fold.tolist()) == {0, 1}
    assert len(np.unique(arrays['positions'])) == n and len(np.unique(arrays['row_groups'])) == n
    for name in RECIPES:
        assert arrays[name].shape == (n, 2, 6) and arrays[name].dtype == np.float64
        assert np.isfinite(arrays[name]).all() and (arrays[name][..., 0] >= 0).all()
    for name in RECIPES[1:]:
        assert np.array_equal(arrays[name][..., [0, 1, 3]], arrays[RECIPES[0]][..., [0, 1, 3]])
    groups = {}
    for i, (f, ch) in enumerate(zip(fold, chain)):
        groups.setdefault((int(f), int(ch)), []).append(i)
    by_fold = {}
    for f in (0, 1):
        keys = [key for key in groups if key[0] == f]
        # Keep each whole chain, including its internal length/class composition.
        by_fold[f] = np.asarray([[arrays[name][groups[key]].sum(0) for name in RECIPES] for key in keys])
        assert len(keys) >= 2, 'Fewer than two chains cannot support a useful fold interval'
    return by_fold


def deltas(pooled, totals):
    scores = [pooled(s) for s in totals]
    return np.array([scores[0] - scores[1], scores[0] - scores[2], scores[3] - scores[1], scores[0] - scores[4]])


def bootstrap(arrays):
    fe = metric_module()
    grouped = cluster_stats(arrays)
    rng = np.random.default_rng(SEED)
    boot = np.empty((REPLICATES, 3, 4), dtype=np.float64)  # draw, overall/fold0/fold1, contrast
    for b in range(REPLICATES):
        totals = []
        for f in (0, 1):
            block = grouped[f]
            picks = rng.integers(0, len(block), len(block))
            totals.append(block[picks].sum(0))
        for scope, value in enumerate((totals[0] + totals[1], *totals)):
            boot[b, scope] = deltas(fe.pooled, value)
    result = {}
    for scope, selected in [('overall', np.ones(len(arrays['fold_id']), bool)),
                            ('fold0', arrays['fold_id'] == 0), ('fold1', arrays['fold_id'] == 1)]:
        i = ('overall', 'fold0', 'fold1').index(scope)
        point = deltas(fe.pooled, [arrays[name][selected].sum(0) for name in RECIPES])
        result[scope] = {name: {'delta': float(point[j]),
                               'percentile_CI95': np.quantile(boot[:, i, j], [.025, .975], method='linear').tolist()}
                         for j, name in enumerate(CONTRASTS)}
    return result, boot, {str(f): len(grouped[f]) for f in (0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--input-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert a.input.resolve().is_relative_to(HERE.resolve())
    assert a.output.resolve().is_relative_to(HERE.resolve())
    assert not a.output.exists() and not a.output.is_symlink(), 'Refuse existing output'
    assert sha(a.input) == a.input_sha256
    start = time.perf_counter()
    with np.load(a.input, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    estimates, _, counts = bootstrap(arrays)
    assert sha(a.input) == a.input_sha256, 'Input changed during bootstrap'
    result = {'input': str(a.input), 'input_sha256': a.input_sha256,
              'source_sha256': sha(__file__), 'metric_source_sha256': METRIC_SHA,
              'replicates': REPLICATES, 'seed': SEED, 'generator': 'NumPy default_rng PCG64',
              'NumPy_version': np.__version__, 'chain_counts_by_fold': counts,
              'resampling': 'Paired whole chains, with replacement, stratified by outer fold; same draws for all four contrasts and all five recipes.',
              'class_mixture': 'No class field in sufficient.npz; no class stratification. Each chain retains all its sequences/class composition. Empirical class shares may vary between resamples within a fold.',
              'statistic': 'Recompute frozen pooled mean-of-two-targets WP from six sums; never average sequence correlations. Zero-weight/degenerate-variance samples use frozen pooled() zero policy; no draws removed.',
              'intervals': estimates, 'seconds': time.perf_counter() - start,
              'uncertainty_scope': 'Chain sampling conditional on these fitted models and the observed pure-chain population.',
              'not_estimated': ['training-seed variation', 'adaptive model/recipe selection', 'hidden-test/domain shift'],
              'simultaneous_or_selection_adjusted': False, 'new_gates': False,
              'confirmation_established': False}
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(a.output), 'sha256': sha(a.output), 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
