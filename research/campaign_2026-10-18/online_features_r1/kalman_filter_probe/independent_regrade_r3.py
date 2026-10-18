from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
PACKET = ROOT / 'research/campaign_2026-10-18/online_features_r1/kalman_filter_probe'
CAMPAIGN = ROOT / 'research/campaign_2026-10-18/online_features_r1'
ATTEMPT = CAMPAIGN / 'runtime/attempts/kalman-gru-replay-20261018-r3.json'
LOG = CAMPAIGN / 'runtime/logs/kalman-gru-replay-20261018-r3.log'
REPLAY = Path('/home/vlad/development/wunder/research_tmp/kalman-online-20261018/kalman_gru_probe_r2')
SPLIT = ROOT / 'research/campaign_2026-09-29/artifacts/splits/fold0.json'
DATA = ROOT / 'datasets/valid.parquet'
NORMALIZATION = CAMPAIGN / 'kalman_parent_normalization.json'
CHECKPOINT = ROOT / 'artifacts/runs/warm_mechanism_f0_s3_c_r3/model_ep1.pt'
OUTPUT = Path('/home/vlad/development/wunder/research_tmp/kalman-online-20261018/independent_regrade_r3.json')
ARMS = ('raw', 'random_walk_posterior', 'local_trend_posterior')
EXPECTED_GROUPS = (4, 5, 7, 9, 12, 18, 19, 22, 26, 35, 39, 41, 42, 50, 51, 53)
EXPECTED_SEQ = (1990332, 2308749, 6746551, 9147005, 11423352, 24654613, 25927382, 28399078, 29519713, 38268719, 45444605, 47001707, 48100848, 51305808, 51608922, 52382536)
SEED = 20261018
N_BOOT = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metric(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict[str, float | int | str]:
    """Direct pooled implementation of global_wp_clip2_v1."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.shape != p.shape or y.ndim != 3 or y.shape[-1] != 2 or m.shape != y.shape[:-1]:
        raise ValueError('invalid target, prediction, or mask shape')
    if not m.any() or not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
        raise ValueError('empty scored panel or nonfinite selected values')
    result = []
    for k in range(2):
        yk = np.clip(y[..., k][m], -2.0, 2.0).astype(np.float64)
        pk = np.clip(p[..., k][m], -2.0, 2.0).astype(np.float64)
        weight = np.abs(yk)
        total = weight.sum(dtype=np.float64)
        if total < 1e-8:
            result.append(0.0)
            continue
        my = np.sum(weight * yk, dtype=np.float64) / total
        mp = np.sum(weight * pk, dtype=np.float64) / total
        cov = np.sum(weight * (yk - my) * (pk - mp), dtype=np.float64) / total
        vy = np.sum(weight * (yk - my) ** 2, dtype=np.float64) / total
        vp = np.sum(weight * (pk - mp) ** 2, dtype=np.float64) / total
        sy, sp = np.sqrt(max(vy, 0.0)), np.sqrt(max(vp, 0.0))
        result.append(0.0 if sy <= 1e-8 or sp <= 1e-8 else float(np.clip(cov / (sy * sp), -1.0, 1.0)))
    return {'t0': result[0], 't1': result[1], 'wp': 0.5 * (result[0] + result[1]), 'selected_rows': int(m.sum()), 'metric_id': 'global_wp_clip2_v1'}


def component_moments(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Build independent weighted centered sufficient moments for one component."""
    result = np.zeros((2, 6), dtype=np.float64)
    for k in range(2):
        yy = np.clip(y[..., k][m], -2.0, 2.0).astype(np.float64)
        pp = np.clip(p[..., k][m], -2.0, 2.0).astype(np.float64)
        ww = np.abs(yy)
        keep = ww > 0
        yy, pp, ww = yy[keep], pp[keep], ww[keep]
        if not len(ww):
            continue
        total = ww.sum(dtype=np.float64)
        my = np.sum(ww * yy, dtype=np.float64) / total
        mp = np.sum(ww * pp, dtype=np.float64) / total
        result[k] = (total, my, mp,
                     np.sum(ww * (yy - my) ** 2, dtype=np.float64),
                     np.sum(ww * (pp - mp) ** 2, dtype=np.float64),
                     np.sum(ww * (yy - my) * (pp - mp), dtype=np.float64))
    return result


def score_component_counts(moments: np.ndarray, counts: np.ndarray) -> float:
    scores = []
    for k in range(2):
        weight = counts.astype(np.float64) * moments[:, k, 0]
        total = weight.sum(dtype=np.float64)
        if total < 1e-8:
            scores.append(0.0)
            continue
        my = np.sum(weight * moments[:, k, 1], dtype=np.float64) / total
        mp = np.sum(weight * moments[:, k, 2], dtype=np.float64) / total
        vy = np.sum(counts * (moments[:, k, 3] + moments[:, k, 0] * (moments[:, k, 1] - my) ** 2), dtype=np.float64) / total
        vp = np.sum(counts * (moments[:, k, 4] + moments[:, k, 0] * (moments[:, k, 2] - mp) ** 2), dtype=np.float64) / total
        cov = np.sum(counts * (moments[:, k, 5] + moments[:, k, 0] * (moments[:, k, 1] - my) * (moments[:, k, 2] - mp)), dtype=np.float64) / total
        sy, sp = np.sqrt(max(vy, 0.0)), np.sqrt(max(vp, 0.0))
        scores.append(0.0 if sy <= 1e-8 or sp <= 1e-8 else float(np.clip(cov / (sy * sp), -1.0, 1.0)))
    return 0.5 * (scores[0] + scores[1])


def bootstrap(y: np.ndarray, predictions: np.ndarray, raw: np.ndarray, mask: np.ndarray, components: np.ndarray) -> dict[str, object]:
    ids = np.unique(components)
    comp_a = np.stack([component_moments(y[components == c], predictions[components == c], mask[components == c]) for c in ids])
    comp_b = np.stack([component_moments(y[components == c], raw[components == c], mask[components == c]) for c in ids])
    counts = np.ones(len(ids), dtype=np.int64)
    actual_a = score_component_counts(comp_a, counts)
    actual_b = score_component_counts(comp_b, counts)
    rng = np.random.default_rng(SEED)
    deltas = np.empty(N_BOOT, dtype=np.float64)
    for j in range(N_BOOT):
        selected = rng.integers(0, len(ids), len(ids))
        counts = np.bincount(selected, minlength=len(ids))
        deltas[j] = score_component_counts(comp_a, counts) - score_component_counts(comp_b, counts)
    return {
        'candidate_wp': actual_a,
        'reference_wp': actual_b,
        'mean_diff': actual_a - actual_b,
        'bootstrap_sd': float(deltas.std(ddof=1)),
        'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(),
        'p_le_0': float(np.mean(deltas <= 0.0)),
        'n_sequences': int(len(components)),
        'n_components': int(len(ids)),
        'n_boot': N_BOOT,
        'seed': SEED,
        'unit': 'unique source component; paired candidate-reference resamples',
    }


def main() -> None:
    started = time.time()
    recipe_path = PACKET / 'recipe_r2.json'
    preflight_path = PACKET / 'preflight_receipt_r2.json'
    grade_path = PACKET / 'grade_r2.json'
    recipe = json.loads(recipe_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    original_grade = json.loads(grade_path.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    sidecars = {arm: json.loads((REPLAY / f'{arm}.json').read_text()) for arm in ARMS}
    split_bytes = SPLIT.read_bytes()
    split = json.loads(split_bytes)
    expected_split_sha = recipe['membership']['split_sha256']
    if hashlib.sha256(split_bytes).hexdigest() != expected_split_sha:
        raise ValueError('split manifest differs from recipe')
    actual_groups = tuple(map(int, split['evaluation']['valid'][:len(EXPECTED_GROUPS)]))
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError('frozen first-16 evaluation groups differ')
    seq_map = {int(row['row_group']): int(row['seq_ix']) for row in split['sequences']['valid']}
    actual_seq = tuple(seq_map[g] for g in actual_groups)
    if actual_seq != EXPECTED_SEQ:
        raise ValueError('source-qualified sequence identity differs')
    comp_map = {int(row['row_group']): int(row['component']) for row in split['sequences']['valid']}
    components = np.asarray([comp_map[g] for g in actual_groups], dtype=np.int64)

    source_meta = split['sources']['datasets/valid.parquet']
    data_stat = DATA.stat()
    if data_stat.st_size != int(source_meta['bytes']) or data_stat.st_mtime_ns != int(source_meta['mtime_ns']):
        raise ValueError('valid.parquet differs from pinned split metadata')
    if sha256(CHECKPOINT) != recipe['parent']['sha256']:
        raise ValueError('parent checkpoint hash differs from recipe')
    if sha256(NORMALIZATION) != recipe['filter_config']['normalization']['sha256']:
        raise ValueError('normalization prior hash differs from recipe')
    if sha256(SPLIT) != recipe['membership']['split_sha256']:
        raise ValueError('split hash differs from recipe')

    current_source_hashes = {name: sha256(ROOT / name) if not name.endswith('.py') or '/' in name else sha256(PACKET / name) for name in preflight['source_hashes']}
    source_drift = {name: {'pinned': digest, 'current': current_source_hashes.get(name)}
                    for name, digest in preflight['source_hashes'].items()
                    if current_source_hashes.get(name) != digest}

    if attempt.get('contract_sha256') != sha256(recipe_path):
        raise ValueError('durable attempt contract hash does not match recipe')
    if attempt.get('status') != 'finished' or attempt.get('exit_status') != 0 or attempt.get('failure') is not None:
        raise ValueError('durable replay attempt is not a successful terminal execution')
    if attempt.get('id') != 'kalman-gru-replay-20261018-r3':
        raise ValueError('wrong durable attempt identifier')

    predictions: dict[str, np.ndarray] = {}
    cache_hashes: dict[str, str] = {}
    for arm in ARMS:
        sidecar = sidecars[arm]
        cache_path = REPLAY / f'{arm}.npz'
        digest = sha256(cache_path)
        cache_hashes[arm] = digest
        if digest != sidecar.get('cache_sha256'):
            raise ValueError(f'{arm} cache SHA does not match its sidecar')
        if sidecar.get('checkpoint_sha256') != recipe['parent']['sha256']:
            raise ValueError(f'{arm} sidecar names a different parent')
        if sidecar.get('preflight_sha256') != sha256(preflight_path):
            raise ValueError(f'{arm} sidecar does not bind the frozen preflight')
        if sidecar.get('source_hashes') != preflight.get('source_hashes'):
            raise ValueError(f'{arm} sidecar source pins differ from preflight')
        if sidecar.get('group_ids') != list(EXPECTED_GROUPS) or sidecar.get('sequence_ids') != list(EXPECTED_SEQ):
            raise ValueError(f'{arm} sidecar membership differs')
        if sidecar.get('targets_or_masks_loaded') is not False or sidecar.get('official_metric_called') is not False:
            raise ValueError(f'{arm} replay sidecar reports target/mask access or scoring')
        with np.load(cache_path, allow_pickle=False) as cache:
            if set(cache.files) != {'group_ids', 'sequence_ids', 'predictions'}:
                raise ValueError(f'{arm} NPZ members differ from expected prediction-only contract')
            if not np.array_equal(cache['group_ids'], np.asarray(EXPECTED_GROUPS, dtype=np.int64)):
                raise ValueError(f'{arm} NPZ row-group identity differs')
            if not np.array_equal(cache['sequence_ids'], np.asarray(EXPECTED_SEQ, dtype=np.int64)):
                raise ValueError(f'{arm} NPZ sequence identity differs')
            preds = np.asarray(cache['predictions'], dtype=np.float32)
        if preds.shape != (16, 20_000, 2) or not np.isfinite(preds).all():
            raise ValueError(f'{arm} predictions have invalid shape or nonfinite values')
        predictions[arm] = preds

    parquet = pq.ParquetFile(DATA)
    targets = np.empty((16, 20_000, 2), dtype=np.float32)
    raw_mask = np.empty((16, 20_000), dtype=bool)
    row_index = np.arange(20_000)
    for i, (group, seq) in enumerate(zip(EXPECTED_GROUPS, EXPECTED_SEQ, strict=True)):
        table = parquet.read_row_group(group, columns=['seq_ix', 't0', 't1', 'is_scored'], use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError(f'row group {group} has {table.num_rows} rows, expected 20000')
        seq_column = table.column('seq_ix').to_numpy()
        if not np.all(seq_column == seq):
            raise ValueError(f'row group {group} sequence ID does not match frozen split')
        targets[i, :, 0] = table.column('t0').to_numpy().astype(np.float32, copy=False)
        targets[i, :, 1] = table.column('t1').to_numpy().astype(np.float32, copy=False)
        raw_mask[i] = table.column('is_scored').to_numpy().astype(bool, copy=False)
        del table
    warmup = 99  # gmetric.py WARMUP; its scorer expects the caller to mask these rows
    mask = raw_mask & (row_index[None, :] >= warmup)
    scores = {arm: metric(targets, predictions[arm], mask) for arm in ARMS}
    comparisons: dict[str, object] = {}
    for arm in ARMS[1:]:
        interval = bootstrap(targets, predictions[arm], predictions['raw'], mask, components)
        comparisons[arm] = {
            'delta_wp': scores[arm]['wp'] - scores['raw']['wp'],
            'independent_component_bootstrap': interval,
        }

    receipt = {
        'schema': 'independent-kalman-regrade-r1',
        'status': 'passed_alignment_and_independent_regrade',
        'created_unix': time.time(),
        'command': 'WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=20% WNN_MEM_MAX=1000M WNN_THREADS=1 scripts/limited .venv/bin/python /home/vlad/development/wunder/research_tmp/kalman-online-20261018/independent_regrade.py',
        'limits': {'cpu_quota': '20%', 'memory_max': '1000M', 'threads': 1, 'gpu': 'none'},
        'runtime_seconds': time.time() - started,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pyarrow': __import__('pyarrow').__version__,
        'torch_imported': 'torch' in sys.modules,
        'recipe_status_stale_at_review': recipe.get('status'),
        'durable_attempt': {
            'id': attempt['id'], 'status': attempt['status'], 'exit_status': attempt['exit_status'],
            'elapsed_seconds': attempt.get('elapsed_seconds'), 'unit': attempt.get('unit'),
            'contract_sha256_matches_recipe': attempt.get('contract_sha256') == sha256(recipe_path),
        },
        'membership': {
            'split_sha256': sha256(SPLIT), 'valid_parquet_bytes': data_stat.st_size,
            'valid_parquet_mtime_ns': data_stat.st_mtime_ns,
            'groups': list(EXPECTED_GROUPS), 'sequence_ids': list(EXPECTED_SEQ),
            'components_in_order': components.tolist(), 'n_unique_components': int(np.unique(components).size),
            'rows_per_group': 20_000, 'target_mask_rows_before_warmup': int(raw_mask.sum()),
            'is_scored_rows_in_first_99': int(raw_mask[:, :warmup].sum()),
            'selected_rows_after_warmup': int(mask.sum()), 'warmup_rows': warmup,
            'row_group_seq_ix_checked': True, 'warmup_source': 'src/wnn/gmetric.py: WARMUP=99; supplied to caller mask',
        },
        'cache_validation': {
            'sha256': cache_hashes, 'prediction_shape': [16, 20_000, 2], 'dtype': 'float32',
            'all_finite': True, 'npz_members': ['group_ids', 'sequence_ids', 'predictions'],
            'raw_identical_to_random_walk': bool(np.array_equal(predictions['raw'], predictions['random_walk_posterior'])),
            'raw_identical_to_local_trend': bool(np.array_equal(predictions['raw'], predictions['local_trend_posterior'])),
            'per_sidecar_cache_checkpoint_preflight_membership_and_no_targets_no_metric_checks': True,
        },
        'scores': scores,
        'comparisons': comparisons,
        'provenance_drift': {
            'current_source_hashes_match_preflight': not source_drift,
            'mismatches': source_drift,
            'checkpoint_sha256': sha256(CHECKPOINT),
            'normalization_sha256': sha256(NORMALIZATION),
        },
        'artifact_sha256': {
            'recipe_r2.json': sha256(recipe_path), 'preflight_receipt_r2.json': sha256(preflight_path),
            'grade_r2.json': sha256(grade_path), 'grade_replay_r2.py_reviewed_not_invoked': sha256(PACKET / 'grade_replay_r2.py'),
            'attempt_r3.json': sha256(ATTEMPT), 'attempt_log_r3.log': sha256(LOG),
            'split_manifest': sha256(SPLIT),
            'independent_scorer_script_sha256': sha256(Path(__file__).resolve()),
            'official_metric_src_wnn_gmetric_py': sha256(ROOT / 'src/wnn/gmetric.py'),
            'comparison_helpers_src_wnn_evaluation_py': sha256(ROOT / 'src/wnn/evaluation.py'),
        },
        'actor_grade_summary': {
            arm: original_grade['scores'][arm] for arm in ARMS
        },
        'actor_grade_sha256': sha256(grade_path),
        'limits_of_inference': [
            'The 16 groups are a preselected subset of exposed F0. These scores are fixed-endpoint descriptions, not clean validation or transfer evidence.',
            'The paired bootstrap resamples only 12 represented source components, conditional on the selected endpoints and panel; it does not quantify training-seed/model-fit variation or selection uncertainty.',
            'The actor comparison uses candidate and raw from the same aligned prediction caches and the same pooled official metric. Its centered per-sequence moments are correctly aggregated by source component before paired resampling.',
            'The local-trend percentile interval reported in grade_r2.json crosses zero despite p_le_0=0.0319; p_le_0 is a tail fraction, not a two-sided significance test.',
        ],
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    receipt['receipt_sha256'] = sha256(OUTPUT)
    print(json.dumps({
        'output': str(OUTPUT), 'scores': scores,
        'comparisons': {arm: {'delta_wp': value['delta_wp'], 'ci95': value['independent_component_bootstrap']['ci95'], 'p_le_0': value['independent_component_bootstrap']['p_le_0']} for arm, value in comparisons.items()},
        'selected_rows': int(mask.sum()), 'n_components': int(np.unique(components).size),
        'source_drift_count': len(source_drift), 'receipt_sha256': receipt['receipt_sha256'],
    }, indent=2))


if __name__ == '__main__':
    main()
