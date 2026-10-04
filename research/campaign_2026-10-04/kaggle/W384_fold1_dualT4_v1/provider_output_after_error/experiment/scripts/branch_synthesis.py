"""Pool all fixed E421 categorical readout controls over reciprocal components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_screen import digest_file, load_cache
from feature_v2_screen import compare


ARMS = {'unchanged', 'global_affine', 'wmse', 'joint_ce', 'joint_tilted_wmse',
        'candidate', 'factorized_tilted_wmse'}


def synthesize(root: Path, output: Path) -> dict:
    """Check frozen row identities and retain every prespecified control."""
    if output.exists():
        raise FileExistsError(output)
    caches = [load_cache(root / f'cache/c20261004_feature_f0_{part}.npz') for part in ('b', 'a')]
    if set(caches[0][0]['component']) & set(caches[1][0]['component']):
        raise ValueError('Reciprocal evaluation components overlap')
    pooled = {key: np.concatenate([data[key] for data, _ in caches])
              for key in ('y', 'component', 'sequence', 'row')}
    arms: dict[str, list[np.ndarray]] = {}
    hashes = {}
    for direction, (cache, metadata) in enumerate(caches):
        folder = root / f'research/campaign_2026-10-04/feature_branch_results/branch_f0_d{direction}_cpu_v1'
        report_file, prediction_file = folder/'report.json', folder/'predictions.npz'
        report = json.loads(report_file.read_text())
        training = report['training']
        if (report['eval_cache_sha256'] != metadata['sha256']
                or report['fit_cache_sha256'] != caches[1-direction][1]['sha256']
                or training['updates'] != 256 or training['batch_rows'] != 512
                or training['parameter_budget'] != 65536):
            raise ValueError('Changed prespecified training protocol')
        with np.load(prediction_file, allow_pickle=False) as values:
            if set(values.files) != ARMS | {'component', 'sequence', 'row'}:
                raise ValueError('Changed complete control inventory')
            for key in ('component', 'sequence', 'row'):
                if not np.array_equal(values[key], cache[key]):
                    raise ValueError('Changed evaluated row identity')
            if not np.array_equal(values['unchanged'], cache['pred']):
                raise ValueError('Changed untouched backbone')
            for name in sorted(ARMS):
                if values[name].shape != cache['y'].shape or not np.isfinite(values[name]).all():
                    raise ValueError('Invalid readout predictions')
                arms.setdefault(name, []).append(values[name])
        for path in (report_file, prediction_file):
            hashes[str(path.relative_to(root))] = digest_file(path)
    predictions = {key: np.concatenate(parts) for key, parts in arms.items()}
    result = {'experiment': 'E-421', 'rows': len(pooled['row']),
              'sequences': len(np.unique(pooled['sequence'])),
              'components': len(np.unique(pooled['component'])),
              'factorized_ce': compare(pooled, predictions, 2000),
              'factorized_hybrid': compare(pooled, {**predictions, 'candidate': predictions['factorized_tilted_wmse']}, 2000),
              'inputs': hashes,
              'scope': 'Previously exposed fold0; sampled rows, fixed frozen-trunk heads, unit fit-row weights',
              'selection': 'None; every fixed arm retained; end-to-end categorical architecture untested'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = synthesize(args.root.resolve(), args.output)
    print(json.dumps({'rows': result['rows'], 'components': result['components']}))
