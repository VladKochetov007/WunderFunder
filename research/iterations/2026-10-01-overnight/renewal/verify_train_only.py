#!/usr/bin/env python3
"""Authorized metadata-only proof; no feature, target, or mask array is opened."""
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_METADATA_SHA = '315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'
EXPECTED_MANIFEST_SHA = '13e39a60cdd88f5f0c35ea2c8abfe1558561c75cce4c8ddc8f1671446704a13d'
EXPECTED_PLAN_SHA = 'b3769bfc8d6f0d3de8063ff308f5378e078521dac01cc81c9e7bdd0db65ffdc9'
READ_KEYS = ('chain', 'split', 'row_group')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def stat(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev}


def main():
    started = time.monotonic()
    output = HERE / 'train-only-provenance.json'
    if output.exists():
        raise FileExistsError('Proof receipt is immutable; refusing overwrite')
    assert sha(HERE / 'selected-chains.json') == EXPECTED_MANIFEST_SHA
    assert sha(HERE / 'execution-plan.v1.json') == EXPECTED_PLAN_SHA
    manifest = json.loads((HERE / 'selected-chains.json').read_text())
    selected = manifest['selected']
    assert len(selected) == 256 and len({r['chain'] for r in selected}) == 256
    assert len({r['row_group'] for r in selected}) == 256
    repo = HERE.parents[3]
    for path_key, hash_key, selection_key in [
            ('source_manifest', 'source_manifest_sha256', 'selected'),
            ('original_manifest', 'original_manifest_sha256', 'selected_before_feature_reads')]:
        historical = repo / manifest[path_key]
        assert sha(historical) == manifest[hash_key]
        assert json.loads(historical.read_text())[selection_key] == selected

    source = Path(manifest['historical_chain_metadata']['path'])
    assert str(source) == '/home/grinchenkoeg/WunderFunder/cache/chain_folds.npz'
    assert manifest['historical_chain_metadata']['sha256'] == EXPECTED_METADATA_SHA
    before = stat(source)
    actual_sha = sha(source)
    assert actual_sha == EXPECTED_METADATA_SHA, 'Historical chain metadata hash mismatch'
    # np.load is lazy for npz: these are the only named arrays decompressed/read.
    # Whole archive bytes above are used solely for the required frozen SHA256.
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in READ_KEYS}
    chain, split, row_group = arrays['chain'], arrays['split'], arrays['row_group']
    assert chain.ndim == split.ndim == row_group.ndim == 1
    assert len(chain) == len(split) == len(row_group)
    split_text = split.astype(str)
    assert set(np.unique(split_text).tolist()) == {'train', 'valid'}
    assert np.issubdtype(chain.dtype, np.integer) and np.issubdtype(row_group.dtype, np.integer)
    proofs = []
    for ordinal, row in enumerate(selected):
        chain_members = chain == row['chain']
        train_group = (split_text == 'train') & (row_group == row['row_group'])
        assert chain_members.any(), 'Selected chain absent from frozen metadata'
        assert int(train_group.sum()) == 1, 'Selected train row group missing or duplicated'
        assert int(chain[train_group][0]) == row['chain'], 'Selected row group belongs to different chain'
        assert np.all(split_text[chain_members] == 'train'), 'Selected chain has a non-train member'
        assert not np.any(chain_members & (split_text == 'valid')), 'Selected chain shares validation rows'
        proofs.append(dict(row, manifest_ordinal=ordinal,
                           train_row_group_unique_match=True,
                           chain_train_row_groups=int((chain_members & (split_text == 'train')).sum()),
                           chain_valid_row_groups=int((chain_members & (split_text == 'valid')).sum()),
                           chain_only_train=True))
    after = stat(source)
    assert before == after and sha(source) == actual_sha, 'Metadata changed during verification'
    result = {
        'id': 'QRENEWAL-01-metadata-only-train-purity', 'status': 'PASS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'authorization': 'Controller explicitly authorized only chain/split/row_group metadata purity verification',
        'script_sha256': sha(Path(__file__)), 'plan_sha256': EXPECTED_PLAN_SHA,
        'selected_manifest_sha256': EXPECTED_MANIFEST_SHA,
        'source_manifest_sha256': manifest['source_manifest_sha256'],
        'original_manifest_sha256': manifest['original_manifest_sha256'],
        'metadata_source': {'path': str(source), 'sha256': actual_sha, 'stat_before': before, 'stat_after': after},
        'array_keys_opened': list(READ_KEYS),
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'sha256_uncompressed_c_bytes': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                   for key, value in arrays.items()},
        'metadata_rows': len(chain), 'metadata_train_rows': int((split_text == 'train').sum()),
        'metadata_valid_rows': int((split_text == 'valid').sum()),
        'exact_original_and_prior_ordered_selection_match': True,
        'selected_count': len(selected), 'distinct_selected_chains': len({r['chain'] for r in selected}),
        'all_train_row_group_chain_correspondences_unique': True,
        'selected_chains_with_validation_members': 0, 'all_selected_chains_train_only': True,
        'selected_proofs': proofs, 'feature_columns_read': [], 'target_columns_read': [],
        'mask_arrays_read': [], 'OOF_caches_read': [], 'parquet_files_opened': [],
        'wall_seconds': time.monotonic() - started,
    }
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': 'PASS', 'chains': len(selected), 'valid_sharing_chains': 0,
                      'receipt_sha256': sha(output), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
