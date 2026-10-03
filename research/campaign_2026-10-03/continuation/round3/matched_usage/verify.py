from __future__ import annotations

import json
import resource
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from probe import R2, REPO, ROOT, choose_donor, parity, sha, train, windows

from wnn.gmetric import wp_from_sums


def main() -> None:
    started = time.monotonic()
    frozen = json.loads((ROOT / 'frozen_sources.json').read_text())
    for path, expected in frozen.items():
        assert sha(ROOT / path) == expected, path
    r2 = json.loads((R2 / 'artifact_provenance.json').read_text())
    for path, meta in r2['artifacts'].items():
        assert sha(R2 / path) == meta['sha256'], path
    panel = json.loads((ROOT / 'samples.json').read_text())
    control = panel['control']
    for path, expected in control['source_sha256'].items():
        assert sha(REPO / path) == expected, path
    sidecar = json.loads((REPO / 'cache/pred_c20261001_d256_s3_f0_valid.json').read_text())
    assert sidecar['checkpoint_sha256'] == control['checkpoint_sha256']
    assert sidecar['split_manifest_sha256'] == control['split_sha256']
    assert sha(REPO / 'cache/pred_c20261001_d256_s3_f0_valid.npy') == sidecar['prediction_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(REPO / control['checkpoint'], map_location='cpu', weights_only=False)
    model = train.make_model(checkpoint['args'], checkpoint['n_in'], control['n_out'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    parquet = pq.ParquetFile(REPO / 'datasets/valid.parquet')
    replay_max = 0
    donor_count = 0
    with torch.inference_mode():
        for i, sample in enumerate(panel['samples']):
            meta = parquet.read_row_group(sample['row_group'], columns=['need_prediction'], use_threads=False)
            assert np.array_equal(meta['need_prediction'].to_numpy(), np.arange(20000) >= 99)
            donors = json.loads((ROOT / f'donors_{i:02d}.json').read_text())
            with np.load(ROOT / f'seq_{i:02d}.npz') as z:
                raw, packed, std = z['raw'], z['packed'], z['std']
                assert sha_bytes(raw) == donors['raw_sha256']
                for d in donors['donors']:
                    assert choose_donor(raw, d['query'], std) == d
                    donor_count += 1
                # Independent actual replay of every arm at first frozen query of every sequence.
                d = donors['donors'][0]
                _, recomputed, direct = windows(raw, packed, d)
                boundary = np.flatnonzero(z['state_boundaries'] == d['start'])[0]
                h = torch.from_numpy(z['original_states'][boundary])
                for arm, w in enumerate([packed[d['start']:d['query']+1], recomputed, direct]):
                    actual, _ = model(torch.from_numpy(w)[None], h.clone())
                    gate = parity(actual[0, :, :2].numpy(), z['replay'][0, arm])
                    assert gate['passed'], (i, arm, gate)
                    replay_max = max(replay_max, gate['max_abs'])
    with np.load(ROOT / 'panel.npz') as z:
        moments = z['moments']
        independent_wp = wp_from_sums(moments.sum(0)).mean(-1)
        results = json.loads((ROOT / 'results.json').read_text())
        assert np.allclose(independent_wp, [a['scoped_sampled_wp'] for a in results['arms']], atol=1e-12, rtol=0)
        assert np.isfinite(moments).all()
        assert z['query_design_weight'].sum() == results['n_eligible']
    output = dict(passed=True, frozen_sources_verified=len(frozen), round2_artifacts_preserved=len(r2['artifacts']), actual_replayed_windows=72, actual_replay_max_abs=replay_max, donor_choices_recomputed=donor_count, target_metric_independent_reduction=True, all_99warmup_need_flags_verified=True, prediction_cache_hash_verified=True, seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (ROOT / 'verification.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def sha_bytes(array: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == '__main__':
    main()
