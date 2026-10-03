from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from wnn.splits import load_campaign_split

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
R2 = REPO / 'research/campaign_2026-10-03/continuation/round2/usage'

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main() -> None:
    started = time.monotonic()
    control = json.loads((R2 / 'control_identity.json').read_text())
    for path, expected in control['source_sha256'].items():
        assert digest(REPO / path) == expected, path
    assert digest(REPO / control['checkpoint']) == control['checkpoint_sha256']
    split = load_campaign_split(REPO / control['split_manifest'], REPO)
    trained = json.loads((REPO / 'artifacts/runs/c20261001_d256_s3_f0/provenance.json').read_text())
    split.assert_fit_groups(trained['fit_row_groups']['train'], trained['fit_row_groups']['valid'])
    with np.load(REPO / 'cache/valid_targets.npz') as z:
        groups, ids = z['row_group'], z['seq_ix']
    sidecar = json.loads((REPO / 'cache/pred_c20261001_d256_s3_f0_valid.json').read_text())
    assert groups.tolist() == sidecar['row_groups']
    lookup = {int(g):(i,int(ids[i])) for i,g in enumerate(groups)}
    comps = {g:c for source,g,c in split.components if source == 'valid'}
    ranked = sorted(split.evaluation_groups, key=lambda g: hashlib.sha256(f'round3-matched-usage-v1:{lookup[g][1]}'.encode()).digest())
    parquet = pq.ParquetFile(REPO / 'datasets/valid.parquet')
    own = json.loads((REPO / 'research/campaign_2026-10-03/eda/statphys/continuation_samples.json').read_text())
    old_feature = set(own['possibly_exposed_row_groups']['valid'])
    old_cpu = {s['row_group'] for s in own['samples']['valid']}
    old_r2 = {s['row_group'] for s in json.loads((R2/'samples.json').read_text())['samples']}
    samples, used = [], set()
    rng = np.random.default_rng(614203)
    for g in ranked:
        c = comps[g]
        if c in used:
            continue
        used.add(c)
        pos, seq = lookup[g]
        t = parquet.read_row_group(g, columns=['seq_ix','step_in_seq','need_prediction','is_scored'],use_threads=False)
        steps = t['step_in_seq'].to_numpy()
        assert np.array_equal(steps,np.arange(20000))
        assert np.all(t['seq_ix'].to_numpy() == seq)
        eligible = steps[(steps>=512)&t['need_prediction'].to_numpy()&t['is_scored'].to_numpy()]
        n,m = len(eligible), min(12,len(eligible))
        queries = np.sort(rng.choice(eligible,m,replace=False)).tolist()
        samples.append(dict(row_group=int(g),component=int(c),seq_ix=seq,cache_position=pos,n_eligible=n,m_selected=m,query_pi=m/n if n else 0,queries=queries,own_target_census_exposed=True,campaign_valid_exposed=True,own_prior_feature_exposed=g in old_feature,own_cpu_exposed=g in old_cpu,own_round2_usage_exposed=g in old_r2))
        if len(samples)==24:
            break
    result = dict(samples=samples,selection_read_only_metadata_and_score_masks=True,component_fit_exclusion_verified=True,component_panel_probability='fixed conditional panel; no population component weighting',prereg_sha256=digest(ROOT/'PREREG.json'),control=control,seconds=time.monotonic()-started)
    (ROOT/'samples.json').write_text(json.dumps(result,indent=2))
    (ROOT/'STATE.json').write_text(json.dumps(dict(phase='sample_frozen_before_targets_predictions',completed_sequences=[],sample_sha256=digest(ROOT/'samples.json'),failure=None),indent=2))
    print(json.dumps(dict(components=len(samples),empty=sum(s['m_selected']==0 for s in samples),queries=sum(s['m_selected'] for s in samples),eligible=sum(s['n_eligible'] for s in samples),seconds=result['seconds'])))

if __name__=='__main__':
    main()
