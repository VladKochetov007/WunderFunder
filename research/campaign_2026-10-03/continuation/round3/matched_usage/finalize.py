from __future__ import annotations

import json
import re

from probe import R2, REPO, ROOT, dump, sha


def elapsed(text: str) -> float:
    value = re.search(r'Elapsed \(wall clock\) time .*?: (\S+)', text).group(1)
    parts = [float(v) for v in value.split(':')]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def main() -> None:
    results = json.loads((ROOT / 'results.json').read_text())
    verification = json.loads((ROOT / 'verification.json').read_text())
    assert verification['passed'] and results['gates']['all_cpu_identity_future_pass']
    panel = json.loads((ROOT / 'samples.json').read_text())
    timed = []
    for i in range(24):
        path = ROOT / f'seq_{i:02d}.log'
        text = path.read_text()
        assert 'Exit status: 0' in text
        timed.append(dict(log=path.name,elapsed_seconds=elapsed(text),peak_rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',text).group(1))))
    verify_log = (ROOT / 'verification.log').read_text()
    cost = dict(per_sequence=timed,sequence_process_wall_seconds=sum(v['elapsed_seconds'] for v in timed),max_sequence_process_wall_seconds=max(v['elapsed_seconds'] for v in timed),max_sequence_process_rss_mib=max(v['peak_rss_kib'] for v in timed)/1024,instrument_body=results['cost'],verification_process_wall_seconds=elapsed(verify_log),verification_body=verification,scope='wrapper admission not included; per-process /usr/bin/time includes imports; no resource failure',commands='one-core2GiB900s run_cpu.sh round3-matched-usage',array_artifact_bytes=sum(p.stat().st_size for p in ROOT.glob('*.npz')))
    dump(ROOT / 'COST.json',cost)
    fields=['row_group','seq_ix','component','n_eligible','m_selected','query_pi','own_target_census_exposed','own_prior_feature_exposed','own_cpu_exposed','own_round2_usage_exposed']
    lines=['\t'.join(fields+['queries'])]
    for s in panel['samples']:
        lines.append('\t'.join([str(s[f]) for f in fields]+[','.join(map(str,s['queries']))]))
    (ROOT / 'IDS.tsv').write_text('\n'.join(lines)+'\n')
    state = json.loads((ROOT / 'STATE.json').read_text())
    assert state['completed_sequences']==list(range(24)) and state['failure'] is None
    state.update(phase='complete_report_frozen',verdict='Raw and packed influence survive; no clean32row engineered washout; no identified predictive gain',cpu_identity_future_pass=True,gpu_cache_bound_failures=7,panel_is_exposed=True,visually_inspected=['matched_usage.png'],resource_failure=False,training_proposal=None,next_test='Clock-reset washout with complete protected suffix input equality',blockers=['economic support of donor/splice','all current AND last32 engineered input equality'])
    dump(ROOT / 'STATE.json',state)
    external = [R2 / 'REPORT.md',R2 / 'usage_core.py',R2 / 'artifact_provenance.json',REPO / 'research/campaign_2026-10-03/continuation/round3/CONTRACT.md']
    critic = REPO / 'research/campaign_2026-10-03/continuation/round2/spec_critic'
    external.extend(critic / f for f in ['PREMORTEM.md','results.json','stream_equivalence_REPORT.md'])
    source = panel['control']['source_sha256']
    for p, expected in source.items():
        assert sha(REPO / p)==expected
    artifacts = {}
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.name not in ['artifact_provenance.json','finalize.log','verification_final.log']:
            artifacts[p.name] = dict(sha256=sha(p),bytes=p.stat().st_size)
    provenance = dict(artifacts=artifacts,checkpoint=panel['control']['checkpoint'],checkpoint_sha256=panel['control']['checkpoint_sha256'],split_manifest=panel['control']['split_manifest'],split_sha256=panel['control']['split_sha256'],prediction_cache_sha256=panel['control']['cached_predictions']['prediction_sha256'],live_source_sha256=source,read_frozen_evidence={str(p.relative_to(REPO)):sha(p) for p in external},prior_exposure='all24 components campaign-exposed; own target census covers all sequences; per-sequence exposure flags in samples/IDS, no fresh confirmation',warmup_rows=99,original_prefix_state='all earlier rows processed once; boundary prefix raw/packed identical',cpu_backend='float32 onecore, reference and replay; cached GPU diagnostic only',visually_inspected_figures=['matched_usage.png'],excluded_hashes='artifact_provenance self plus finalization/current final-audit output logs; cache directories omitted',source_preservation='frozen_sources.json original scored script hashes are unchanged;42 R2 hashes verified',resource_cost=cost)
    dump(ROOT / 'artifact_provenance.json',provenance)
    print(json.dumps(dict(phase=state['phase'],artifacts=len(artifacts),process_wall_seconds=cost['sequence_process_wall_seconds'],max_process_seconds=cost['max_sequence_process_wall_seconds'],peak_process_rss_mib=cost['max_sequence_process_rss_mib'],array_bytes=cost['array_artifact_bytes'])))


if __name__=='__main__':
    main()
