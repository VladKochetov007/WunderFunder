"""Post-readout audit requested by root: evaluate ALREADY FITTED ridge on fit Grams.

No refitting, lambda changes, predictor changes, real-row reads or candidate WP.
The original negative heldout result and its receipt remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--analysis-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    start = time.monotonic()
    er = json.loads((args.input_dir/'receipt.json').read_text())
    ar = json.loads((args.analysis_dir/'receipt.json').read_text())
    gp = args.input_dir/'chain-gram.npz'
    pp = args.analysis_dir/'probe-sufficient-results.npz'
    rp = args.analysis_dir/'results.json'
    assert sha(gp) == er['outputs'][gp.name]['sha256']
    for p in (pp, rp):
        assert sha(p) == ar['outputs'][p.name]['sha256']
    result = json.loads(rp.read_text())
    with np.load(pp, allow_pickle=False) as z:
        beta, alpha, metadata, half = [z[k] for k in ('coefficients', 'q_coefficients', 'metadata', 'halves')]
    with np.load(gp, allow_pickle=False) as z:
        raw, keys = z['gram'], z['chain_metadata']
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    grams = raw[np.asarray([lookup[tuple(k)] for k in metadata])]
    records = []
    fits = result['probe']['fits']
    assert len(fits) == len(beta) == len(alpha) == 32
    for j, rec in enumerate(fits):
        f, h, t = rec['fold'], rec['heldout_half'], rec['target']
        keep = (metadata[:, 0] == f) & (half != h)
        G = grams[keep, t].sum(0)
        const = np.zeros(G.shape[0])
        const[0] = rec['q']['q_fit_mean']
        d, base = alpha[j]-beta[j], alpha[j]-const
        SSE, baseline = float(d@G@d), float(base@G@base)
        assert abs(baseline/G[0, 0]-1) < 1e-7
        records.append(dict(fold=f, heldout_half=h, target=t, block=rec['block'],
                            fitting_chains=int(keep.sum()), W=float(G[0, 0]), SSE=SSE,
                            baseline_SSE=baseline, fitting_MSE=SSE/float(G[0, 0]),
                            fitting_skill=1-SSE/baseline))
    pooled = []
    for b, name in enumerate(result['probe']['blocks']):
        rows = [r for r in records if r['block'] == name]
        skill = [1-sum(r['SSE'] for r in rows if r['target'] == t)/sum(r['baseline_SSE'] for r in rows if r['target'] == t) for t in range(2)]
        pooled.append(dict(block=name, fit_skill_by_target=skill, fit_skill_mean=float(np.mean(skill)),
                           heldout_skill_by_target=result['probe']['summaries']['pooled']['skill_R2'][b],
                           heldout_skill_mean=result['probe']['summaries']['pooled']['equal_target_mean_skill'][b]))
    out = dict(status='COMPLETE_POSTREADOUT_FIXED_COEFFICIENT_AUDIT_NO_REFIT',
               source_sha256=sha(__file__), provenance={str(p):sha(p) for p in (gp, pp, rp)},
               seconds=time.monotonic()-start, pooled=pooled, fits=records,
               interpretation='Fitting skill describes the existing probes only. Post-readout supplementary audit; no new fit, metric candidate, feature block or lambda. History block mixes duplicated current features, nonlinear summaries and new history; incremental change is not a pure memory effect.')
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'status':out['status'], 'pooled':pooled, 'seconds':out['seconds']}))


if __name__ == '__main__':
    main()
