"""Post-readout descriptive cross-tab, one existing NPZ read; no fits or corrections."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np

AGE_CUTS = np.array([1, 4, 16, 64, 256, 1024])
AGE_LABELS = ['0', '1–3', '4–15', '16–63', '64–255', '256–1023', '≥1024']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def record(W, S, global_W):
    # W and S: two target-wise sums of weight and weighted standardized error².
    mass = .5*np.sum(W/global_W)
    budget = .25*np.sum(S/global_W)
    return dict(target_weight=W.tolist(), target_weighted_squared_error=S.tolist(),
                target_conditional_error_intensity=np.divide(S, W, out=np.zeros(2), where=W>0).tolist(),
                metric_weight_share=float(mass), loss_budget=float(budget),
                loss_per_metric_mass=float(budget/mass) if mass>0 else None,
                metric_mass_weighted_error_intensity=float(2*budget/mass) if mass>0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--analysis-dir', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    signal.alarm(300)
    start = time.monotonic()
    assert not a.output.exists()
    ip = a.input_dir/'scored-rows.npz'
    rp = a.analysis_dir/'results.json'
    ir = json.loads((a.input_dir/'receipt.json').read_text())
    ar = json.loads((a.analysis_dir/'receipt.json').read_text())
    assert sha(ip) == ir['outputs'][ip.name]['sha256']
    assert sha(rp) == ar['outputs'][rp.name]['sha256']
    plan = json.loads(a.plan.read_text())
    assert plan['source_sha256'] == sha(__file__)
    assert plan['age_cuts'] == AGE_CUTS.tolist() and plan['fresh_age_max_exclusive'] == 16 and plan['quiet_age_min'] == 64
    provenance = dict(source_sha256=sha(__file__), plan_sha256=sha(a.plan),
                      input_sha256=sha(ip), existing_results_sha256=sha(rp),
                      post_readout=True, fits=0, inference_runs=0, candidate_WP=0,
                      threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    started_path = a.output.with_name(a.output.stem+'-started.json')
    assert not started_path.exists()
    started_path.write_text(json.dumps(provenance, indent=2)+'\n')
    old = json.loads(rp.read_text())
    par = old['budget']['models']['current_student4']['metric']
    Wg = np.array(par['W'])
    # Only one opening/read of the existing row cache; discard unused columns immediately.
    with np.load(ip, allow_pickle=False) as z:
        fi = z['feature_names'].tolist().index('log_since_i0px')
        age = np.rint(np.expm1(z['X'][:, fi].astype(np.float64))).astype(np.int32)
        p = z['predictions'][:, 0, :].astype(np.float64)
        y = z['y'].astype(np.float64)
        md = z['metadata']
    assert len(y) == 1614484 and np.isfinite(y).all() and np.isfinite(p).all()
    y, p = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = abs(y)
    u = (y-np.array(par['mu_y']))/np.array(par['sy'])
    z = (p-np.array(par['mu_p']))/np.array(par['sp'])
    loss = w*(u-z)**2
    np.testing.assert_allclose(w.sum(0), Wg, rtol=1e-10, atol=1e-8)
    classes, ci = np.unique(md[:, 5], return_inverse=True)
    folds = md[:, 4]
    agebin = np.searchsorted(AGE_CUTS, age, side='right')
    ng = len(classes)*2*7
    codes = (ci*2+folds)*7+agebin
    counts = np.bincount(codes, minlength=ng)
    weights = np.column_stack([np.bincount(codes, weights=w[:, t], minlength=ng) for t in (0, 1)]).reshape(len(classes),2,7,2)
    errors = np.column_stack([np.bincount(codes, weights=loss[:, t], minlength=ng) for t in (0, 1)]).reshape(len(classes),2,7,2)
    cross = []
    for c, cl in enumerate(classes):
        for f in (0, 1):
            for b in range(7):
                cross.append(dict(retrospective_class=int(cl), fold=f, age=AGE_LABELS[b],
                                  rows=int(counts[(c*2+f)*7+b]), **record(weights[c,f,b],errors[c,f,b],Wg)))
    assert abs(sum(r['loss_budget'] for r in cross)-(1-par['score'])) < 1e-10
    assert abs(sum(r['metric_weight_share'] for r in cross)-1) < 1e-10
    chain_keys, ch = np.unique(md[:,[4,3,5]], axis=0, return_inverse=True)
    assert len(chain_keys) == 250
    # Fresh/quiet comparison excludes age16..63 by the preregistered fixed contrast.
    population = np.where(age<16, 0, np.where(age>=64, 1, -1))
    good = population>=0
    code = ch[good]*2+population[good]
    cweights = np.column_stack([np.bincount(code, weights=w[good,t], minlength=500) for t in (0,1)]).reshape(250,2,2)
    cerrors = np.column_stack([np.bincount(code, weights=loss[good,t], minlength=500) for t in (0,1)]).reshape(250,2,2)
    ccounts = np.bincount(code, minlength=500).reshape(250,2)
    paired = np.all(cweights>0, axis=(1,2))
    intensity = np.divide(cerrors,cweights,out=np.full_like(cerrors,np.nan),where=cweights>0)
    difference = np.mean(intensity[:,1,:]-intensity[:,0,:],axis=1)
    summaries = []
    for cl in [None,*classes.tolist()]:
        for fold in [None,0,1]:
            scope = np.ones(250,bool)
            if cl is not None: scope &= chain_keys[:,2] == cl
            if fold is not None: scope &= chain_keys[:,0] == fold
            keep = paired & scope
            d = difference[keep]
            fresh = record(cweights[keep,0].sum(0),cerrors[keep,0].sum(0),Wg)
            quiet = record(cweights[keep,1].sum(0),cerrors[keep,1].sum(0),Wg)
            pooled_target_difference = np.asarray(quiet['target_conditional_error_intensity'])-np.asarray(fresh['target_conditional_error_intensity'])
            summaries.append(dict(retrospective_class='all' if cl is None else cl,
                fold='all' if fold is None else fold, eligible_scope_chains=int(scope.sum()),
                paired_chains=int(keep.sum()), excluded_without_both_positive_weight_populations=int((scope&~paired).sum()),
                paired_population_rows=ccounts[keep].sum(0).tolist(),
                chain_median_equal_target_intensity_difference=float(np.median(d)) if len(d) else None,
                chain_quartiles=np.quantile(d,[.25,.75]).tolist() if len(d) else None,
                chains_quiet_higher=int((d>1e-12).sum()), chains_quiet_lower=int((d< -1e-12).sum()),
                chains_tied=int((abs(d)<=1e-12).sum()),
                pooled_equal_target_intensity_difference=float(pooled_target_difference.mean()),
                pooled_target_intensity_difference=pooled_target_difference.tolist(),
                fresh=fresh, quiet=quiet))
    chain_records = [dict(fold=int(k[0]),chain=int(k[1]),retrospective_class=int(k[2]),
                          paired=bool(paired[i]),population_rows=ccounts[i].tolist(),
                          difference=float(difference[i]) if paired[i] else None,
                          fresh_target_intensity=intensity[i,0].tolist(),quiet_target_intensity=intensity[i,1].tolist())
                     for i,k in enumerate(chain_keys)]
    out = dict(status='COMPLETE_POST_READOUT_CONDITIONAL_DESCRIPTIVE_ONLY',provenance=provenance,
        age_labels=AGE_LABELS,age_cuts=AGE_CUTS.tolist(),cross_table=cross,paired_summaries=summaries,
        chain_records=chain_records,loss_identity=float(sum(r['loss_budget'] for r in cross)),
        seconds=time.monotonic()-start,peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation='Globally normalized fixed-model loss. Pairing controls which chains occur, not event timing, within-chain conditions, selection by target-weight or physical causality. Equal-chain medians/signs are descriptive, not independent causal replicates or a new score.')
    # All chains are paired in normal use; preserve JSON strictness if any are undefined.
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    a.output.write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'seconds':out['seconds'],'paired_summaries':summaries}))


if __name__ == '__main__':
    main()
