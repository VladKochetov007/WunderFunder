#!/usr/bin/env python3
"""Posthoc explanation of THREE existing reports; never load predictions/targets.

Reads fixed saved sufficient statistics, their reports, frozen evaluator source,
and only split/fold/pure/row_group metadata from chain_folds.npz. No transformations,
new blends, model scoring, bootstrap, gates, selection, or evaluation-ledger writes.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IDS = ('TRANSHEAD-01-primary', 'CTX-joint-memory-vs-current', 'Q3-quant-vs-baseline3')
METADATA_SHA = '315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'
TOL = 2e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata():
    """One small remote metadata-only read; no target or prediction paths."""
    code = '''import hashlib,json,numpy as np
from pathlib import Path
p=Path("cache/chain_folds.npz")
h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=="%s"
with np.load(p,allow_pickle=False) as z:
 a={k:z[k] for k in ("split","fold","pure","row_group")}
v=a["split"]=="valid"
assert len(np.unique(a["row_group"][v]))==int(v.sum())
print(json.dumps({"source":str(p),"sha256":h,"arrays_read":list(a),
 "valid_count":int(v.sum()),
 "pure_fold_counts":{str(f):int((v&a["pure"].astype(bool)&(a["fold"]==f)).sum()) for f in (0,1)},
 "array_sha256":{k:hashlib.sha256(x.tobytes()).hexdigest() for k,x in a.items()}}))
''' % METADATA_SHA
    command = ('cd /home/grinchenkoeg/WunderFunder-creative && '
               'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
               '.venv/bin/python -c ' + shlex.quote(code))
    result = subprocess.run(['ssh', 'beleriand', command], check=True,
                            capture_output=True, text=True, timeout=30)
    value = json.loads(result.stdout)
    assert value['valid_count'] == 1873
    assert value['pure_fold_counts'] == {'0': 425, '1': 435}
    value['order_rule'] = ('Frozen evaluator loops folds 0 then 1 and wrapper concatenates '
                           'per-sequence sums in that order; pure counts give boundary 425. '
                           'Both reported fold deltas independently verify the boundary.')
    return value


def moments(stats):
    """(...,6) -> weighted population moments; no Bessel correction."""
    w, sy, sp, syy, spp, syp = np.moveaxis(np.asarray(stats, np.float64), -1, 0)
    assert np.all(w > 1e-8)
    my, mp = sy / w, sp / w
    vy, vp, cov = syy / w - my * my, spp / w - mp * mp, syp / w - my * mp
    assert np.all(vy > 1e-16) and np.all(vp > 1e-16)
    corr = cov / np.sqrt(vy * vp)
    assert np.all(np.abs(corr) < 1)  # no clipping active in these saved cases
    return dict(weight=w, target_mean=my, prediction_mean=mp,
                target_variance=vy, prediction_variance=vp, covariance=cov,
                correlation=corr)


def decompose(fold_sums):
    f = moments(fold_sums)
    a = f['weight'] / f['weight'].sum(axis=0)
    my = (a * f['target_mean']).sum(axis=0)
    mp = (a * f['prediction_mean']).sum(axis=0)
    dy, dp = f['target_mean'] - my, f['prediction_mean'] - mp
    cw = (a * f['covariance']).sum(axis=0)
    cb = (a * dy * dp).sum(axis=0)
    yw = (a * f['target_variance']).sum(axis=0)
    yb = (a * dy * dy).sum(axis=0)
    pw = (a * f['prediction_variance']).sum(axis=0)
    pb = (a * dp * dp).sum(axis=0)
    denominator = np.sqrt((yw + yb) * (pw + pb))
    kappa = a * np.sqrt(f['target_variance'] * f['prediction_variance']) / denominator
    between_term = cb / denominator
    reconstructed = (kappa * f['correlation']).sum(axis=0) + between_term
    pooled = moments(fold_sums.sum(axis=0))
    errors = {'covariance': float(np.max(np.abs(cw + cb - pooled['covariance']))),
              'target_variance': float(np.max(np.abs(yw + yb - pooled['target_variance']))),
              'prediction_variance': float(np.max(np.abs(pw + pb - pooled['prediction_variance']))),
              'correlation': float(np.max(np.abs(reconstructed - pooled['correlation'])))}
    assert max(errors.values()) < TOL
    return dict(folds=f, fold_weight_fraction=a, pooled=pooled,
                covariance_within=cw, covariance_between=cb,
                target_variance_within=yw, target_variance_between=yb,
                prediction_variance_within=pw, prediction_variance_between=pb,
                denominator=denominator, fold_coefficient=kappa,
                normalized_between_covariance=between_term,
                decomposition_max_abs_errors=errors)


def symmetric_delta(candidate, reference):
    kc, kr = candidate['fold_coefficient'], reference['fold_coefficient']
    rc, rr = candidate['folds']['correlation'], reference['folds']['correlation']
    within = (0.5 * (kc + kr) * (rc - rr)).sum(axis=0)
    coefficient = (0.5 * (rc + rr) * (kc - kr)).sum(axis=0)
    between = (candidate['normalized_between_covariance'] -
               reference['normalized_between_covariance'])
    exact = candidate['pooled']['correlation'] - reference['pooled']['correlation']
    error = float(np.max(np.abs(within + coefficient + between - exact)))
    assert error < TOL
    terms = {'within_fold_correlation': within,
             'fold_coefficient_and_normalization': coefficient,
             'normalized_between_fold_covariance': between}
    return {'by_target': terms,
            'mean_wp': {k: float(v.mean()) for k, v in terms.items()},
            'sum_by_target': within + coefficient + between,
            'max_abs_reconstruction_error': error}


def generated_checks():
    """Check moments against literal centering, including a generated sign reversal."""
    rng = np.random.default_rng(2026100104)
    max_error = 0.0
    cases = 0
    for groups in (1, 2, 5):
        for _ in range(8):
            y = rng.normal(size=(groups, 23, 2))
            p = .7 * y + rng.normal(size=y.shape) + rng.normal(size=(groups, 1, 2))
            w = rng.uniform(.1, 2, size=y.shape)
            s = np.stack([w.sum(1), (w*y).sum(1), (w*p).sum(1),
                          (w*y*y).sum(1), (w*p*p).sum(1), (w*y*p).sum(1)], -1)
            d = decompose(s)
            wy, wp = (w*y).sum((0,1))/w.sum((0,1)), (w*p).sum((0,1))/w.sum((0,1))
            direct = [(w*(y-wy)*(p-wp)).sum((0,1))/w.sum((0,1)),
                      (w*(y-wy)**2).sum((0,1))/w.sum((0,1)),
                      (w*(p-wp)**2).sum((0,1))/w.sum((0,1))]
            rebuilt = [d['covariance_within']+d['covariance_between'],
                       d['target_variance_within']+d['target_variance_between'],
                       d['prediction_variance_within']+d['prediction_variance_between']]
            err = max(float(np.max(np.abs(a-b))) for a,b in zip(direct,rebuilt))
            assert err < TOL
            assert symmetric_delta(d,d)['max_abs_reconstruction_error'] == 0
            max_error = max(max_error, err)
            cases += 1
    local = np.array([-.1, 0., .1])
    means = np.array([-1., 1.])[:,None]
    y = means + local
    w = np.abs(y)
    reference = -means + local
    candidate = means + .8*local + np.array([.02,-.04,.02])
    def ss(p):
        return np.stack([w.sum(1),(w*y).sum(1),(w*p).sum(1),(w*y*y).sum(1),
                         (w*p*p).sum(1),(w*y*p).sum(1)],-1)[:,None,:].repeat(2,axis=1)
    # This perfect reference has r=1, while moments() guards real-case r strictly <1.
    reference = reference + np.array([.0001,-.0002,.0001])
    c, r = decompose(ss(candidate)), decompose(ss(reference))
    assert np.all(c['folds']['correlation'] < r['folds']['correlation'])
    assert np.all(c['pooled']['correlation'] > r['pooled']['correlation'])
    delta = symmetric_delta(c,r)
    return {'status':'PASS','direct_centering_cases':cases,
            'max_abs_direct_centering_error':max_error,
            'generated_both_folds_negative_pooled_positive':True,
            'generated_symmetric_reconstruction_error':delta['max_abs_reconstruction_error']}


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [jsonable(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest-only', action='store_true')
    args = ap.parse_args()
    started = time.monotonic()
    tests = generated_checks()
    if args.selftest_only:
        print(json.dumps(tests,indent=2)); return
    meta = metadata()
    results = {}
    hashes = {str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name in IDS:
        report_path = HERE/'evaluations'/(name+'.json')
        sums_path = HERE/'evaluations'/(name+'-sufficient.npz')
        report = json.loads(report_path.read_text())
        assert report['pure_only'] and report['sequences'] == 860
        assert report['provenance']['data_sources']['cache/chain_folds.npz']['sha256'] == METADATA_SHA
        assert list(report['delta_by_fold']) == ['0','1']
        for source, expected in report['provenance']['grader_sources'].items():
            actual = sha(ROOT/source)
            assert actual == expected
            hashes[source] = actual
        wrapper = HERE/'grader/evaluate.py'
        assert sha(wrapper) == report['provenance']['wrapper_sha256']
        hashes[str(wrapper.relative_to(ROOT))] = sha(wrapper)
        for path in (report_path, sums_path): hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(sums_path,allow_pickle=False) as z:
            assert set(z.files) == {'candidate','reference'}
            arrays = {arm:z[arm] for arm in ('candidate','reference')}
        for a in arrays.values(): assert a.shape == (860,2,6) and a.dtype == np.float64 and np.isfinite(a).all()
        assert np.array_equal(arrays['candidate'][...,[0,1,3]],arrays['reference'][...,[0,1,3]])
        arms = {arm:decompose(np.stack([a[:425].sum(0),a[425:].sum(0)])) for arm,a in arrays.items()}
        reproduced = {}
        errors = []
        for arm in ('candidate','reference'):
            direct = moments(arrays[arm].sum(0))['correlation']
            reproduced[arm+'_wp'] = float(direct.mean())
            reproduced[arm+'_wp_by_target'] = direct
            errors.append(abs(direct.mean()-report[arm+'_wp']))
            errors.append(float(np.max(np.abs(direct-report[arm+'_wp_by_target']))))
        reproduced['delta'] = reproduced['candidate_wp']-reproduced['reference_wp']
        reproduced['delta_by_fold'] = {str(f):float((arms['candidate']['folds']['correlation'][f]-
            arms['reference']['folds']['correlation'][f]).mean()) for f in (0,1)}
        errors.append(abs(reproduced['delta']-report['delta']))
        errors += [abs(reproduced['delta_by_fold'][f]-report['delta_by_fold'][f]) for f in ('0','1')]
        assert max(errors) < TOL
        results[name] = {'candidate':report['candidate'],'reference':report['reference'],
                         'reproduced_existing_readouts':reproduced,
                         'maximum_existing_report_reproduction_error':max(errors),
                         'arms':arms,'symmetric_delta':symmetric_delta(arms['candidate'],arms['reference']),
                         'existing_ci95_not_recomputed':report['delta_boot_ci95'],
                         'existing_uncertainty_scope':report['uncertainty_scope']}
    result = {'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'Posthoc descriptive decomposition of exactly three previously reported comparisons.',
              'original_targets_or_predictions_loaded':False,'new_metric_queries':0,
              'new_transforms_blends_models_or_gates':False,
              'source_sha256':hashes,'metadata':meta,'generated_checks':tests,
              'float64_absolute_tolerance':TOL,'axis_order':'fold [0,1], target [t0,t1]',
              'decomposition':'Symmetric product allocation: avg(kappa)*delta(r_fold) + avg(r_fold)*delta(kappa) + delta(C_between/D). Descriptive, non-causal; component attribution depends on chosen identity.',
              'limitations':['Exposed folds; this is posthoc explanation, not fresh confirmation.',
                             'Pooled WP remains the valid official metric; sign reversal is not evidence of leakage or a scoring defect.',
                             'No calibration recipe, model rescue, gate override, or selection follows.',
                             'Existing uncertainty is conditional on fitted models and does not cover seed variation, adaptive selection, or domain shift.'],
              'comparisons':results,'elapsed_seconds':time.monotonic()-started}
    out = HERE/'pooled-fold-diagnostic.json'
    out.write_text(json.dumps(jsonable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','output':str(out),'sha256':sha(out),
                      'seconds':result['elapsed_seconds'],
                      'readout_reproduction_error':max(v['maximum_existing_report_reproduction_error'] for v in results.values())},indent=2))


if __name__ == '__main__': main()
