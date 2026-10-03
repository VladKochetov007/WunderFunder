from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from wnn.splits import file_sha256, load_campaign_split

LOGGER = logging.getLogger(__name__)
INPUTS = [f'i0_p{i}' for i in range(22)] + [f'i0_dv{i}' for i in range(4)] + ['a2', 'a3']
ARMS = ['raw', 'affine', 'offset']


def save(name: str, value: object) -> None:
    """Save a reviewable artifact within owned scope."""
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def state(phase: str, **fields: object) -> None:
    """Persist state around bounded work."""
    value = json.loads((OUT / 'STATE.json').read_text())
    value.update(phase=phase, **fields)
    save('STATE.json', value)


def module(path: Path, name: str) -> object:
    """Import a frozen source without writing bytecode."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def features(x: np.ndarray) -> np.ndarray:
    """Compute exactly the five fixed input-only causal states."""
    delta = np.zeros_like(x[:, :26])
    delta[1:] = x[1:, :26] - x[:-1, :26]
    book = np.any(delta[:, :22] != 0, axis=1).astype(np.float32)
    trade = np.any(delta[:, 22:26] != 0, axis=1).astype(np.float32)
    alpha = 1 - 0.5 ** .01
    rows = np.arange(len(x), dtype=np.float32)
    last = np.maximum.accumulate(np.where(book > 0, rows, -10001))
    return np.column_stack([
        x[:, 26], x[:, 27],
        lfilter([alpha], [1, -(1 - alpha)], book).astype(np.float32),
        lfilter([alpha], [1, -(1 - alpha)], trade).astype(np.float32),
        np.log1p(rows - last).astype(np.float32),
    ])


def moments(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return six additive moments per target using exact scoring weights."""
    assert np.isfinite(y).all() and np.isfinite(p).all()
    y = np.clip(y.astype(np.float64), -2, 2)
    p = np.clip(p.astype(np.float64), -2, 2)
    w = np.abs(y)
    return np.stack([w.sum(0), (w*y).sum(0), (w*p).sum(0),
                     (w*y*y).sum(0), (w*p*p).sum(0), (w*y*p).sum(0)], axis=-1)


def score(m: np.ndarray) -> np.ndarray:
    """Pool additive moments before computing correlations and equal target mean."""
    w, y, p, yy, pp, yp = np.moveaxis(m, -1, 0)
    safe = np.maximum(w, 1e-300)
    vy, vp = np.maximum(yy-y*y/safe, 0), np.maximum(pp-p*p/safe, 0)
    good = (w >= 1e-8) & (vy/safe > 1e-16) & (vp/safe > 1e-16)
    corr = np.zeros_like(w)
    np.divide(yp-y*p/safe, np.sqrt(vy*vp), out=corr, where=good)
    corr = np.clip(corr, -1, 1)
    return np.concatenate([corr, corr.mean(axis=-1, keepdims=True)], axis=-1)


def interval(draws: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    """Describe fixed paired component bootstrap on t0, t1, average."""
    return {'observed_delta': observed.tolist(), 'bootstrap_mean': draws.mean(0).tolist(),
            'sd': draws.std(0, ddof=1).tolist(), 'ci95': np.quantile(draws, [.025,.975], axis=0).T.tolist()}


def summarize(m: np.ndarray, components: np.ndarray) -> dict[str, object]:
    """Aggregate sequences to components and resample paired arms."""
    unique = np.unique(components)
    cm = np.stack([m[components == c].sum(0) for c in unique])
    scores = score(cm.sum(0))
    rng = np.random.default_rng(271828)
    draws = np.stack([score(cm[rng.integers(len(cm), size=len(cm))].sum(0)) for _ in range(1000)])
    comparisons = {}
    for cand, ctrl in [(2,1), (2,0), (1,0)]:
        comparisons[f'{ARMS[cand]}_minus_{ARMS[ctrl]}'] = interval(draws[:,cand]-draws[:,ctrl], scores[cand]-scores[ctrl])
    return {'sequences': len(m), 'components': len(unique), 'scores': dict(zip(ARMS, scores.tolist())), 'comparisons': comparisons}


def main() -> None:
    """Run one frozen calibration/evaluation direction with streamed arrays."""
    start = time.monotonic()
    assert json.loads((OUT/'HYPOTHESES.json').read_text())['status'] == 'frozen_before_scores'
    cs = module(ROOT/'research/campaign_2026-09-30/audit/campaign_score.py', 'frozen_scoring')
    targets = cs.mmap_stored_npz(ROOT/'cache/valid_targets.npz')
    order = targets['row_group'].astype(int).tolist()
    positions = {g:i for i,g in enumerate(order)}
    splits = [load_campaign_split(ROOT/f'research/campaign_2026-09-29/artifacts/splits/fold{f}.json', ROOT) for f in range(4)]
    component_of = {g:c for source,g,c in splits[0].components if source == 'valid'}
    assert all({g:c for source,g,c in s.components if source == 'valid'} == component_of for s in splits)
    pools = {key:[(f,g,component_of[g]) for f in fs for g in splits[f].evaluation_groups] for key,fs in [('calibration',[2,3]), ('evaluation',[0,1])]}
    cal_c = {c for _,_,c in pools['calibration']}
    eval_c = {c for _,_,c in pools['evaluation']}
    assert not cal_c & eval_c
    selected = {}
    sampling = {}
    for key, pool in pools.items():
        ranked = sorted({c for _,_,c in pool}, key=lambda c: hashlib.sha256(f'strong-offset-v1:{c}'.encode()).hexdigest())
        chosen = ranked[:128]
        selected[key] = [(f,g,c) for f,g,c in pool if c in set(chosen)]
        sampling[key] = {'pool_components': len(ranked), 'pool_sequences': len(pool), 'components': chosen,
                         'cutoff_sha256': hashlib.sha256(f'strong-offset-v1:{chosen[-1]}'.encode()).hexdigest(),
                         'sequences': selected[key]}
    assert {f for f,_,_ in selected['evaluation']} == {0,1}
    save('sampling.json', sampling)
    state('verifying_parent_alignment', inspected_sequence_ids=sampling,
          commands=['research/campaign_2026-10-03/continuation/run_cpu.sh scientific-reproducer env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/reproduction/strong_falsifier.py'])
    identities = {}
    predictions = []
    for f,s in enumerate(splits):
        name = f'c20261001_d256_s3_f{f}'
        pred_path = ROOT/f'cache/pred_{name}_valid.npy'
        side_path = pred_path.with_suffix('.json')
        side = json.loads(side_path.read_text())
        prov_path = ROOT/f'artifacts/runs/{name}/provenance.json'
        prov = json.loads(prov_path.read_text())
        checkpoint = ROOT/f'artifacts/runs/{name}/model.pt'
        ep = checkpoint.with_name('model_ep3.pt')
        digests = {str(p.relative_to(ROOT)):file_sha256(p) for p in [pred_path,side_path,prov_path,checkpoint,ep,s.path]}
        assert side['prediction_sha256'] == digests[str(pred_path.relative_to(ROOT))]
        assert side['checkpoint_sha256'] == digests[str(ep.relative_to(ROOT))]
        assert side['row_groups'] == order and side['run'] == name and side['split'] == 'valid'
        assert side['epoch'] == 3 and side['gate_mode'] == 'soft'
        assert side['split_manifest_sha256'] == s.sha256 == prov['split_manifest_sha256']
        a = prov['args']
        assert a['name'] == name and a['hidden'] == 256 and a['arch'] == 'gru' and a['seed'] == 3
        assert a['checkpoint_policy'] == 'fixed' and a['checkpoint_epoch'] == 3 and a['no_eval_during_fit']
        assert prov['selected_epoch'] == 3 and a['valid_fold'] == f
        assert prov['checkpoint_sha256']['model_ep3.pt'] == side['checkpoint_sha256']
        assert set(prov['fit_row_groups']['train']) == set(s.train_groups)
        assert set(prov['fit_row_groups']['valid']) == set(s.valid_groups)
        assert set(prov['evaluation_row_groups']) == set(s.evaluation_groups)
        s.assert_fit_groups(prov['fit_row_groups']['train'], prov['fit_row_groups']['valid'])
        p = np.load(pred_path, mmap_mode='r')
        assert p.shape == targets['y'].shape and p.dtype == np.float32
        predictions.append(p)
        identities[name] = digests
        LOGGER.info('parent fold%d hashes/order/provenance verified', f)
    source_paths = [Path(__file__), OUT/'HYPOTHESES.json', OUT/'sampling.json', ROOT/'METRIC.md', ROOT/'scripts/screen_score.py', ROOT/'src/wnn/splits.py', ROOT/'src/wnn/data.py', ROOT/'cache/chains.npz', ROOT/'cache/chain_folds.npz', ROOT/'cache/valid_targets.npz', ROOT/'research/campaign_2026-09-30/audit/campaign_score.py']
    source_paths += [ROOT/'research/campaign_2026-10-03/eda/nonlinear'/n for n in ['REPORT.txt','HYPOTHESES.json','continuation_common.py','probe_offsets.py']]
    save('identities.json', {'parents':identities, 'sources':{str(p.relative_to(ROOT)):file_sha256(p) for p in source_paths},
                            'sampling_sha256':file_sha256(OUT/'sampling.json'), 'component_disjoint':True,
                            'runtime_model_verification':'Requested gpt-6.1-sol medium; runtime identity not exposed to script.'})
    parquet = pq.ParquetFile(ROOT/'datasets/valid.parquet')
    # This read deliberately excludes targets and masks.
    first_group = selected['evaluation'][0][1]
    original = module(ROOT/'research/campaign_2026-10-03/eda/nonlinear/continuation_common.py', 'nonlinear_frozen')
    table = parquet.read_row_group(first_group, columns=original.FEATURES, use_threads=False)
    full = np.column_stack([table[c].to_numpy() for c in original.FEATURES])
    x = full[:, list(range(22))+list(range(48,52))+[106,107]]
    expected = original.causal_design(full)[1]
    actual = features(x)
    assert np.array_equal(actual, expected)
    prefix_errors = {}
    for n in [1,99,100,512,1000,19999]:
        error = float(np.max(np.abs(features(x[:n])-actual[:n])))
        assert error == 0
        prefix_errors[str(n)] = error
    synthetic_y = np.array([[0,1],[2,-2],[-3,0],[.125,.5]], dtype=np.float32)
    synthetic_p = np.array([[9,-3],[.3,1],[-8,2],[.1,.7]], dtype=np.float32)
    our_score = score(moments(synthetic_y,synthetic_p))
    authoritative = cs.pooled(np.stack([np.stack([cs.moments(synthetic_y[:,k],synthetic_p[:,k],np.ones(4)) for k in range(2)])]))
    assert np.allclose(our_score, [authoritative['t0'],authoritative['t1'],authoritative['wp']], atol=1e-12)
    save('tests.json', {'input_only_causality_row_group':first_group,'prefix_max_abs_error':prefix_errors,
                        'nonlinear_source_feature_bit_parity':True,'synthetic_metric_parity':True})
    del table,full,x,expected,actual
    provenance = {'calibration':[], 'evaluation':[]}

    def read(f: int, g: int, c: int, phase: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Read one sequence and verify authoritative cache alignment."""
        tab = parquet.read_row_group(g, columns=INPUTS+['t0','t1','seq_ix','need_prediction','is_scored'], use_threads=False)
        xx = np.column_stack([tab[n].to_numpy() for n in INPUTS])
        yy = np.column_stack([tab[n].to_numpy() for n in ['t0','t1']]).astype(np.float32)
        need = tab['need_prediction'].to_numpy().astype(bool)
        mask = need & tab['is_scored'].to_numpy().astype(bool)
        i = positions[g]
        assert len(xx) == 20000 and int(tab['seq_ix'][0].as_py()) == int(targets['seq_ix'][i])
        assert np.array_equal(yy, targets['y'][i], equal_nan=True) and np.array_equal(mask,targets['mask'][i])
        pp = np.asarray(predictions[f][i])
        assert np.isfinite(pp[need]).all() and np.isfinite(xx).all()
        digest = hashlib.sha256()
        for array in [xx,yy,need,mask,pp]:
            digest.update(array.tobytes())
        provenance[phase].append({'fold':f,'row_group':g,'component':c,'seq_ix':int(targets['seq_ix'][i]),
                                  'selected_rows':int(mask.sum()),'input_target_masks_prediction_sha256':digest.hexdigest(),
                                  'warmup_input_sha256':hashlib.sha256(xx[:99].tobytes()).hexdigest()})
        return features(xx), yy, pp, mask

    state('streaming_calibration_sufficient_statistics')
    gram = np.zeros((7,7))
    rhs = np.zeros(7)
    total = np.zeros(5)
    squares = np.zeros(5)
    count = 0
    for j,(f,g,c) in enumerate(selected['calibration']):
        ss,yy,pp,mask = read(f,g,c,'calibration')
        ss = ss[mask].astype(np.float64)
        y = np.clip(yy[mask,1].astype(np.float64),-2,2)
        design = np.column_stack([np.ones(len(y)),pp[mask,1],ss])
        w = np.abs(y)+.001
        gram += design.T @ (w[:,None]*design)
        rhs += design.T @ (w*y)
        total += ss.sum(0)
        squares += (ss*ss).sum(0)
        count += len(y)
        if j%50 == 0:
            LOGGER.info('calibration %d/%d',j+1,len(selected['calibration']))
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean,0))+1e-6
    transform = np.eye(7)
    transform[2:,2:] = np.diag(1/std)
    transform[0,2:] = -mean/std
    penalty = np.diag([0,0]+[.001*count]*5)
    affine = np.linalg.solve(gram[:2,:2],rhs[:2])
    normalized_beta = np.linalg.solve(transform.T@gram@transform+penalty,transform.T@rhs)
    offset = transform@normalized_beta
    np.savez(OUT/'corrector.npz',mean=mean,std=std,affine=affine,offset_raw_state=offset,
             offset_standardized=normalized_beta,gram=gram,rhs=rhs,count=count,penalty=penalty)
    save('coefficients.json', {'calibration_rows':count,'state_mean':mean.tolist(),'state_std':std.tolist(),
                              'affine_intercept_slope':affine.tolist(),'offset_standardized':normalized_beta.tolist(),
                              'offset_raw_states':offset.tolist(),'condition_number':float(np.linalg.cond(transform.T@gram@transform+penalty))})
    save('provenance_calibration.json',provenance['calibration'])
    state('evaluating_fixed_correctors',artifacts=['sampling.json','identities.json','tests.json','corrector.npz','coefficients.json','provenance_calibration.json'])
    rows,folds,classes,components,groups = [],[],[],[],[]
    parity_max = 0.
    shifts = []
    for j,(f,g,c) in enumerate(selected['evaluation']):
        ss,yy,pp,mask = read(f,g,c,'evaluation')
        p = pp[mask]
        y = yy[mask]
        design = np.column_stack([np.ones(len(p)),p[:,1],ss[mask]])
        ap,op = p.copy(),p.copy()
        ap[:,1] = design[:,:2]@affine
        op[:,1] = design@offset
        assert np.array_equal(ap[:,0],p[:,0]) and np.array_equal(op[:,0],p[:,0])
        assert np.isfinite(ap).all() and np.isfinite(op).all()
        m = np.stack([moments(y,arm) for arm in [p,ap,op]])
        if j < 3:
            for k,arm in enumerate([p,ap,op]):
                ref = cs.pooled(np.stack([np.stack([cs.moments(y[:,t],arm[:,t],np.ones(len(y))) for t in range(2)])]))
                err = float(np.max(np.abs(score(m[k])-[ref['t0'],ref['t1'],ref['wp']])))
                parity_max = max(parity_max,err)
                assert err < 1e-10
        rows.append(m)
        folds.append(f)
        classes.append(int(targets['market_class'][positions[g]]))
        components.append(c)
        groups.append(g)
        shifts.append([float(np.mean(ap[:,1]-p[:,1])),float(np.mean(op[:,1]-ap[:,1])),float(np.mean(np.abs(op[:,1]-ap[:,1])))])
        if j%50 == 0:
            LOGGER.info('evaluation %d/%d',j+1,len(selected['evaluation']))
    m = np.stack(rows)
    folds,classes,components,groups = [np.array(v) for v in [folds,classes,components,groups]]
    np.savez(OUT/'sequence_moments.npz',moments=m,folds=folds,classes=classes,components=components,row_groups=groups,arms=np.array(ARMS),shifts=np.array(shifts))
    unique = np.unique(components)
    np.savez(OUT/'component_moments.npz',moments=np.stack([m[components==c].sum(0) for c in unique]),components=unique,arms=np.array(ARMS))
    save('provenance_evaluation.json',provenance['evaluation'])
    state('paired_component_bootstrap')
    pooled = summarize(m,components)
    by_fold = {str(f):summarize(m[folds==f],components[folds==f]) for f in np.unique(folds)}
    by_class = {str(k):summarize(m[classes==k],components[classes==k]) for k in np.unique(classes)}
    delta = pooled['comparisons']['offset_minus_affine']['observed_delta']
    lower = pooled['comparisons']['offset_minus_affine']['ci95'][2][0]
    gate = delta[1]>=.005 and lower>0 and all(part['comparisons']['offset_minus_affine']['observed_delta'][1]>0 for part in by_fold.values()) and pooled['scores']['offset'][2]>pooled['scores']['raw'][2]
    result = {'metric_id':'global_wp_clip2_v1','pooled':pooled,'folds':by_fold,'classes':by_class,
              'gate_passed':bool(gate),'wall_seconds':time.monotonic()-start,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'calibration_sequences':len(selected['calibration']),'calibration_rows':count,'evaluation_selected_rows':sum(r['selected_rows'] for r in provenance['evaluation']),
              't0_bit_identical':True,'frozen_scorer_sequence_parity_max_abs_error':parity_max,
              'shift_columns':['affine_minus_raw_signed_mean','offset_minus_affine_signed_mean','offset_minus_affine_absolute_mean'],
              'shift_sequence_means':np.mean(shifts,axis=0).tolist(), 'exposure':'Exploratory exposed validation; calibration/evaluation components disjoint; parent training is fold-bound, not nested across calibration/evaluation pools.'}
    save('results.json',result)
    tests = json.loads((OUT/'tests.json').read_text())
    tests.update(t0_bit_identical=True,real_sequence_metric_parity_max_abs_error=parity_max,cache_parquet_alignment_all_selected=True,all_parent_hashes_verified=True)
    save('tests.json',tests)
    lines = ['INDEPENDENT STRONG-BASELINE FALSIFIER',
             'VERDICT: '+('Frozen strong-baseline gate passed.' if gate else 'Reject useful strong-baseline transport under the frozen gate.'),
             'This is exposed exploratory validation, never fresh or public confirmation.',
             'One fixed direction: folds2+3 calibration, folds0+1 evaluation. Five causal states; t1 only. No sweep.',
             f'Calibration: {len(selected["calibration"])} sequences,128 components,{count} selected rows.',
             f'Evaluation: {len(m)} sequences,{len(unique)} components,{result["evaluation_selected_rows"]} selected rows.',
             'First128 component SHA256 ranks frozen in sampling.json before any scores; all member sequences retained.',
             'Scores below list t0,t1,equal target mean; deltas are differences of pooled GlobalWP, not per-sequence score averages.']
    for label,part in [('pooled',pooled)]+[(f'fold{f}',v) for f,v in by_fold.items()]+[(f'class{k} diagnostic',v) for k,v in by_class.items()]:
        lines.append(f'\n{label}: {part["sequences"]} sequences,{part["components"]} components')
        for arm,scores in part['scores'].items():
            lines.append(f'  {arm}: '+', '.join(f'{v:.9f}' for v in scores))
        for name,cmp in part['comparisons'].items():
            lines.append(f'  {name}: observed {cmp["observed_delta"]}; bootstrap mean {cmp["bootstrap_mean"]};95%CI {cmp["ci95"]}')
    lines += ['\nCalibration coefficients [intercept,parent slope,five standardized offsets]: '+str(normalized_beta.tolist()),
              'Affine [intercept,parent slope]: '+str(affine.tolist()),
              'State mean/std: '+str(mean.tolist())+' / '+str(std.tolist()),
              'Prediction shifts (unweighted sequence means): '+str(result['shift_sequence_means']),
              'Fit mirrors frozen nonlinear weightedMSE: abs(clipped y)+.001; state ridge .001*N; affine terms unpenalized. Exact scoring weights have NO epsilon.',
              'All warmup/unscored inputs update states; age starts10001; EWMA zero initialized. No labels/mask/classes at correction inference.',
              'Tests: source feature bit parity; input-only prefixes1/99/100/512/1000/19999 exact; synthetic and real frozen metric parity; t0 bit identical; all selected parquet/cache targets/masks/IDs aligned.',
              'Parent sidecars, prediction bytes, model_ep3.pt binding and checkpoint hashes, provenance hidden256/GRU/seed3/fixedepoch3 and fold-specific component exclusion verified. Components reconstructed by authoritative split loader from chains.npz.',
              'LIMITATION: The calibrator training components are disjoint from evaluation components, but calibration parent folds2/3 were trained on other folds, including evaluation components. Parent OOF errors are therefore not nested-independent of this outer calibration split. No retraining authorized; paired CI is exploratory conditional on fixed parents.',
              'Retrospective target classes are diagnostics only, never inference inputs or adaptive selection.',
              'Negative gates and all arms retained. No scalar search, reverse cross-fit, new architecture or public-score forecast.',
              f'CPU wrapper:1core2GiB15min; measured wall {result["wall_seconds"]:.2f}s,maxRSS {result["max_rss_mib"]:.1f}MiB. No GPU or installs.',
              'Runtime agent identity is not exposed to this script; requested gpt-6.1-sol medium recorded, not independently attested.',
              'Artifacts: standalone strong_falsifier.py,HYPOTHESES.json,sampling.json,identities.json,tests.json,provenance*.json,corrector.npz,coefficients.json,sequence_moments.npz,component_moments.npz,results.json,run.log.',
              'Reproduce: research/campaign_2026-10-03/continuation/run_cpu.sh scientific-reproducer env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/reproduction/strong_falsifier.py']
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    state('completed',pending_work=[],last_failure=None,verdict=lines[1],wall_seconds=result['wall_seconds'],artifacts=sorted(p.name for p in OUT.iterdir() if p.is_file()))
    LOGGER.info('%s; pooled delta %s; CI %s',lines[1],delta,pooled['comparisons']['offset_minus_affine']['ci95'])


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OUT/'failure.log').write_text(failure)
        state('failed',last_failure=failure)
        raise
