"""Two fixed context-reset contrasts, independently scored on the same block rows.

Reads only completed replay output after root authorization; does not run inference,
fit models, choose origins, change features/weights, or select a winning mixture.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ARMS = ('full', 'reset256', 'reset2048')
ENGINES = ('amp_fp16','fp32_tf32_off')
STARTS = np.array([4096, 8192, 12288, 16384])
TARGETS = 2
BOOTSTRAPS = 2000
SEED = 20261001
META = ('cache_position','row_group','step','chain','fold','market_class_retrospective','age_i0')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def clean(obj):
    if isinstance(obj, dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, (np.integer,np.bool_)): return obj.item()
    if isinstance(obj, (float,np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')


def blend(member, partners):
    """Preserve original float32 sequential member addition, for EVERY arm."""
    assert member.dtype == partners.dtype == np.float32
    assert member.shape[-1] == partners.shape[-1] == 2
    return (((member+partners[:,0])+partners[:,1])+partners[:,2])/np.float32(4)


def sums_by_group(y, predictions, group, groups):
    """[group,arm,target,(W,WY,WP,WYY,WPP,WYP)] after final-ensemble clipping."""
    y = np.clip(np.asarray(y, np.float64), -2, 2)
    p = np.clip(np.asarray(predictions, np.float64), -2, 2)
    w = np.abs(y)
    out = np.empty((groups,p.shape[1],2,6),np.float64)
    for a in range(p.shape[1]):
        for t in range(2):
            vals = (w[:,t], w[:,t]*y[:,t], w[:,t]*p[:,a,t],
                    w[:,t]*y[:,t]**2, w[:,t]*p[:,a,t]**2, w[:,t]*y[:,t]*p[:,a,t])
            for s,v in enumerate(vals): out[:,a,t,s] = np.bincount(group,weights=v,minlength=groups)
    return out


def metric(s):
    W, Y, P, YY, PP, YP = np.moveaxis(np.asarray(s,np.float64),-1,0)
    safe = np.where(W>0,W,1)
    my, mp = Y/safe,P/safe
    vy, vp = np.maximum(YY/safe-my*my,0),np.maximum(PP/safe-mp*mp,0)
    sy,sp = np.sqrt(vy),np.sqrt(vp)
    valid = (W>=1e-8)&(sy>1e-8)&(sp>1e-8)
    rho = np.zeros_like(W)
    np.divide(YP/safe-my*mp,sy*sp,out=rho,where=valid)
    rho = np.clip(rho,-1,1)
    return dict(W=W,mu_y=my,mu_p=mp,sy=sy,sp=sp,rho=rho,
                score=rho.mean(axis=-1),valid=valid)


def fixed_rows(y, p, full_parameters):
    """Reference-normalized finite error change and first-order postclip WP effect."""
    y,p = np.clip(y.astype(np.float64),-2,2),np.clip(p.astype(np.float64),-2,2)
    w = abs(y)
    par = full_parameters
    assert np.all(par['valid'])
    u = (y-par['mu_y'])/par['sy']
    z0 = (p[:,0]-par['mu_p'])/par['sp']
    e = u-z0
    q = u-par['rho']*z0
    dz = (p-p[:,0:1])/par['sp']
    # First column is the fixed-reference energy itself; deltas subtract arm0 later.
    error = w[:,None]*(e[:,None]-dz)**2
    linear = w[:,None]*q[:,None]*dz
    return error,linear,dz,w


def bootstrap_counts(chain_metadata,n=BOOTSTRAPS):
    """Paired resampling of whole chains, stratified by existing fold and class."""
    rng = np.random.default_rng(SEED)
    draws = np.zeros((n,len(chain_metadata)),np.int16)
    strata = []
    for f in sorted(np.unique(chain_metadata[:,0])):
        for c in sorted(np.unique(chain_metadata[:,2])):
            idx = np.flatnonzero((chain_metadata[:,0]==f)&(chain_metadata[:,2]==c))
            if not len(idx): continue
            sample = rng.integers(len(idx),size=(n,len(idx)))
            for j in range(n): draws[j,idx] = np.bincount(sample[j],minlength=len(idx))
            strata.append(dict(fold=int(f),retrospective_class=int(c),chains=len(idx)))
    return draws,strata


def ci(values,valid=None):
    v = np.asarray(values)
    good = np.isfinite(v) if valid is None else np.isfinite(v)&valid
    return dict(valid_draws=int(good.sum()),interval95=np.quantile(v[good],[.025,.975]) if good.any() else None)


def summarize_scope(chain_sums,chain_fixed,chain_linear,chain_count,chain_metadata,draws,global_W):
    s = chain_sums.sum(0)
    par = metric(s)
    chain_weights = chain_sums[:,0,:,0]
    weights = chain_weights.sum(0)
    fsum = chain_fixed.sum(0)
    lsum = chain_linear.sum(0)
    energy = .25*np.sum(fsum/global_W,axis=-1)
    first_order = .5*np.sum(lsum/global_W,axis=-1)
    primary = dict(scored_rows=int(chain_count.sum()),chains_with_scored_rows=int((chain_count>0).sum()),
        metric=par,reference_fixed_energy=energy,
        fixed_energy_difference=energy-energy[0],
        postclip_first_order_WP_contribution=first_order,
        target_fixed_error_intensity=np.divide(fsum,weights[None],out=np.full_like(fsum,np.nan),where=weights[None]>0),
        metric_weight_share=.5*np.sum(weights/global_W))
    boot = (draws@chain_sums.reshape(len(chain_sums),-1)).reshape(len(draws),3,2,6)
    bm = metric(boot)
    valid_all = bm['valid'].all(-1)
    fixed = .25*np.sum((draws@chain_fixed.reshape(len(chain_sums),-1)).reshape(len(draws),3,2)/global_W,axis=-1)
    linear = .5*np.sum((draws@chain_linear.reshape(len(chain_sums),-1)).reshape(len(draws),3,2)/global_W,axis=-1)
    contrasts = {}
    for a,name in enumerate(ARMS[1:],1):
        d = bm['score'][:,a]-bm['score'][:,0]
        dr = bm['rho'][:,a]-bm['rho'][:,0]
        contrasts[name] = dict(WP_difference=float(par['score'][a]-par['score'][0]),
            rho_difference=par['rho'][a]-par['rho'][0],
            conditional_WP_CI=ci(d,valid_all[:,a]&valid_all[:,0]),
            conditional_rho_CI=[ci(dr[:,t],bm['valid'][:,a,t]&bm['valid'][:,0,t]) for t in (0,1)],
            fixed_energy_difference=float(energy[a]-energy[0]),
            conditional_fixed_energy_CI=ci(fixed[:,a]-fixed[:,0]),
            postclip_first_order_WP_contribution=float(first_order[a]),
            conditional_first_order_CI=ci(linear[:,a]))
        eligible=np.all(chain_weights>0,axis=1)
        cd=.5*np.sum(np.divide(chain_fixed[:,a]-chain_fixed[:,0],chain_weights,
                               out=np.zeros_like(chain_weights),where=chain_weights>0),axis=1)[eligible]
        contrasts[name]['chain_fixed_intensity_difference']=dict(
            eligible_chains=int(eligible.sum()),median=float(np.median(cd)) if len(cd) else None,
            lower_error=int((cd < -1e-12).sum()),higher_error=int((cd > 1e-12).sum()),
            numerical_ties=int((abs(cd)<=1e-12).sum()),
            definition='Equal-target within-chain mean fixed-normalized squared-error change. Negative helps; ties use absolute1e-12 solely for descriptive rounding.')
    primary['contrasts'] = contrasts
    return primary


def analyze(y,p,metadata,mask,bootstrap=BOOTSTRAPS):
    assert mask.dtype == bool and mask.shape == (len(y),)
    assert y.shape == (len(y),2) and metadata.shape == (len(y),7) and p.shape == (len(y),3,2)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    assert np.all(metadata[:,2]>=99)
    block = np.searchsorted(STARTS,metadata[:,2],side='right')-1
    assert np.all(block>=0) and np.all(metadata[:,2]-STARTS[block]<512)
    md = metadata[mask]
    yy,pp = y[mask],p[mask]
    keys,ch = np.unique(md[:,[4,3,5]],axis=0,return_inverse=True)
    assert len({(int(f),int(c)) for f,c,_ in keys}) == len(keys)
    full_sums = sums_by_group(yy,pp,ch,len(keys))
    full_par = metric(full_sums.sum(0)[0])
    err,lin,dz,w = fixed_rows(yy,pp,full_par)
    baseline_energy=.25*np.sum(err[:,0].sum(0)/full_par['W'])
    assert abs(baseline_energy-(1-float(full_par['score']))) < 1e-10
    draws,strata = bootstrap_counts(keys,bootstrap)
    classes,ages=md[:,5],md[:,6]
    scopes = {'all':np.ones(len(md),bool)}
    for c in (2,3):
        scopes[f'class{c}_quiet64']=(classes==c)&(ages>=64)
        scopes[f'class{c}_fresh0_15']=(classes==c)&(ages<16)
    all_results = {}
    saved = {}
    for name,keep in scopes.items():
        cs = sums_by_group(yy[keep],pp[keep],ch[keep],len(keys))
        cf = np.empty((len(keys),3,2))
        cl = np.empty_like(cf)
        for a in range(3):
            for t in range(2):
                cf[:,a,t] = np.bincount(ch[keep],weights=err[keep,a,t],minlength=len(keys))
                cl[:,a,t] = np.bincount(ch[keep],weights=lin[keep,a,t],minlength=len(keys))
        cn=np.bincount(ch[keep],minlength=len(keys))
        summary=summarize_scope(cs,cf,cl,cn,keys,draws,full_par['W'])
        summary['folds']={}
        for f in (0,1):
            fk=keys[:,0]==f
            summary['folds'][str(f)] = summarize_scope(cs[fk],cf[fk],cl[fk],cn[fk],keys[fk],draws[:,fk],full_par['W'])
        all_results[name]=summary
        saved[f'{name}_metric_sums']=cs
        saved[f'{name}_fixed_error_sums']=cf
        saved[f'{name}_linear_sums']=cl
        saved[f'{name}_row_counts']=cn
    sensitivity=[]
    offset=md[:,2]-STARTS[np.searchsorted(STARTS,md[:,2],side='right')-1]
    for label,keep in [('all',np.ones(len(md),bool)),('offset0_63',offset<64),('offset64_255',(offset>=64)&(offset<256)),('offset256_511',offset>=256),*scopes.items()]:
        if label=='all' and any(r['scope']=='all' for r in sensitivity): continue
        for a in (1,2):
            raw=pp[keep,a].astype(np.float64)-pp[keep,0].astype(np.float64)
            clip=np.clip(pp[keep,a].astype(np.float64),-2,2)-np.clip(pp[keep,0].astype(np.float64),-2,2)
            weight=w[keep].sum(0)
            sensitivity.append(dict(scope=label,arm=ARMS[a],rows=int(keep.sum()),
                raw_mean_difference=raw.mean(0) if len(raw) else [None,None],
                raw_abs_quantiles_by_target=np.quantile(abs(raw),[.5,.95,.99,1],axis=0).T if len(raw) else None,
                clipped_weighted_RMS=np.sqrt(np.divide((w[keep]*clip**2).sum(0),weight,out=np.full(2,np.nan),where=weight>0)),
                clipped_weighted_mean=np.divide((w[keep]*clip).sum(0),weight,out=np.full(2,np.nan),where=weight>0)))
    result=dict(reference_normalizers=full_par,scopes=all_results,sensitivity=sensitivity,
        uncertainty=dict(draws=bootstrap,seed=SEED,strata=strata,
            scope='Paired whole-chain bootstrap conditional on fixed checkpoints, block selection and arms. WP renormalizes each resample; fixed-energy/linear intervals resample ORIGINAL per-chain contributions, retaining original observed full-control W, means and SDs. They are not fully renormalized bootstrap deficits. Marginal 95% intervals, not simultaneous familywise intervals.'),
        definitions=dict(primary='Two planned reset-minus-full pooled clipped WP contrasts over all scored rows of all four blocks; not average block/sequence correlation.',
            fixed_error='Global reference normalizers are estimated on full-engine control over the same four-block scored population. Intervention fixed-energy change is NOT exactly minus its renormalized WP change.',
            folds='Fold WP renormalizes within that fold; fixed-energy/linear fold entries remain contributions using pooled-two-fold reference normalizers and total W.',
            linear='Half the target sum of E_w[(u-rho*z_full)*(clip(p_reset)-clip(p_full))/sd_p_full], a first-order approximation in postclip space; not an extra correction.',
            focus='class3 quiet>=64; class2 quiet/fresh and class3 fresh contrasts are descriptive localization, not selection gates.',
            inference='GRU-state ablation with fixed full-prefix features. A benefit would not by itself establish a deployable reset or a training gain.'))
    return result,dict(chain_metadata=keys,**saved)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--source-sha256',required=True)
    ap.add_argument('--plan-sha256',required=True)
    a=ap.parse_args();start=time.monotonic()
    assert sha(__file__)==a.source_sha256 and sha(a.plan)==a.plan_sha256
    assert not a.output_dir.exists()
    receipt=json.loads(a.receipt.read_text())
    assert receipt['outputs'][a.input.name]['sha256']==sha(a.input)
    # Producer's exact completion/parity fields are finalized after source review.
    assert receipt.get('parity_checks_passed') is True
    assert receipt.get('status') == 'COMPLETE'
    provenance=dict(source_sha256=a.source_sha256,plan_sha256=a.plan_sha256,
                    input_sha256=sha(a.input),replay_receipt_sha256=sha(a.receipt))
    a.output_dir.mkdir(parents=True)
    write_json(a.output_dir/'started.json',provenance)
    try:
        with np.load(a.input,allow_pickle=False) as z:
            md,y,mask,member,partners,cached=[z[k] for k in ('metadata','y','is_scored','member_predictions','partner_predictions','cached_shallow_reference')]
            assert tuple(z['metadata_names'].tolist())==META
            assert tuple(z['arm_names'].tolist())==ARMS
            assert tuple(z['engine_names'].tolist())==ENGINES
            assert tuple(z['partner_names'].tolist())==('on_ctl128s3','on_ctl128s4','on_ctlc500')
            np.testing.assert_array_equal(z['block_starts'],STARTS)
            assert int(z['block_length'])==512
        assert len(y)==860*4*512 and len(np.unique(md[:,0]))==860
        assert member.shape==(len(y),2,3,2) and partners.shape==(len(y),3,2)
        result,arrays={'engines':{},'precision_context':{}},{}
        engine_deltas=[]
        for e,engine in enumerate(ENGINES):
            predictions=np.stack([blend(member[:,e,a],partners) for a in range(3)],axis=1)
            er,ea=analyze(y,predictions,md,mask)
            result['engines'][engine]=er
            arrays.update({f'{engine}_{k}':v for k,v in ea.items()})
            drift=member[:,e,0].astype(np.float64)-cached.astype(np.float64)
            result['precision_context'][engine]=dict(
                full_member_vs_archived_AMP_cache_all_block_rows_max_abs=float(np.max(abs(drift))),
                full_member_vs_archived_AMP_cache_all_block_rows_RMS=float(np.sqrt(np.mean(drift**2))),
                scored_rows_max_abs_by_target=np.max(abs(drift[mask]),axis=0),
                scored_rows_RMS_by_target=np.sqrt(np.mean(drift[mask]**2,axis=0)))
            engine_deltas.append(predictions[:,1:].astype(np.float64)-predictions[:,0:1].astype(np.float64))
        precision_difference=engine_deltas[0][mask]-engine_deltas[1][mask]
        result['precision_context']['difference_of_reset_effects_AMP_minus_FP32']=dict(
            RMS_by_arm_target=np.sqrt(np.mean(precision_difference**2,axis=0)),
            max_abs_by_arm_target=np.max(abs(precision_difference),axis=0),
            note='Descriptive engine sensitivity, not another selected intervention. Cached three partners stay fixed GPU/AMP in both engines; FP32 applies to the replayed shallow member only.')
        result.update(status='COMPLETE_TWO_FIXED_CONTEXT_CONTRASTS_NO_FITS',provenance=provenance,
                      seconds=time.monotonic()-start,peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(a.output_dir/'results.json',result)
        np.savez_compressed(a.output_dir/'sufficient-results.npz',**arrays)
        write_json(a.output_dir/'receipt.json',dict(status=result['status'],**provenance,
            outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(a.output_dir.iterdir()) if p.is_file()}))
        print(json.dumps(clean(dict(status=result['status'],seconds=result['seconds'],primary={e:r['scopes']['all']['contrasts'] for e,r in result['engines'].items()}))))
    except Exception as exc:
        write_json(a.output_dir/'failure.json',dict(exception_type=type(exc).__name__,message=str(exc),provenance=provenance))
        raise


if __name__=='__main__':main()
