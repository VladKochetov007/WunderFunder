"""One fixed ensemble-objective study: fresh matched control plus old operational reference.

Only main() reads study arrays. --schema and generated tests are source-only.
Binding schema follows the previous student grader and names the completed root
allocation explicitly; evaluator source is frozen before all four new OOF are scored.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CONT=ROOT/'research/iterations/2026-10-01-continuation'
OLD=ROOT/'research/iterations/2026-10-01-overnight'
EDA=ROOT/'research/iterations/2026-10-01-error-analysis'
MEMBERS=('eb_candidate_s3_h2','eb_control_s3_h2','cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')
RECIPES={'candidate4':(MEMBERS[0],*MEMBERS[3:]),
         'fresh_control4':(MEMBERS[1],*MEMBERS[3:]),'current_student4':MEMBERS[2:]}
COMPARISONS={'primary':('candidate4','fresh_control4'),'operational':('candidate4','current_student4')}
SCOPES=('all','class3_quiet64')
COMPLETION='COMPLETED_8_FITS_4_OOF_NO_METRICS'
METRIC_SHA='d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'
BASELINE_WP=.7023796676142182
BASELINE_RHO=np.array([.6986436001508309,.7061157350776048])
SEED=20261001
BOOTSTRAPS=2000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def write(path,value):path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def path_for(rel):
    p=Path(rel)
    assert isinstance(rel,str) and rel and not p.is_absolute() and '..' not in p.parts
    assert str(p)==rel
    return ROOT/p


def schema():
    return dict(status='ROOT_ALLOCATED_FOR_FIXED_ENSEMBLE_BACKBONE',folds=[0,1],
        recipes={k:list(v) for k,v in RECIPES.items()},
        sources={'repository-relative source path':'sha256, including evaluator and predict_fixed.py'},
        data={'cache/valid_targets.npz':{'bytes':'integer','sha256':'hex'},
              'cache/chain_folds.npz':{'bytes':'integer','sha256':'hex'}},
        root_allocation={'path':'completed root pipeline receipt relative to repository',
                         'sha256':'hex','required_field':'explicit completion status key',
                         'required_value':'exact successful value proving all eight fits and four OOF complete'},
        predictions={f'{name}_f{fold}':{'path':'repository-relative .npy','sha256':'hex',
                      'receipt':'repository-relative .inputs.json','receipt_sha256':'hex','checkpoint_sha256':'hex'}
                     for fold in (0,1) for name in MEMBERS},
        existing_eda={'path':str((EDA/'extraction-01/scored-rows.npz').relative_to(ROOT)),
                      'sha256':'hex','receipt':str((EDA/'extraction-01/receipt.json').relative_to(ROOT)),
                      'receipt_sha256':'hex'},
        primary='Fixed candidate4 versus fresh_control4; both folds, all860pure sequences/250chains/1614484 scored rows afterwarmup99.',
        operational='Fixed candidate4 versus current_student4; not a competing selectable primary.',
        comparisons={k:list(v) for k,v in COMPARISONS.items()},
        uncertainty='2000paired whole-chain draws, seed20261001, stratafold x retrospectiveclass; fixed-model conditional intervals.',
        diagnostics=list(SCOPES[1:]),standalone_queries=0,raw_reextraction=0)


def blend(parts):
    assert len(parts)==4 and all(p.dtype==np.float32 for p in parts)
    return (((parts[0]+parts[1])+parts[2])+parts[3])/np.float32(4)


def sufficient(y,p):
    """Independent six-sum implementation, clipping only AFTER ensemble formation."""
    y=np.clip(np.asarray(y,np.float64),-2,2)
    p=np.clip(np.asarray(p,np.float64),-2,2)
    assert y.shape==p.shape and y.shape[-1]==2
    w=abs(y)
    return np.stack([w.sum(0),(w*y).sum(0),(w*p).sum(0),
                     (w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)],axis=-1)


def metric(s):
    W,Y,P,YY,PP,YP=np.moveaxis(np.asarray(s,np.float64),-1,0)
    safe=np.where(W>0,W,1)
    my,mp=Y/safe,P/safe
    vy,vp=np.maximum(YY/safe-my*my,0),np.maximum(PP/safe-mp*mp,0)
    valid=(W>=1e-8)&(vy>1e-16)&(vp>1e-16)
    rho=np.zeros_like(W)
    np.divide(YP/safe-my*mp,np.sqrt(vy*vp),out=rho,where=valid)
    rho=np.clip(rho,-1,1)
    return dict(W=W,rho=rho,WP=rho.mean(-1),valid=valid,
                mu_y=my,mu_p=mp,sd_y=np.sqrt(vy),sd_p=np.sqrt(vp))


def resamples(metadata,n=BOOTSTRAPS):
    rng=np.random.default_rng(SEED)
    draws=np.zeros((n,len(metadata)),np.int16)
    strata=[]
    for fold in (0,1):
        for cl in sorted(np.unique(metadata[:,2])):
            idx=np.flatnonzero((metadata[:,0]==fold)&(metadata[:,2]==cl))
            if not len(idx):continue
            sampled=rng.integers(len(idx),size=(n,len(idx)))
            for b in range(n):draws[b,idx]=np.bincount(sampled[b],minlength=len(idx))
            strata.append(dict(fold=fold,retrospective_class=int(cl),chains=len(idx)))
    return draws,strata


def interval(values,valid):
    keep=valid&np.isfinite(values)
    return dict(valid_draws=int(keep.sum()),interval95=np.quantile(values[keep],[.025,.975]) if keep.any() else None)


def comparison(chain_stats,counts,draws):
    # Candidate index0; same fixed quarter-weight control index1.
    m=metric(chain_stats.sum(0))
    bs=(draws@chain_stats.reshape(len(chain_stats),-1)).reshape(len(draws),2,2,6)
    bm=metric(bs)
    valid=bm['valid'][:,0]&bm['valid'][:,1]
    difference=bm['WP'][:,0]-bm['WP'][:,1]
    out=dict(scored_rows=int(counts.sum()),chains_with_scored_rows=int((counts>0).sum()),
        candidate= {k:v[0] for k,v in m.items()},control={k:v[1] for k,v in m.items()},
        delta_WP=float(m['WP'][0]-m['WP'][1]),delta_rho=m['rho'][0]-m['rho'][1],
        paired_chain_CI=interval(difference,valid.all(-1)),
        paired_target_CI=[interval(bm['rho'][:,0,t]-bm['rho'][:,1,t],valid[:,t]) for t in (0,1)])
    return out


def describe(chain_stats,counts,metadata,bootstrap=BOOTSTRAPS):
    draws,strata=resamples(metadata,bootstrap)
    result={}
    names=list(RECIPES)
    for label,pair in COMPARISONS.items():
        pair_idx=[names.index(name) for name in pair]
        scopes={}
        for j,name in enumerate(SCOPES):
            selected=chain_stats[:,j][:,pair_idx]
            scopes[name]=comparison(selected,counts[:,j],draws)
            scopes[name]['folds']={}
            for f in (0,1):
                k=metadata[:,0]==f
                scopes[name]['folds'][str(f)]=comparison(selected[k],counts[k,j],draws[:,k])
        result[label]=dict(candidate=pair[0],control=pair[1],scopes=scopes)
    return dict(comparisons=result,uncertainty=dict(draws=bootstrap,seed=SEED,strata=strata,
        interpretation='Same paired whole-chain draws for both fixed contrasts; each draw recomputes target weights/means/variances. Fixed-model conditional, marginal95% intervals, not simultaneous; omit seed, training, selection and domain-shift uncertainty.'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--schema',action='store_true')
    ap.add_argument('--bindings',type=Path)
    ap.add_argument('--bindings-sha256')
    ap.add_argument('--output-dir',type=Path)
    a=ap.parse_args()
    if a.schema:print(json.dumps(schema(),indent=2));return
    assert a.bindings and a.bindings_sha256 and a.output_dir
    assert sha(a.bindings)==a.bindings_sha256
    bind=json.loads(a.bindings.read_text())
    assert bind['status']=='ROOT_ALLOCATED_FOR_FIXED_ENSEMBLE_BACKBONE' and bind['folds']==[0,1]
    assert bind['recipes']=={k:list(v) for k,v in RECIPES.items()}
    assert bind['comparisons']=={k:list(v) for k,v in COMPARISONS.items()}
    out=a.output_dir.resolve();assert out.is_relative_to(HERE.resolve()) and out!=HERE.resolve() and not out.exists()
    tracked={};input_stats={}
    def pin(path,digest):
        assert len(digest)==64 and sha(path)==digest,('input hash',str(path))
        tracked[str(path)]=digest
        st=path.stat();input_stats[str(path)]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    assert bind['sources'][str(Path(__file__).resolve().relative_to(ROOT))]==sha(__file__)
    assert {'cache/valid_targets.npz','cache/chain_folds.npz'}.issubset(bind['data'])
    for rel,h in bind['sources'].items():pin(path_for(rel),h)
    pin(a.bindings,a.bindings_sha256)
    assert bind['sources']['scripts/fold_eval.py']==METRIC_SHA
    for rel,rec in bind['data'].items():
        p=path_for(rel);assert p.stat().st_size==rec['bytes'];pin(p,rec['sha256'])
    allocation=bind['root_allocation'];pin(path_for(allocation['path']),allocation['sha256'])
    done=json.loads(path_for(allocation['path']).read_text())
    assert allocation['required_field']=='status' and allocation['required_value']==COMPLETION
    assert done['status']==COMPLETION and done['fits']==8 and done['OOF_jobs']==4
    assert done['metric_calls']==0 and done['retries']==0 and done['failure'] is None
    assert len(done['jobs'])==12
    expected_jobs={(f'eb_{arm}_s3_parent_f{fold}','parent')
                   for arm in ('candidate','control') for fold in (0,1)}
    expected_jobs|={(f'eb_{arm}_s3_h2_f{fold}',stage)
                    for arm in ('candidate','control') for fold in (0,1) for stage in ('H2','OOF')}
    assert {(j['name'],j['stage']) for j in done['jobs']}==expected_jobs
    assert all(j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[] for j in done['jobs'])
    # New prediction identities are checked before any metric or target read.
    arrays={};receipts={}
    for fold in (0,1):
        for name in MEMBERS:
            base=HERE if name in MEMBERS[:2] else CONT if name==MEMBERS[2] else OLD
            p=base/'cache'/f'pred_{name}_f{fold}_valid.npy';rp=p.with_suffix('.inputs.json')
            rec=bind['predictions'][f'{name}_f{fold}']
            assert rec['path']==str(p.relative_to(ROOT)) and rec['receipt']==str(rp.relative_to(ROOT))
            pin(p,rec['sha256']);pin(rp,rec['receipt_sha256'])
            r=json.loads(rp.read_text())
            assert r['run']==f'{name}_f{fold}' and r['fold']==fold and r['epoch']==1
            assert r['prediction_sha256']==rec['sha256'] and r['checkpoint_sha256']==rec['checkpoint_sha256']
            assert Path(r['checkpoint']).name=='model_ep1.pt' and Path(r['checkpoint']).parent.name==f'{name}_f{fold}'
            if name in MEMBERS[:2]:
                assert r['script_sha256']==bind['sources'][str((HERE/'predict_fixed.py').relative_to(ROOT))]
            receipts[name,fold]=r
            arr=np.load(p,mmap_mode='r',allow_pickle=False)
            assert arr.shape==(1873,20000,2) and arr.dtype==np.float32
            arrays[name,fold]=arr
    er=bind['existing_eda'];ep=path_for(er['path']);erp=path_for(er['receipt'])
    assert ep==EDA/'extraction-01/scored-rows.npz' and erp==EDA/'extraction-01/receipt.json'
    pin(ep,er['sha256']);pin(erp,er['receipt_sha256'])
    assert json.loads(erp.read_text())['outputs'][ep.name]['sha256']==er['sha256']
    out.mkdir();started=time.monotonic()
    provenance=dict(utc=datetime.now(timezone.utc).isoformat(),bindings_sha256=a.bindings_sha256,
        source_sha256=sha(__file__),input_sha256=tracked,recipes=bind['recipes'])
    write(out/'started.json',provenance)
    try:
        with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:
            y,mask,rg,classes=z['y'],z['mask'],z['row_group'],z['market_class']
        assert y.shape==(1873,20000,2) and mask.shape==(1873,20000) and mask.dtype==bool
        assert len(np.unique(rg))==1873
        with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
            v=z['split']=='valid'
            info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v],strict=True)}
            trainchains=set(z['chain'][z['split']=='train'].tolist())
        assert len(info)==len(rg) and set(info)==set(rg.tolist())
        for fold,chain,pure in info.values():
            if pure:assert chain not in trainchains
        selected=[(f,i) for f in (0,1) for i,g in enumerate(rg) if info[int(g)][0]==f and info[int(g)][2]]
        assert len(selected)==860
        order_sha=hashlib.sha256(rg.tobytes()).hexdigest()
        for (name,f),r in receipts.items():
            rows=[i for i,g in enumerate(rg) if info[int(g)][0]==f]
            assert r['source_row_group_order_sha256']==order_sha and r['rows']==rows and r['row_groups']==rg[rows].tolist()
            assert r['chain_folds_sha256']==bind['data']['cache/chain_folds.npz']['sha256']
        # Existing EDA only supplies causal age and keys; labels and baseline predictions
        # below come independently from original bound targets and prediction caches.
        with np.load(ep,allow_pickle=False) as z:
            em=z['metadata'];assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
            fi=z['feature_names'].tolist().index('log_since_i0px')
            age=np.rint(np.expm1(z['X'][:,fi].astype(np.float64))).astype(np.int32)
        assert em.shape==(1614484,6) and age.shape==(1614484,)
        changes=np.flatnonzero(np.any(em[1:, [0,4]]!=em[:-1,[0,4]],axis=1))+1
        cuts=np.r_[0,changes,len(em)]
        mapping={(int(em[a,4]),int(em[a,0])):(int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])}
        assert len(mapping)==860 and set(mapping)==set(selected)
        keys=np.asarray(sorted({(f,info[int(rg[i])][1],int(classes[i])) for f,i in selected}),np.int64)
        assert len(keys)==250 and len(np.unique(keys[:,1]))==250
        ki={tuple(k):j for j,k in enumerate(keys)}
        chain_stats=np.zeros((250,len(SCOPES),len(RECIPES),2,6),np.float64)
        counts=np.zeros((250,len(SCOPES)),np.int64)
        seq_stats=[];positions=[];sequence_meta=[];total=0
        for f,i in selected:
            chain=info[int(rg[i])][1];cl=int(classes[i]);j=ki[f,chain,cl]
            steps=np.flatnonzero(mask[i] & (np.arange(20000)>=99))
            a0,a1=mapping[f,i];mm=em[a0:a1];ages=age[a0:a1]
            assert len(mm)==len(steps) and np.array_equal(mm[:,2],steps)
            assert np.all(mm[:,0]==i) and np.all(mm[:,1]==rg[i]) and np.all(mm[:,3]==chain) and np.all(mm[:,4]==f) and np.all(mm[:,5]==cl)
            yi=y[i,steps];assert np.isfinite(yi).all()
            parts={name:arrays[name,f][i,steps] for name in MEMBERS}
            assert all(np.isfinite(p).all() for p in parts.values())
            pp=[blend([parts[name] for name in names]) for names in RECIPES.values()]
            keep=[np.ones(len(steps),bool),(cl==3)&(ages>=64)]
            for s,k in enumerate(keep):
                counts[j,s]+=int(k.sum())
                for m,p in enumerate(pp):chain_stats[j,s,m]+=sufficient(yi[k],p[k])
            seq_stats.append(np.stack([sufficient(yi,p) for p in pp]));positions.append(i)
            sequence_meta.append((f,chain,cl));total+=len(steps)
        assert total==1614484 and counts[:,0].sum()==total
        full=metric(chain_stats[:,0].sum(0))
        assert abs(full['WP'][2]-BASELINE_WP)<2e-10
        np.testing.assert_allclose(full['rho'][2],BASELINE_RHO,rtol=0,atol=2e-10)
        # A separate frozen implementation checks overall and each outer-fold score.
        spec=importlib.util.spec_from_file_location('canonical_fold_eval',ROOT/'scripts/fold_eval.py')
        fe=importlib.util.module_from_spec(spec);spec.loader.exec_module(fe)
        for selector in (np.ones(250,bool),keys[:,0]==0,keys[:,0]==1):
            ss=chain_stats[selector,0].sum(0)
            for m in range(len(RECIPES)):assert abs(float(metric(ss[m])['WP'])-fe.pooled(ss[m]))<1e-12
        result=describe(chain_stats,counts,keys)
        result.update(status='COMPLETE_FIXED_ENSEMBLE_BACKBONE_COMPARISONS',provenance=provenance,
            sequences=860,chains=250,scored_rows=total,
            baseline_reproduction=dict(expected_WP=BASELINE_WP,observed_WP=float(full['WP'][2]),absolute_error=abs(float(full['WP'][2])-BASELINE_WP)),
            scope='Previously exposed OOF; one paired seed and fixed endpoints. Primary loss-alignment candidate versus fresh matched control; operational contrast versus old student4. Conditional fixed-model intervals are not seed robustness or external confirmation. No native runtime/public qualification or automatic promotion.',
            standalone_queries=0,blend_search=0,raw_reextractions=0,seconds=time.monotonic()-started,
            peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        for p,old in input_stats.items():
            st=Path(p).stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==old,('input changed',p)
        assert sha(a.bindings)==a.bindings_sha256 and sha(__file__)==provenance['source_sha256']
        result['identity_checks']='All inputsSHA256 before scoring; device/inode/size/mtime unchanged after; bindings/source SHA unchanged.'
        np.savez_compressed(out/'sufficient.npz',chain_stats=chain_stats,chain_counts=counts,chain_metadata=keys,
            scope_names=np.array(SCOPES),recipe_names=np.array(list(RECIPES)),sequence_stats=np.asarray(seq_stats),
            sequence_metadata=np.asarray(sequence_meta),positions=np.asarray(positions),row_groups=rg[positions])
        write(out/'report.json',result)
        write(out/'receipt.json',dict(status=result['status'],**provenance,
            outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()}))
        print(json.dumps(clean(dict(status=result['status'],primary=result['comparisons']['primary']['scopes']['all'],operational=result['comparisons']['operational']['scopes']['all'],seconds=result['seconds']))))
    except Exception as exc:
        write(out/'failure.json',dict(exception_type=type(exc).__name__,message=str(exc),provenance=provenance))
        raise


if __name__=='__main__':main()
