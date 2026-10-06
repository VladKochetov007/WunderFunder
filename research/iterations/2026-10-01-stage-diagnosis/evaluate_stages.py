"""Fixed parent-to-H2 attribution; no fit, epoch/weight search or promotion gate.

Only main() reads bound study artifacts. --self-test is generated NumPy only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
B=ROOT/'research/iterations/2026-10-01-ensemble-backbone'
OLD=ROOT/'research/iterations/2026-10-01-overnight'
EDA=ROOT/'research/iterations/2026-10-01-error-analysis'
B_SHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
BP=B/'evaluate.py'
assert hashlib.sha256(BP.read_bytes()).hexdigest()==B_SHA
spec=importlib.util.spec_from_file_location('frozen_backbone_grader',BP)
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
MODELS=('parent_candidate4','parent_control4','H2_candidate4','H2_control4')
PARENTS=('eb_candidate_s3_parent','eb_control_s3_parent')
ANCHORS=('on_ctl128s3','on_ctl128s4','on_ctlc500')
COEFFICIENTS={
    'parent_candidate_minus_control':(1,-1,0,0),
    'H2_candidate_minus_control':(0,0,1,-1),
    'interaction_H2_delta_minus_parent_delta':(-1,1,1,-1),
    'candidate_H2_minus_parent':(-1,0,1,0),
    'control_H2_minus_parent':(0,-1,0,1),
}


def summarize(stats,counts,draws):
    """One shared resampling of four fixed profiles; contrast CIs, not CI subtraction."""
    point=E.metric(stats.sum(0))
    sample=E.metric((draws@stats.reshape(len(stats),-1)).reshape(len(draws),4,2,6))
    result={'scored_rows':int(counts.sum()),'chains_with_scored_rows':int((counts>0).sum()),
            'models':{name:{k:v[j] for k,v in point.items()} for j,name in enumerate(MODELS)},'contrasts':{}}
    for name,weights in COEFFICIENTS.items():
        c=np.asarray(weights);used=c!=0
        rho=np.einsum('m,bmt->bt',c,sample['rho']);wp=np.einsum('m,bm->b',c,sample['WP'])
        valid=sample['valid'][:,used].all(1)
        result['contrasts'][name]={'delta_WP':float(c@point['WP']),'delta_rho':c@point['rho'],
            'paired_chain_CI':E.interval(wp,valid.all(-1)),
            'paired_target_CI':[E.interval(rho[:,t],valid[:,t]) for t in (0,1)]}
    return result


def equal_report(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k,v in a.items():equal_report(v,b[k])
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        for x,y in zip(a,b,strict=True):equal_report(x,y)
    elif a is None or isinstance(a,(str,bool)):assert a==b
    else:np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)


def self_test():
    # Independent centered-correlation oracle verifies signs and paired finite
    # interaction intervals; it never imports or reads study arrays.
    rng=np.random.default_rng(51);ys=[];ps=[];stats=[]
    for n in (13,19):
        y=rng.normal(size=(n,2))*2
        p=np.stack([.8*y+rng.normal(size=(n,2))*s for s in (.8,1.2,.7,1.1)])
        ys.append(y);ps.append(p);stats.append(np.stack([E.sufficient(y,v) for v in p]))
    stats=np.array(stats);counts=np.array([13,19]);draws=np.array([[2,0],[0,2],[1,1]])
    def direct(y,p):
        y=np.clip(y,-2,2);p=np.clip(p,-2,2);w=abs(y)
        yc=y-(w*y).sum(0)/w.sum(0);pc=p-(w*p).sum(0)/w.sum(0)
        return (w*yc*pc).sum(0)/np.sqrt((w*yc**2).sum(0)*(w*pc**2).sum(0))
    oracle=[]
    for d in draws:
        rows=np.repeat(np.arange(2),d);y=np.concatenate([ys[i] for i in rows])
        oracle.append(np.stack([direct(y,np.concatenate([ps[i][j] for i in rows])) for j in range(4)]))
    oracle=np.asarray(oracle);r=summarize(stats,counts,draws)
    for name,c in COEFFICIENTS.items():
        values=np.einsum('m,bmt->bt',np.array(c),oracle)
        q=r['contrasts'][name]
        np.testing.assert_allclose(q['delta_rho'],values[2],atol=1e-12,rtol=0)
        np.testing.assert_allclose(q['paired_chain_CI']['interval95'],np.quantile(values.mean(-1),[.025,.975]),atol=1e-12,rtol=0)
        for t in (0,1):np.testing.assert_allclose(q['paired_target_CI'][t]['interval95'],np.quantile(values[:,t],[.025,.975]),atol=1e-12,rtol=0)
    for j,name in enumerate(MODELS):np.testing.assert_allclose(r['models'][name]['rho'],oracle[2,j],atol=1e-12,rtol=0)
    stats[:,2:]=stats[:,:2]
    q=summarize(stats,counts,draws)['contrasts']['interaction_H2_delta_minus_parent_delta']
    np.testing.assert_allclose(q['paired_chain_CI']['interval95'],[0,0],atol=1e-15,rtol=0)
    print(json.dumps({'status':'PASS_GENERATED_DIRECT_STAGE_CONTRASTS','source_sha256':E.sha(__file__),
                      'checks':'four absolute profiles, five point/target/paired-CI contrasts, zero-stage-change interaction',
                      'actual_arrays_loaded':0,'model_inference':0}))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    assert args.output_dir
    out=args.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists()
    tracked={};states={}
    def pin(p,h):
        p=Path(p);assert isinstance(h,str) and len(h)==64 and E.sha(p)==h,('hash',str(p))
        tracked[str(p)]=h;s=p.stat();states[str(p)]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    apath=HERE/'allocation.json';ah=E.sha(apath);pin(apath,ah);allocation=json.loads(apath.read_text())
    assert allocation['execution_authorized'] is True and allocation['success_status']=='FOUR_PARENT_OOF'
    sources=allocation['sources'];assert sources[str(Path(__file__).relative_to(ROOT))]==E.sha(__file__)
    assert sources[str(BP.relative_to(ROOT))]==B_SHA
    for rel,h in sources.items():pin(E.path_for(rel),h)
    bound={}
    for record in allocation['input_bindings']:
        p=E.path_for(record['path']);pin(p,record['sha256'])
        for rec in json.loads(p.read_text())['records']:
            p=Path(rec['path']);assert p.is_absolute()
            st=p.stat();assert st.st_size==rec['size']
            pin(p,rec['sha256']);assert str(p) not in bound;bound[str(p)]=rec
    def required(p):assert str(p) in bound,('unbound input',str(p));return bound[str(p)]
    donepath=HERE/'execution-01/result.json';pin(donepath,E.sha(donepath));done=json.loads(donepath.read_text())
    assert allocation['execution_directory']==str(donepath.parent.relative_to(ROOT))
    assert done['status']=='FOUR_PARENT_OOF' and done['allocation_sha256']==ah
    assert done['runner_sha256']==sources[str((HERE/'run_queue.py').relative_to(ROOT))]
    assert done['failure'] is None and done['fits']==0 and done['OOF_jobs']==4 and done['metric_calls']==0 and done['retries']==0
    expected={f'{name}_f{f}' for name in PARENTS for f in (0,1)}
    assert len(done['jobs'])==len(allocation['jobs'])==4
    assert {j['name'] for j in done['jobs']}=={j['name'] for j in allocation['jobs']}==expected
    jobs={j['name']:j for j in allocation['jobs']};workers={}
    for j in done['jobs']:
        assert j['stage']=='OOF' and j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[]
        wr=j['worker_result'];wp=E.path_for(wr['path']);pin(wp,wr['sha256']);w=json.loads(wp.read_text())
        assert w['status']=='COMPLETE' and w['allocation_sha256']==ah and w['name']==j['name'] and w['stage']=='OOF'
        assert w['returncode']==0 and w['argv']==jobs[j['name']]['argv']
        assert set(w['outputs'])==set(jobs[j['name']]['required_outputs'])
        workers[j['name']]=w
    bindpath=B/'evaluation-bindings-01.json';required(bindpath);bb=json.loads(bindpath.read_text())
    reportpath=B/'evaluation-01/report.json';statpath=B/'evaluation-01/sufficient.npz';receiptpath=B/'evaluation-01/receipt.json'
    for p in (reportpath,statpath,receiptpath):required(p)
    oldreceipt=json.loads(receiptpath.read_text());oldreport=json.loads(reportpath.read_text())
    assert oldreceipt['status']==oldreport['status']=='COMPLETE_FIXED_ENSEMBLE_BACKBONE_COMPARISONS'
    assert oldreport['provenance']['source_sha256']==B_SHA
    assert oldreport['provenance']['bindings_sha256']==tracked[str(bindpath)]
    for p in (reportpath,statpath):assert oldreceipt['outputs'][p.name]['sha256']==tracked[str(p)]
    arrays={};receipts={}
    for f in (0,1):
        for name in (*PARENTS,*ANCHORS):
            new=name in PARENTS;run=f'{name}_f{f}';epoch=3 if new else 1
            p=(HERE if new else OLD)/'cache'/f'pred_{run}_valid.npy';rp=p.with_suffix('.inputs.json')
            if new:
                for q in (p,rp):
                    proof=workers[run]['outputs'][str(q.relative_to(ROOT))]
                    assert q.stat().st_size==proof['bytes'];pin(q,proof['sha256'])
            else:
                for q in (p,rp):required(q)
                rec=bb['predictions'][run]
                assert rec['path']==str(p.relative_to(ROOT)) and rec['receipt']==str(rp.relative_to(ROOT))
                assert rec['sha256']==tracked[str(p)] and rec['receipt_sha256']==tracked[str(rp)]
            r=json.loads(rp.read_text())
            assert r['run']==run and r['fold']==f and r['epoch']==epoch and r['prediction_sha256']==tracked[str(p)]
            assert Path(r['checkpoint']).name==f'model_ep{epoch}.pt' and Path(r['checkpoint']).parent.name==run
            if new:
                cp=ROOT/f'artifacts/runs/{run}/model_ep3.pt';required(cp)
                assert r['checkpoint_sha256']==tracked[str(cp)] and r['metrics_computed']==0
                assert r['script_sha256']==sources[str((HERE/'predict_fixed.py').relative_to(ROOT))]
                wrs=[Path(q) for q in bound if Path(q).name==f'{run}.parent.worker-result.json'];assert len(wrs)==1
                parent_worker=json.loads(wrs[0].read_text())
                assert parent_worker['status']=='COMPLETE' and parent_worker['name']==run and parent_worker['stage']=='parent'
                assert parent_worker['outputs'][str(cp.relative_to(ROOT))]['sha256']==r['checkpoint_sha256']
                assert bb['checkpoint_validation_receipts'][run]['checkpoint']['sha256']==r['checkpoint_sha256']
            else:assert r['checkpoint_sha256']==bb['predictions'][run]['checkpoint_sha256']
            receipts[name,f]=r
            z=np.load(p,mmap_mode='r',allow_pickle=False);assert z.shape==(1873,20000,2) and z.dtype==np.float32
            arrays[name,f]=z
    # Nothing above computed a score; all sources/input and four new output bytes
    # are pinned before target arrays/statistics are read below.
    out.mkdir();started=time.monotonic();E.write(out/'started.json',{'allocation_sha256':ah,'source_sha256':E.sha(__file__),'input_sha256':tracked})
    try:
        analyze(out,started,allocation,ah,tracked,states,bound,arrays,receipts,statpath,oldreport)
    except BaseException as exc:
        E.write(out/'failure.json',{'exception':repr(exc),'status':'FAILED_NO_RETRY'});raise


def analyze(out,started,allocation,ah,tracked,states,bound,arrays,receipts,statpath,oldreport):
    tp=ROOT/'cache/valid_targets.npz';fp=ROOT/'cache/chain_folds.npz'
    assert str(tp) in bound and str(fp) in bound
    with np.load(statpath,allow_pickle=False) as z:
        hs=z['chain_stats'];counts=z['chain_counts'];keys=z['chain_metadata']
        assert z['scope_names'].tolist()==list(E.SCOPES) and z['recipe_names'].tolist()==list(E.RECIPES)
    assert hs.shape==(250,2,3,2,6) and counts.shape==(250,2) and keys.shape==(250,3)
    assert len(np.unique(keys[:,1]))==250
    reproduced=E.clean(E.describe(hs,counts,keys))
    equal_report(reproduced['comparisons'],oldreport['comparisons'])
    equal_report(reproduced['uncertainty'],oldreport['uncertainty'])
    with np.load(tp,allow_pickle=False) as z:y,mask,rg,classes=z['y'],z['mask'],z['row_group'],z['market_class']
    assert y.shape==(1873,20000,2) and mask.shape==(1873,20000) and mask.dtype==bool and len(np.unique(rg))==1873
    with np.load(fp,allow_pickle=False) as z:
        v=z['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v],strict=True)}
        trainchains=set(z['chain'][z['split']=='train'].tolist())
    assert len(info)==1873 and set(info)==set(rg.tolist())
    for f,c,pure in info.values():
        if pure:assert c not in trainchains
    selected=[(f,i) for f in (0,1) for i,g in enumerate(rg) if info[int(g)][0]==f and info[int(g)][2]]
    assert len(selected)==860
    derived=np.asarray(sorted({(f,info[int(rg[i])][1],int(classes[i])) for f,i in selected}),np.int64)
    np.testing.assert_array_equal(derived,keys)
    order=hashlib.sha256(rg.tobytes()).hexdigest()
    for (_,f),r in receipts.items():
        rows=[i for i,g in enumerate(rg) if info[int(g)][0]==f]
        assert r['source_row_group_order_sha256']==order and r['rows']==rows and r['row_groups']==rg[rows].tolist()
        assert r['chain_folds_sha256']==tracked[str(fp)]
    ep=EDA/'extraction-01/scored-rows.npz';erp=EDA/'extraction-01/receipt.json'
    assert str(ep) in bound and str(erp) in bound
    assert json.loads(erp.read_text())['outputs'][ep.name]['sha256']==tracked[str(ep)]
    with np.load(ep,allow_pickle=False) as z:
        em=z['metadata'];assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
        age=np.rint(np.expm1(z['X'][:,z['feature_names'].tolist().index('log_since_i0px')].astype(np.float64))).astype(np.int32)
    assert em.shape==(1614484,6) and age.shape==(1614484,)
    cuts=np.r_[0,np.flatnonzero(np.any(em[1:,[0,4]]!=em[:-1,[0,4]],axis=1))+1,len(em)]
    mapping={(int(em[a,4]),int(em[a,0])):(int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])}
    assert len(mapping)==860 and set(mapping)==set(selected)
    ki={tuple(k):j for j,k in enumerate(keys)};ps=np.zeros((250,2,2,2,6),float);pc=np.zeros((250,2),np.int64)
    for f,i in selected:
        c=info[int(rg[i])][1];cl=int(classes[i]);j=ki[f,c,cl]
        steps=np.flatnonzero(mask[i]&(np.arange(20000)>=99));a,b=mapping[f,i];mm=em[a:b];ages=age[a:b]
        assert len(mm)==len(steps) and np.array_equal(mm[:,2],steps)
        assert np.all(mm[:,0]==i) and np.all(mm[:,1]==rg[i]) and np.all(mm[:,3]==c) and np.all(mm[:,4]==f) and np.all(mm[:,5]==cl)
        yy=y[i,steps];parts={name:arrays[name,f][i,steps] for name in (*PARENTS,*ANCHORS)}
        assert np.isfinite(yy).all() and all(np.isfinite(v).all() for v in parts.values())
        pp=[E.blend([parts[name],*[parts[k] for k in ANCHORS]]) for name in PARENTS]
        for s,keep in enumerate((np.ones(len(steps),bool),(cl==3)&(ages>=64))):
            pc[j,s]+=int(keep.sum())
            for arm,p in enumerate(pp):ps[j,s,arm]+=E.sufficient(yy[keep],p[keep])
    np.testing.assert_array_equal(pc,counts);assert pc[:,0].sum()==1614484
    # Labels, masks, target weighting and chain/slice membership must match H2.
    np.testing.assert_allclose(np.take(ps[:,:,0],[0,1,3],axis=-1),np.take(hs[:,:,0],[0,1,3],axis=-1),rtol=0,atol=2e-8)
    stats=np.concatenate((ps,hs[:,:,:2]),axis=2)
    draws,strata=E.resamples(keys);scopes={}
    for j,name in enumerate(E.SCOPES):
        scopes[name]=summarize(stats[:,j],counts[:,j],draws)
        scopes[name]['folds']={str(f):summarize(stats[keys[:,0]==f,j],counts[keys[:,0]==f,j],draws[:,keys[:,0]==f]) for f in (0,1)}
    result={'status':'COMPLETE_FIXED_STAGE_DIAGNOSIS','scopes':scopes,'models':MODELS,'contrasts':COEFFICIENTS,
        'sequences':860,'chains':250,'scored_rows':1614484,'H2_report_reproduction':'Both saved B comparisons, scopes, folds, targets and bootstrap intervals agree within2e-12.',
        'uncertainty':{'draws':E.BOOTSTRAPS,'seed':E.SEED,'strata':strata,'scope':'Same paired whole-chain draws for all four profiles. Recompute each WP then form interactions within each draw. Conditional on fixed models; marginal intervals, no seed/selection/external-transfer uncertainty.'},
        'interpretation':'Post-readout diagnostic on exposed folds. Parent and H2 were fixed before these four inference jobs. Stage interaction is descriptive, not isolated causal attribution to one H2 implementation component. Absolute four-ensemble table supports operational context; no recipe promotion or new gate.',
        'fits':0,'parent_OOF_jobs':4,'raw_reextractions':0,'standalone_queries':0,'weight_or_epoch_search':0,
        'allocation_sha256':ah,'source_sha256':E.sha(__file__),'numerical_module_sha256':B_SHA,'input_sha256':tracked,
        'seconds':time.monotonic()-started,'peak_RSS_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for p,previous in states.items():
        s=Path(p).stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==previous,('input changed',p)
    for rel,h in allocation['sources'].items():assert E.sha(E.path_for(rel))==h
    assert E.sha(HERE/'allocation.json')==ah
    np.savez_compressed(out/'sufficient.npz',chain_stats=stats,chain_counts=counts,chain_metadata=keys,scope_names=np.array(E.SCOPES),model_names=np.array(MODELS))
    E.write(out/'report.json',result)
    E.write(out/'receipt.json',{'status':result['status'],'source_sha256':E.sha(__file__),'allocation_sha256':ah,
        'outputs':{p.name:{'sha256':E.sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps(E.clean({'status':result['status'],'all':scopes['all'],'seconds':result['seconds']})))


if __name__=='__main__':main()
