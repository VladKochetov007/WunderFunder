"""One fixed late categorical residual screen; four fits, two paired OOF passes."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
B=ROOT/'research/iterations/2026-10-01-ensemble-backbone'; OLD=ROOT/'research/iterations/2026-10-01-overnight'; EDA=ROOT/'research/iterations/2026-10-01-error-analysis'
BP=B/'evaluate.py'; BSHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
assert hashlib.sha256(BP.read_bytes()).hexdigest()==BSHA
spec=importlib.util.spec_from_file_location('atom_frozen_metric',BP); E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
VARIANTS=('candidate','control','base'); MODELS=('candidate','control','incumbent'); ANCHORS=('on_ctl128s3','on_ctl128s4','on_ctlc500')
PAIRS={'primary_candidate_minus_incumbent':(0,2),'candidate_minus_CE0_control':(0,1),'CE0_control_minus_incumbent':(1,2)}

def summarize(stats,counts,draws):
    point=E.metric(stats.sum(0)); sample=E.metric((draws@stats.reshape(len(stats),-1)).reshape(len(draws),3,2,6))
    out={'scored_rows':int(counts.sum()),'chains_with_scored_rows':int((counts>0).sum()),'models':{n:{k:v[j] for k,v in point.items()} for j,n in enumerate(MODELS)},'contrasts':{}}
    for name,(c,r) in PAIRS.items():
        valid=sample['valid'][:,c]&sample['valid'][:,r]; rho=sample['rho'][:,c]-sample['rho'][:,r]
        out['contrasts'][name]={'delta_WP':float(point['WP'][c]-point['WP'][r]),'delta_rho':point['rho'][c]-point['rho'][r],
            'paired_chain_CI':E.interval(sample['WP'][:,c]-sample['WP'][:,r],valid.all(-1)),
            'paired_target_CI':[E.interval(rho[:,t],valid[:,t]) for t in (0,1)]}
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    out=args.output_dir.resolve(); assert out.is_relative_to(HERE) and out!=HERE and not out.exists(); started=time.monotonic(); tracked={}; states={}
    def pin(p,h):
        p=Path(p); assert E.sha(p)==h,('hash',str(p)); tracked[str(p)]=h; s=p.stat(); states[str(p)]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    apath=HERE/'allocation.json'; ah=E.sha(apath); pin(apath,ah); a=json.loads(apath.read_text())
    assert a['execution_authorized'] and a['success_status']=='FOUR_LATE_HEAD_FITS_TWO_OOF'
    assert a['sources'][str(Path(__file__).relative_to(ROOT))]==E.sha(__file__) and a['sources'][str(BP.relative_to(ROOT))]==BSHA
    for rel,h in a['sources'].items():pin(E.path_for(rel),h)
    bound={}
    for rec in a['input_bindings']:
        p=E.path_for(rec['path']); pin(p,rec['sha256'])
        for r in json.loads(p.read_text())['records']:
            p=Path(r['path']); st=p.stat(); assert p.is_absolute() and st.st_size==r['size']; bound[str(p)]=r
            if r.get('sha256') is not None:pin(p,r['sha256'])
            else:
                assert p.name in ('train.parquet','valid.parquet') and str(p.resolve())==r['resolved']
                state=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
                assert state==(r['device'],r['inode'],r['size'],r['mtime_ns']); states[str(p)]=state
    def required(p):assert str(p) in bound,('unbound',str(p)); return p
    dp=HERE/'execution-01/result.json'; pin(dp,E.sha(dp)); d=json.loads(dp.read_text())
    assert a['execution_directory']==str(dp.parent.relative_to(ROOT)) and d['status']=='FOUR_LATE_HEAD_FITS_TWO_OOF'
    assert d['allocation_sha256']==ah and d['failure'] is None and (d['fits'],d['OOF_jobs'],d['metric_calls'],d['retries'])==(4,2,0,0)
    assert d['runner_sha256']==a['sources'][str((HERE/'run_queue.py').relative_to(ROOT))]
    jobs={j['name']:j for j in a['jobs']}; outputs={}
    assert len(d['jobs'])==len(jobs)==6 and {j['name'] for j in d['jobs']}==set(jobs)
    for j in d['jobs']:
        assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[] and j['stage']==jobs[j['name']]['stage']
        wr=j['worker_result']; p=E.path_for(wr['path']); pin(p,wr['sha256']); w=json.loads(p.read_text())
        assert w['status']=='COMPLETE' and w['allocation_sha256']==ah and w['name']==j['name'] and w['stage']==j['stage'] and w['returncode']==0
        assert w['argv']==jobs[j['name']]['argv'] and set(w['outputs'])==set(jobs[j['name']]['required_outputs'])
        for rel,pr in w['outputs'].items():
            p=E.path_for(rel); assert p.stat().st_size==pr['bytes'] and rel not in outputs; pin(p,pr['sha256']); outputs[rel]=pr
    assert sum(j['stage']=='H2' for j in d['jobs'])==4 and sum(j['stage']=='OOF' for j in d['jobs'])==2
    fit_integrity={}
    assert {j['name'] for j in d['jobs'] if j['stage']=='H2'}=={f'lateatom_{arm}_s3_f{f}' for arm in ('candidate','control') for f in (0,1)}
    for f in (0,1):
        fit_integrity[str(f)]={}
        for arm in ('candidate','control'):
            run=f'lateatom_{arm}_s3_f{f}'; p=HERE/'runs'/run/'receipt.json'; assert str(p.relative_to(ROOT)) in outputs; q=json.loads(p.read_text())
            assert q['status']=='COMPLETE_FIXED_EP1' and q['frozen_base_exact'] is True and q['only_residual_trainable'] is True
            cp=HERE/'runs'/run/'model_ep1.pt'; bp=required(ROOT/f'artifacts/runs/cont_sh256_s3h2_f{f}/model_ep1.pt')
            assert q['checkpoint']=={'path':str(cp),'sha256':tracked[str(cp)]} and q['baseline']=={'path':str(bp),'sha256':tracked[str(bp)]}
            assert tracked[str(required(Path(q['vocab_path']))) ]==q['vocab_sha256'] and len(q['sampler_order_sha256'])==2
            fit_integrity[str(f)][arm]=q
        c,r=fit_integrity[str(f)]['candidate'],fit_integrity[str(f)]['control']
        for field in ('initial_residual_sha256','sampler_order_sha256','vocab_sha256'):assert c[field]==r[field]
    # These exact scored rows retain the original y, mask/warmup keys and age.
    ep=required(EDA/'extraction-01/scored-rows.npz'); er=required(EDA/'extraction-01/receipt.json')
    assert json.loads(er.read_text())['outputs'][ep.name]['sha256']==tracked[str(ep)]
    sp=required(B/'evaluation-01/sufficient.npz')
    with np.load(sp,allow_pickle=False) as z:
        old=z['chain_stats']; counts=z['chain_counts']; keys=z['chain_metadata']
        assert z['recipe_names'].tolist()==list(E.RECIPES) and z['scope_names'].tolist()==list(E.SCOPES)
    assert old.shape==(250,2,3,2,6) and keys.shape==(250,3) and counts.shape==(250,2) and counts[:,0].sum()==1614484
    with np.load(ep,allow_pickle=False) as z:
        y=z['y']; meta=z['metadata']; assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
        base_reference=z['predictions'][:,z['prediction_names'].tolist().index('shallow_seed3'),:]
        age=np.rint(np.expm1(z['X'][:,z['feature_names'].tolist().index('log_since_i0px')].astype(np.float64))).astype(np.int32)
    assert meta.shape==(1614484,6) and y.shape==(1614484,2) and np.isfinite(y).all()
    ki={tuple(k):i for i,k in enumerate(keys)}; stats=np.zeros((250,2,3,2,6)); checkcounts=np.zeros_like(counts); seqcount=0
    out.mkdir(); E.write(out/'started.json',{'source_sha256':E.sha(__file__),'allocation_sha256':ah,'input_sha256':tracked})
    for f in (0,1):
        pp=HERE/'predictions-01'/f'fold{f}.npz'; rp=pp.with_suffix('.json')
        assert str(pp.relative_to(ROOT)) in outputs and str(rp.relative_to(ROOT)) in outputs; r=json.loads(rp.read_text())
        assert r['status']=='COMPLETE_OOF' and r['fold']==f and r['variants']==list(VARIANTS) and r['model_inputs_target_free'] is True
        assert r['outputs'][pp.name]['sha256']==tracked[str(pp)] and r['outputs'][pp.name]['bytes']==pp.stat().st_size
        assert set(r['checkpoints'])==set(VARIANTS)
        for arm,rec in r['checkpoints'].items():
            cp=Path(rec['path']); assert tracked[str(cp)]==rec['sha256'] and cp.name=='model_ep1.pt'
            if arm=='base':assert cp==required(ROOT/f'artifacts/runs/cont_sh256_s3h2_f{f}/model_ep1.pt')
            else:assert cp.parent.name==f'lateatom_{arm}_s3_f{f}' and str(cp.relative_to(ROOT)) in outputs
        assert tracked[str(Path(r['vocab_path']))]==r['vocab_sha256']
        for rel,h in r['source_sha256'].items():assert a['sources'][rel]==h
        selected=meta[:,4]==f; mm=meta[selected]; yy=y[selected]; aa=age[selected]
        with np.load(pp,allow_pickle=False) as z:
            pred=z['predictions']; assert z['variants'].tolist()==list(VARIANTS)
            for col,name in enumerate(('cache_position','row_group','step','chain','fold')):np.testing.assert_array_equal(z[name],mm[:,col])
        assert pred.shape==(len(mm),3,2) and pred.dtype==np.float32 and np.isfinite(pred).all() and r['scored_rows']==len(mm)
        np.testing.assert_array_equal(pred[:,2],base_reference[selected])
        anchors=[]
        for name in ANCHORS:
            p=required(OLD/'cache'/f'pred_{name}_f{f}_valid.npy'); rp=required(p.with_suffix('.inputs.json')); ar=json.loads(rp.read_text())
            assert ar['prediction_sha256']==tracked[str(p)] and ar['fold']==f and ar['epoch']==1 and ar['run']==f'{name}_f{f}'
            arr=np.load(p,mmap_mode='r',allow_pickle=False); assert arr.shape==(1873,20000,2) and arr.dtype==np.float32; anchors.append(arr)
        cuts=np.r_[0,np.flatnonzero(mm[1:,0]!=mm[:-1,0])+1,len(mm)]
        assert len(cuts)-1==r['sequences']==(425 if f==0 else 435); seqcount+=len(cuts)-1
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            rows=mm[lo:hi]; pos=int(rows[0,0]); cl=int(rows[0,5]); j=ki[f,int(rows[0,3]),cl]; steps=rows[:,2]
            assert np.all(rows[:,0]==pos) and np.all(rows[:,3:]==rows[0,3:]) and np.all(steps>=99) and np.all(np.diff(steps)>0)
            parts=[v[pos,steps] for v in anchors]; assert all(np.isfinite(v).all() for v in parts)
            blended=[E.blend([pred[lo:hi,m],*parts]) for m in range(3)]
            for s,keep in enumerate((np.ones(hi-lo,bool),(cl==3)&(aa[lo:hi]>=64))):
                checkcounts[j,s]+=int(keep.sum())
                for m,p in enumerate(blended):stats[j,s,m]+=E.sufficient(yy[lo:hi][keep],p[keep])
    assert seqcount==860; np.testing.assert_array_equal(checkcounts,counts)
    for m in range(3):np.testing.assert_allclose(np.take(stats[:,:,m],[0,1,3],axis=-1),np.take(old[:,:,2],[0,1,3],axis=-1),atol=2e-8,rtol=0)
    np.testing.assert_allclose(stats[:,:,2],old[:,:,2],atol=2e-8,rtol=1e-12)
    assert abs(E.metric(stats[:,0,2].sum(0))['WP']-E.BASELINE_WP)<2e-12
    draws,strata=E.resamples(keys); scopes={}
    for s,name in enumerate(E.SCOPES):
        scopes[name]=summarize(stats[:,s],counts[:,s],draws)
        scopes[name]['folds']={str(f):summarize(stats[keys[:,0]==f,s],counts[keys[:,0]==f,s],draws[:,keys[:,0]==f]) for f in (0,1)}
    report={'status':'COMPLETE_FIXED_LATE_ATOM_HEAD_SCREEN','scopes':scopes,'models':MODELS,'contrasts':PAIRS,'sequences':860,'chains':250,'scored_rows':1614484,
        'uncertainty':{'draws':E.BOOTSTRAPS,'seed':E.SEED,'strata':strata,'interpretation':'Shared paired whole-chain draws; each WP pooled across rows before contrasts. Marginal conditional95% intervals, no seed/selection/hidden-test uncertainty.'},
        'interpretation':'One new residual readout on a frozen backbone, identical ordered float32 quarter anchors. Primary candidate−incumbent; CE0 contrasts diagnose categorical supervision including its zero-gain gradient asymmetry. Exposed folds, fixed endpoints, no best-head selection or universal score gate.',
        'fit_integrity':fit_integrity,'baseline_reproduced':E.BASELINE_WP,'new_path_base_raw_predictions':'bit-exact to original E shallow_seed3 cache','fits':4,'paired_inference_jobs':2,'source_sha256':E.sha(__file__),'allocation_sha256':ah,'input_sha256':tracked,'stat_only_data_inputs':{p:r for p,r in bound.items() if r.get('sha256') is None},'seconds':time.monotonic()-started}
    for p,state in states.items():
        st=Path(p).stat(); assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==state,('input changed',p)
    np.savez_compressed(out/'sufficient.npz',chain_stats=stats,chain_counts=counts,chain_metadata=keys,scope_names=np.array(E.SCOPES),model_names=np.array(MODELS))
    E.write(out/'report.json',report); E.write(out/'receipt.json',{'status':report['status'],'source_sha256':E.sha(__file__),'allocation_sha256':ah,'outputs':{p.name:{'sha256':E.sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    print(json.dumps(E.clean({'status':report['status'],'all':scopes['all'],'seconds':report['seconds']})))

if __name__=='__main__':main()
