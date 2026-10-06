"""Two saved seed4 parent OOFs; four fixed profiles, no fits or selection gate."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
SP=HERE/'evaluate_stages.py'; SSHA='d621c914bb514824b81fe650e14a1ea97cfd7da08abd1e4277543d1740b6bbfd'
assert hashlib.sha256(SP.read_bytes()).hexdigest()==SSHA
spec=importlib.util.spec_from_file_location('fixed_stage_math',SP); S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
E=S.E; C=ROOT/'research/iterations/2026-10-01-continuation/student-seed4'
S.MODELS=('s3_parent4','s3_H2_4','s4_parent4','s4_H2_4')
S.COEFFICIENTS={'s3_parent_minus_H2':(1,-1,0,0),'s4_parent_minus_H2':(0,0,1,-1),
    'mean_two_seed_parent_minus_H2':(.5,-.5,.5,-.5),'s4_parent_minus_incumbent_s3_H2':(0,-1,1,0)}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    out=args.output_dir.resolve(); assert out.is_relative_to(HERE) and out!=HERE and not out.exists()
    start=time.monotonic(); tracked={}; states={}
    def pin(p,h):
        p=Path(p); assert E.sha(p)==h,('hash',str(p)); tracked[str(p)]=h
        st=p.stat(); states[str(p)]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    alloc=HERE/'allocation-02.json'; ah=E.sha(alloc); pin(alloc,ah); a=json.loads(alloc.read_text())
    assert a['execution_authorized'] and a['success_status']=='TWO_SEED4_PARENT_OOF'
    assert a['sources'][str(SP.relative_to(ROOT))]==SSHA
    assert a['sources'][str(Path(__file__).relative_to(ROOT))]==E.sha(__file__)
    for rel,h in a['sources'].items():pin(E.path_for(rel),h)
    bound={}
    for rec in a['input_bindings']:
        p=E.path_for(rec['path']); pin(p,rec['sha256'])
        for r in json.loads(p.read_text())['records']:
            p=Path(r['path']); assert p.is_absolute() and p.stat().st_size==r['size']; pin(p,r['sha256']); bound[str(p)]=r
    def read(p):
        assert str(p) in bound,('unbound',str(p)); return json.loads(p.read_text())
    dp=HERE/'execution-02/result.json'; pin(dp,E.sha(dp)); d=json.loads(dp.read_text())
    assert a['execution_directory']==str(dp.parent.relative_to(ROOT))
    assert d['status']=='TWO_SEED4_PARENT_OOF' and d['allocation_sha256']==ah and d['failure'] is None
    assert (d['fits'],d['OOF_jobs'],d['metric_calls'],d['retries'])==(0,2,0,0)
    assert d['runner_sha256']==a['sources'][str((HERE/'run_queue.py').relative_to(ROOT))]
    jobs={j['name']:j for j in a['jobs']}; workers={}; names={f'cont_sh256_s4_f{f}' for f in (0,1)}
    assert len(d['jobs'])==len(jobs)==2 and set(jobs)=={j['name'] for j in d['jobs']}==names
    for j in d['jobs']:
        assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[] and j['stage']=='OOF'
        wr=j['worker_result']; p=E.path_for(wr['path']); pin(p,wr['sha256']); w=json.loads(p.read_text())
        assert w['status']=='COMPLETE' and w['allocation_sha256']==ah and w['name']==j['name'] and w['stage']=='OOF'
        assert w['returncode']==0 and w['argv']==jobs[j['name']]['argv'] and set(w['outputs'])==set(jobs[j['name']]['required_outputs'])
        workers[j['name']]=w
    lineage=read(C/'fit-inspection-recovery.json'); arrays={}; receipts={}
    for f in (0,1):
        for name in ('cont_sh256_s4',*S.ANCHORS):
            new=name=='cont_sh256_s4'; run=f'{name}_f{f}'; p=(HERE if new else S.OLD)/'cache'/f'pred_{run}_valid.npy'; rp=p.with_suffix('.inputs.json')
            if new:
                for q in (p,rp):
                    pr=workers[run]['outputs'][str(q.relative_to(ROOT))]; assert q.stat().st_size==pr['bytes']; pin(q,pr['sha256'])
                r=json.loads(rp.read_text()); cp=ROOT/f'artifacts/runs/{run}/model_ep3.pt'; assert str(cp) in bound
                assert r['checkpoint_sha256']==tracked[str(cp)]==lineage['all_new_run_files'][str(cp.relative_to(ROOT))]['sha256']
                assert r['script_sha256']==a['sources'][str((HERE/'predict_fixed.py').relative_to(ROOT))] and r['metrics_computed']==0
            else:
                assert str(p) in bound; r=read(rp)
            assert r['run']==run and r['fold']==f and r['epoch']==(3 if new else 1) and r['prediction_sha256']==tracked[str(p)]
            assert Path(r['checkpoint']).parent.name==run and Path(r['checkpoint']).name==f"model_ep{r['epoch']}.pt"
            z=np.load(p,mmap_mode='r',allow_pickle=False); assert z.shape==(1873,20000,2) and z.dtype==np.float32
            arrays[name,f]=z; receipts[name,f]=r
    # All immutable source/input/new-output hashes precede loading numerical inputs.
    sp=HERE/'evaluation-01/sufficient.npz'; hp=C/'evaluation-v2-01/sufficient.npz'
    tp=ROOT/'cache/valid_targets.npz'; fp=ROOT/'cache/chain_folds.npz'
    for p in (sp,hp,tp,fp):assert str(p) in bound
    prior=read(HERE/'evaluation-01/report.json')
    with np.load(sp,allow_pickle=False) as z:
        ss=z['chain_stats']; counts=z['chain_counts'][:,0]; keys=z['chain_metadata']
        assert z['model_names'].tolist()==['parent_candidate4','parent_control4','H2_candidate4','H2_control4'] and z['scope_names'].tolist()==list(E.SCOPES)
    assert ss.shape==(250,2,4,2,6) and keys.shape==(250,3) and counts.sum()==1614484
    with np.load(tp,allow_pickle=False) as z:y,mask,rg,classes=z['y'],z['mask'],z['row_group'],z['market_class']
    assert y.shape==(1873,20000,2) and mask.shape==(1873,20000) and mask.dtype==bool
    with np.load(fp,allow_pickle=False) as z:
        v=z['split']=='valid'; info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v],strict=True)}
        trainchains=set(z['chain'][z['split']=='train'].tolist())
    selected=[(f,i) for f in (0,1) for i,g in enumerate(rg) if info[int(g)][0]==f and info[int(g)][2]]
    assert len(selected)==860 and all(info[int(rg[i])][1] not in trainchains for f,i in selected)
    np.testing.assert_array_equal(keys,np.asarray(sorted({(f,info[int(rg[i])][1],int(classes[i])) for f,i in selected})))
    order=hashlib.sha256(rg.tobytes()).hexdigest()
    for (_,f),r in receipts.items():
        rows=[i for i,g in enumerate(rg) if info[int(g)][0]==f]
        assert r['rows']==rows and r['row_groups']==rg[rows].tolist() and r['source_row_group_order_sha256']==order and r['chain_folds_sha256']==tracked[str(fp)]
    with np.load(hp,allow_pickle=False) as z:
        hs=z['single_seed4']; assert hs.shape==(860,2,6)
        for key,expected in {'positions':[i for f,i in selected],'fold_id':[f for f,i in selected],'row_groups':[rg[i] for f,i in selected],'chains':[info[int(rg[i])][1] for f,i in selected]}.items():np.testing.assert_array_equal(z[key],expected)
    ki={tuple(k):j for j,k in enumerate(keys)}; stats=np.zeros((250,4,2,6)); stats[:,0]=ss[:,0,1]; stats[:,1]=ss[:,0,3]; checkcounts=np.zeros(250,np.int64)
    out.mkdir(); E.write(out/'started.json',{'source_sha256':E.sha(__file__),'allocation_sha256':ah,'input_sha256':tracked})
    for n,(f,i) in enumerate(selected):
        j=ki[f,info[int(rg[i])][1],int(classes[i])]; steps=np.flatnonzero(mask[i]&(np.arange(20000)>=99)); yy=y[i,steps]
        parts=[arrays[name,f][i,steps] for name in ('cont_sh256_s4',*S.ANCHORS)]
        assert np.isfinite(yy).all() and all(np.isfinite(p).all() for p in parts)
        stats[j,2]+=E.sufficient(yy,E.blend(parts)); stats[j,3]+=hs[n]; checkcounts[j]+=len(steps)
    np.testing.assert_array_equal(checkcounts,counts)
    for m in (1,2,3):np.testing.assert_allclose(np.take(stats[:,m],[0,1,3],axis=-1),np.take(stats[:,0],[0,1,3],axis=-1),rtol=0,atol=2e-8)
    draws,strata=E.resamples(keys); result=S.summarize(stats,counts,draws)
    result['folds']={str(f):S.summarize(stats[keys[:,0]==f],counts[keys[:,0]==f],draws[:,keys[:,0]==f]) for f in (0,1)}
    for scope,old in [(result,prior['scopes']['all']),*[(result['folds'][str(f)],prior['scopes']['all']['folds'][str(f)]) for f in (0,1)]]:
        for new,name in [('s3_parent4','parent_control4'),('s3_H2_4','H2_control4')]:S.equal_report(E.clean(scope['models'][new]),old['models'][name])
    assert abs(result['models']['s4_H2_4']['WP']-.7022343699230701)<2e-12
    report={'status':'COMPLETE_FIXED_SEED4_PARENT_DIAGNOSIS','all':result,'models':S.MODELS,'contrasts':S.COEFFICIENTS,'sequences':860,'chains':250,'scored_rows':1614484,
        'uncertainty':{'draws':E.BOOTSTRAPS,'seed':E.SEED,'strata':strata,'interpretation':'Paired whole-chain draws across all four fixed profiles. Mean of two score deltas, not a new blended forecast. Conditional on these fixed models; two observed seeds do not estimate seed-population or selection uncertainty.'},
        'interpretation':'Post-readout fixed saved-endpoint diagnostic on exposed folds. No fit, weight/epoch search, automatic gate or causal attribution of H2 components.',
        'seed3_reproduction':'Both prior absolute profiles, folds and targets within2e-12','seed4_H2_reproduction':.7022343699230701,
        'fits':0,'parent_OOF_jobs':2,'source_sha256':E.sha(__file__),'allocation_sha256':ah,'input_sha256':tracked,'seconds':time.monotonic()-start}
    for p,state in states.items():
        st=Path(p).stat(); assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==state,('input changed',p)
    np.savez_compressed(out/'sufficient.npz',chain_stats=stats,chain_counts=counts,chain_metadata=keys,model_names=np.array(S.MODELS))
    E.write(out/'report.json',report); E.write(out/'receipt.json',{'status':report['status'],'source_sha256':E.sha(__file__),'allocation_sha256':ah,'outputs':{p.name:{'sha256':E.sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    print(json.dumps(E.clean({'status':report['status'],'all':result,'seconds':report['seconds']})))

if __name__=='__main__':main()
