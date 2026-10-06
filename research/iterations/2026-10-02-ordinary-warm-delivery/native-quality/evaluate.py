"""Fixed native warm4 versus native incumbent; GPU-warm drift is secondary."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from common import P,ROOT,Q4,NAMES,METRIC_SHA,NATIVE_SHA,sha,write,identity,guards,rowmaps
B=ROOT/'research/iterations/2026-10-01-ensemble-backbone/evaluate.py'
BSHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
WSHA='f29b97efa4e767cb85f83e9c44dcee678ed4b84e9f9e28dce18e31f22d6cf865'
CSHA='9dd71d06373121176133e1cfa3bd2455eabcd40f35f3e38824908c52a88f4986'
INCUMBENT=.702355321332776
MODELS=('ordinary_warm_native4','current_student_native4','ordinary_warm_GPU4')
PAIRS={'primary':(0,1),'native_minus_GPU_warm':(0,2)}

def main():
    start=time.monotonic();allocation,limits=guards(30);out=P/'evaluation';assert not out.exists();tracked={};states={}
    assert P==ROOT/'research/iterations/2026-10-02-ordinary-warm-delivery/native-quality'
    assert Q4==ROOT/'research/iterations/2026-10-01-continuation/student/native-quality-01'
    def pin(path,digest):
        path=Path(path)
        if str(path) in tracked:assert tracked[str(path)]==digest;return
        assert sha(path)==digest,('hash',str(path));tracked[str(path)]=digest;states[str(path)]=identity(path)
    pin(P/'allocation.json',sha(P/'allocation.json'));allocation_hash=tracked[str(P/'allocation.json')]
    for rel,h in allocation['source_sha256'].items():pin(ROOT/rel,h)
    assert allocation['source_sha256'][str(Path(__file__).relative_to(ROOT))]==sha(__file__)
    pin(B,BSHA);pin(W/'evaluate.py',WSHA);pin(Q4/'evaluate.py',CSHA);pin(ROOT/'scripts/fold_eval.py',METRIC_SHA)
    spec=importlib.util.spec_from_file_location('fixed_native_warm_math',B);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    prepared_path=P/'prepared.json';pin(prepared_path,sha(prepared_path));prepared=json.loads(prepared_path.read_text())
    assert prepared['status']=='BOUND_TWO_FOLDS_BEFORE_PREDICTION' and prepared['allocation_sha256']==allocation_hash
    for rel,h in allocation['evaluation_input_sha256'].items():pin(ROOT/rel,h)
    required=[W/'evaluation-01/report.json',W/'evaluation-01/sufficient.npz',Q4/'evaluation/report.json',Q4/'evaluation/sufficient.npz',*[Q4/'cache'/f'pred_student_quant_f{f}_valid.json' for f in (0,1)]]
    for p in required:assert str(p.relative_to(ROOT)) in allocation['evaluation_input_sha256']
    wr=json.loads(required[0].read_text());oldreport=json.loads(required[2].read_text())
    assert wr['status']=='COMPLETE_FIXED_ORDINARY_WARM_SCREEN' and wr['source_sha256']==WSHA
    assert oldreport['status']=='COMPLETE_ONE_FIXED_NATIVE_COMPARISON' and oldreport['evaluation_source_sha256']==CSHA
    assert oldreport['metric_source_sha256']==METRIC_SHA and oldreport['sufficient_sha256']==tracked[str(required[3])]
    assert abs(oldreport['primary']['candidate_wp']-INCUMBENT)<2e-14
    for rel in ('cache/valid_targets.npz','cache/chain_folds.npz'):
        r=prepared['inputs'][rel];assert identity(ROOT/rel)=={k:r[k] for k in ('resolved','bytes','mtime_ns')};pin(ROOT/rel,r['sha256'])
    groups,info,orderhash=rowmaps();selected=[(f,i) for f in (0,1) for i,g in enumerate(groups) if info[int(g)][0]==f and info[int(g)][2]];assert len(selected)==860
    predictions={};references={};new_receipts={}
    for f in (0,1):
        exitpath=P/f'fold{f}-exit.json';pin(exitpath,sha(exitpath));done=json.loads(exitpath.read_text());assert done['exit_code']==0 and done['name']==f'fold{f}'
        path=P/'cache'/f'pred_ordinary_warm_quant_f{f}_valid.npy';rp=path.with_suffix('.json');pin(rp,sha(rp));m=json.loads(rp.read_text());pin(path,m['prediction_sha256'])
        assert m['status']=='COMPLETE_ONE_NATIVE_INT8_OOF_PASS' and m['run']==f'ordinary_warm_quant_f{f}' and m['fold']==f and m['epoch']==1
        assert m['allocation_sha256']==allocation_hash and m['source_sha256']==allocation['source_sha256'][str((P/'predict.py').relative_to(ROOT))]
        assert m['targets_decoded'] is False and m['masks_decoded'] is False and m['WP_queries']==0 and m['coefficients']==[.25]*4 and m['native_sha256']==NATIVE_SHA
        assert m['native_backend']=='natural auto VNNI (code2)' and m['prediction_path']==str(path)
        rows=[i for i,g in enumerate(groups) if info[int(g)][0]==f]
        assert m['rows']==rows and m['row_groups']==groups[rows].tolist() and m['source_row_group_order_sha256']==orderhash and m['chain_folds_sha256']==tracked[str(ROOT/'cache/chain_folds.npz')]
        manifest=P/f'fold{f}/checkpoint.json';assert m['checkpoint']==str(manifest);pin(manifest,m['checkpoint_sha256'])
        mm=json.loads(manifest.read_text());assert prepared['fold_manifests'][str(f)]==tracked[str(manifest)] and mm['children']==m['child_checkpoints']
        assert mm['allocation_sha256']==allocation_hash and mm['native_sha256']==NATIVE_SHA and mm['fold']==f and mm['predictor_sha256']==m['source_sha256']
        oldrp=Q4/'cache'/f'pred_student_quant_f{f}_valid.json';old=json.loads(oldrp.read_text());ref=prepared['reference'][str(f)]
        assert ref['receipt']==str(oldrp) and ref['receipt_sha256']==tracked[str(oldrp)] and old['prediction_sha256']==ref['sha256']
        assert old['prediction_path']==ref['path'] and oldreport['provenance'][f'candidate_f{f}']['sha256']==ref['sha256'] and oldreport['provenance'][f'candidate_f{f}']['receipt_sha256']==tracked[str(oldrp)]
        assert old['rows']==rows and old['row_groups']==groups[rows].tolist() and old['source_row_group_order_sha256']==orderhash and old['chain_folds_sha256']==m['chain_folds_sha256']
        assert old['native_sha256']==NATIVE_SHA and old['coefficients']==[.25]*4 and old['fold']==f and old['epoch']==1
        assert [c['run'] for c in old['child_checkpoints']]==[f'{n}_f{f}' for n in ('cont_sh256_s3h2',*NAMES[1:])]
        assert [c['run'] for c in m['child_checkpoints']]==[f'{n}_f{f}' for n in NAMES]
        for i,c in enumerate(m['child_checkpoints']):
            assert c['coef']==.25 and c['epoch']==1 and c['fold']==f;pin(Path(c['checkpoint']),c['checkpoint_sha256'])
            assert c['checkpoint_sha256']==(allocation['warm_checkpoint_sha256'][str(f)] if i==0 else old['child_checkpoints'][i]['checkpoint_sha256'])
            if i==0:assert c['checkpoint_sha256']==wr['fit_integrity'][f'{NAMES[0]}_f{f}']['checkpoint']['sha256']
        array=np.load(path,mmap_mode='r',allow_pickle=False);assert array.shape==(1873,20000,2) and array.dtype==np.float32;chosen=set(rows)
        for i in range(1873):assert np.isfinite(array[i]).all() if i in chosen else np.isnan(array[i]).all()
        predictions[f]=array;new_receipts[str(f)]=m;references[str(f)]=dict(prediction_sha256=ref['sha256'],receipt_sha256=tracked[str(oldrp)])
    # Both native passes and all old/new identities checked before target/statistical reads.
    with np.load(required[1],allow_pickle=False) as z:
        gpu=z['chain_stats'];counts=z['chain_counts'][:,0];keys=z['chain_metadata'];assert z['scope_names'].tolist()==['all','class3_quiet64'] and z['model_names'].tolist()==['ordinary_warm4','interval_instantaneous4','current_student4']
    assert gpu.shape==(250,2,3,2,6) and keys.shape==(250,3) and counts.shape==(250,) and counts.sum()==1614484
    ki={(int(f),int(c)):j for j,(f,c,cl) in enumerate(keys)};assert len(ki)==250
    with np.load(required[3],allow_pickle=False) as z:
        historical=z['candidate'];assert historical.shape==(860,2,6) and np.isfinite(historical).all()
        for k,expected in {'positions':[i for f,i in selected],'fold_id':[f for f,i in selected],'row_groups':[groups[i] for f,i in selected],'chains':[info[int(groups[i])][1] for f,i in selected]}.items():np.testing.assert_array_equal(z[k],expected)
    with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:y=z['y'];mask=z['mask']
    assert y.shape==(1873,20000,2) and mask.shape==(1873,20000) and mask.dtype==np.bool_
    stats=np.zeros((250,3,2,6));stats[:,2]=gpu[:,0,0];checkcounts=np.zeros(250,np.int64)
    out.mkdir();write(out/'started.json',dict(allocation_sha256=allocation_hash,source_sha256=sha(__file__),input_sha256=tracked,metric_readouts=1))
    for n,(f,i) in enumerate(selected):
        j=ki[f,info[int(groups[i])][1]];steps=np.flatnonzero(mask[i]&(np.arange(20000)>=99));yy=y[i,steps];pp=predictions[f][i,steps]
        assert np.isfinite(yy).all() and np.isfinite(pp).all();stats[j,0]+=E.sufficient(yy,pp);stats[j,1]+=historical[n];checkcounts[j]+=len(steps)
    np.testing.assert_array_equal(checkcounts,counts)
    for k in (0,1):np.testing.assert_allclose(np.take(stats[:,k],[0,1,3],axis=-1),np.take(stats[:,2],[0,1,3],axis=-1),rtol=0,atol=2e-8)
    assert E.metric(stats.sum(0))['valid'].all() and np.isfinite(E.metric(stats.sum(0))['WP']).all()
    assert abs(E.metric(stats[:,1].sum(0))['WP']-INCUMBENT)<2e-12
    assert abs(E.metric(stats[:,2].sum(0))['WP']-wr['comparisons']['primary']['scopes']['all']['candidate']['WP'])<2e-12
    for f in (0,1):
        keep=keys[:,0]==f;assert abs(E.metric(stats[keep,1].sum(0))['WP']-oldreport['by_fold'][str(f)]['candidate_wp'])<2e-12
        assert abs(E.metric(stats[keep,2].sum(0))['WP']-wr['comparisons']['primary']['scopes']['all']['folds'][str(f)]['candidate']['WP'])<2e-12
    draws,strata=E.resamples(keys);comparisons={}
    for label,pair in PAIRS.items():
        part=stats[:,list(pair)];result=E.comparison(part,counts,draws)
        result['folds']={str(f):E.comparison(part[keys[:,0]==f],counts[keys[:,0]==f],draws[:,keys[:,0]==f]) for f in (0,1)}
        comparisons[label]=dict(candidate=MODELS[pair[0]],control=MODELS[pair[1]],scopes={'all':result})
    report=dict(status='COMPLETE_FIXED_NATIVE_ORDINARY_WARM_COMPARISON',models=MODELS,comparisons=comparisons,primary='ordinary_warm_native4 minus current_student_native4',secondary='ordinary_warm_native4 minus ordinary_warm_GPU4',native_incumbent_reproduced=INCUMBENT,pure_sequences=860,pure_chains=250,scored_rows=1614484,coefficients=[.25]*4,native_sha256=NATIVE_SHA,metric_source_sha256=METRIC_SHA,pooled_helper_sha256=BSHA,source_sha256=sha(__file__),allocation_sha256=allocation_hash,provenance=dict(new_native_receipts=new_receipts,old_native_receipts=references,input_sha256=tracked),uncertainty=dict(replicates=E.BOOTSTRAPS,seed=E.SEED,strata=strata,interpretation='Shared whole-chain fold/class-stratified paired draws. Conditional on fixed fitted models and exposed folds; excludes training-seed, adaptive-selection and hidden-test uncertainty. Old native score reproduced; its prior fold-only interval is not reused.'),GPU_comparison_interpretation='Same warm weights/profile; combined AMP, native arithmetic and dynamic-quantization drift, not an isolated quantization penalty or standalone model quality.',new_models=0,blend_search=False,metric_readouts=1,limits=limits,seconds=time.monotonic()-start)
    for path,h in tracked.items():assert identity(path)==states[path] and sha(path)==h,('input changed',path)
    np.savez_compressed(out/'sufficient.npz',chain_stats=stats[:,None],chain_counts=counts[:,None],chain_metadata=keys,scope_names=np.array(['all']),model_names=np.array(MODELS))
    report['sufficient_sha256']=sha(out/'sufficient.npz');write(out/'report.json',E.clean(report));print(json.dumps(E.clean(dict(status=report['status'],comparisons=comparisons,seconds=report['seconds']))),flush=True)
if __name__=='__main__':main()
