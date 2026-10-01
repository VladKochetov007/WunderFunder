"""One fixed native candidate/reference score and paired whole-chain bootstrap."""
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
from common import P,ROOT,Q4,NAMES,METRIC_SHA,NATIVE_SHA,sha,write,guards,rowmaps

def bootstrap(fe,candidate,reference,folds,chains):
    grouped={}
    for f in (0,1):
        keys=list(dict.fromkeys(int(c) for c in chains[folds==f]))
        grouped[f]=np.stack([np.stack([x[(folds==f)&(chains==c)].sum(0) for x in (candidate,reference)]) for c in keys])
    assert sum(map(len,grouped.values()))==250
    rng=np.random.Generator(np.random.PCG64(0));boot=np.empty((2000,3),np.float64)
    for b in range(2000):
        totals=[]
        for f in (0,1):
            block=grouped[f];picks=rng.integers(0,len(block),len(block));totals.append(block[picks].sum(0))
        for i,t in enumerate((totals[0]+totals[1],*totals)):boot[b,i]=fe.pooled(t[0])-fe.pooled(t[1])
    return {'replicates':2000,'seed':0,'generator':'NumPy PCG64','numpy_version':np.__version__,'chain_counts_by_fold':{str(k):len(v) for k,v in grouped.items()},'percentile_CI95':{scope:np.quantile(boot[:,i],[.025,.975],method='linear').tolist() for i,scope in enumerate(('overall','fold0','fold1'))},'resampling':'Paired whole chains with replacement, stratified by outer fold; all sequences and class composition of each selected chain retained. No class stratification.','uncertainty_scope':'Conditional on these fitted models and exposed pure-chain population; excludes training-seed variability, adaptive selection, and hidden-test/domain shift.','selection_adjusted':False,'fresh_confirmation':False}

def main():
    start=time.monotonic();allocation,limits=guards(30);out=P/'evaluation';out.mkdir(exist_ok=False)
    prepared=json.loads((P/'prepared.json').read_text());write(out/'started.json',{'allocation_sha256':sha(P/'allocation.json'),'prepared_sha256':sha(P/'prepared.json'),'source_sha256':sha(__file__),'metric_readouts':1})
    metric=ROOT/'scripts/fold_eval.py';assert sha(metric)==METRIC_SHA
    spec=importlib.util.spec_from_file_location('frozen_native_student_metric',metric);fe=importlib.util.module_from_spec(spec);spec.loader.exec_module(fe)
    frozen=json.loads((ROOT/'research/iterations/2026-10-01-overnight/grader/frozen-data.json').read_text())
    for rel,record in frozen.items():assert sha(ROOT/rel)==record['sha256'] and (ROOT/rel).stat().st_size==record['bytes']
    assert sha(Q4/'quality.json')==allocation['reference_quality_sha256']
    old=json.loads((Q4/'quality.json').read_text());assert old['primary']['quant_wp']==allocation['reference_expected_wp']
    groups,info,orderhash=rowmaps()
    with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:y=z['y'];mask=z['mask']
    assert y.shape==(1873,20000,2) and mask.shape==(1873,20000)
    stats={'candidate':[],'reference':[]};positions=[];foldids=[];chains=[];provenance={}
    for fold in (0,1):
        selected=[i for i,g in enumerate(groups) if info[int(g)][0]==fold];arrays={}
        for name,path in [('candidate',P/'cache'/f'pred_student_quant_f{fold}_valid.npy'),('reference',Q4/'cache'/f'pred_on_q4quant_f{fold}_valid.npy')]:
            receipt=path.with_suffix('.json');m=json.loads(receipt.read_text());digest=sha(path)
            assert digest==m['prediction_sha256'] and m['fold']==fold and m['epoch']==1
            assert m['rows']==selected and m['row_groups']==groups[selected].tolist()
            assert m['source_row_group_order_sha256']==orderhash and m['chain_folds_sha256']==frozen['cache/chain_folds.npz']['sha256']
            assert m['coefficients']==[.25]*4 and m['native_sha256']==NATIVE_SHA
            expected=NAMES if name=='candidate' else ('on_ctld256','on_ctl128s3','on_ctl128s4','on_ctlc500')
            assert [c['run'] for c in m['child_checkpoints']]==[f'{n}_f{fold}' for n in expected]
            assert sha(m['checkpoint'])==m['checkpoint_sha256']
            if name=='candidate':assert m['allocation_sha256']==sha(P/'allocation.json') and m['source_sha256']==sha(P/'predict.py')
            else:assert digest==prepared['reference'][str(fold)]['sha256'] and sha(receipt)==prepared['reference'][str(fold)]['receipt_sha256']
            arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False);assert arrays[name].shape==(1873,20000,2) and arrays[name].dtype==np.float32
            chosen=set(selected)
            for i in range(1873):assert np.isfinite(arrays[name][i]).all() if i in chosen else np.isnan(arrays[name][i]).all()
            provenance[f'{name}_f{fold}']={'path':str(path),'sha256':digest,'receipt_sha256':sha(receipt)}
        for row in selected:
            if not info[int(groups[row])][2]:continue
            m=mask[row:row+1].copy();m[:,:fe.WARMUP]=False
            assert np.isfinite(y[row:row+1]).all()
            for name in stats:stats[name].append(fe.sums(y[row:row+1],arrays[name][row:row+1],m)[0])
            positions.append(row);foldids.append(fold);chains.append(info[int(groups[row])][1])
    stats={k:np.stack(v) for k,v in stats.items()};foldids=np.asarray(foldids);chains=np.asarray(chains);assert len(positions)==860 and len(np.unique(chains))==250
    assert np.array_equal(stats['candidate'][...,[0,1,3]],stats['reference'][...,[0,1,3]])
    def score(sel):
        c=fe.pooled(stats['candidate'][sel].sum(0));r=fe.pooled(stats['reference'][sel].sum(0));return {'candidate_wp':c,'reference_wp':r,'delta':c-r}
    primary=score(np.ones(860,bool));assert abs(primary['reference_wp']-allocation['reference_expected_wp'])<1e-14
    report={'status':'COMPLETE_ONE_FIXED_NATIVE_COMPARISON','primary':primary,'by_fold':{str(f):score(foldids==f) for f in (0,1)},'pure_sequences':860,'pure_chains':250,'coefficients':[.25]*4,'native':'same unchanged guarded VNNI INT8 implementation for candidate and reference','bootstrap':bootstrap(fe,stats['candidate'],stats['reference'],foldids,chains),'prior_GPU_candidate_wp':allocation['prior_GPU_candidate_wp'],'native_minus_prior_GPU_candidate':primary['candidate_wp']-allocation['prior_GPU_candidate_wp'],'GPU_comparison_interpretation':'Combined backend, AMP and dynamic-quantization drift; not an isolated quantization penalty. Prior score is the supplied rounded value.','adaptive_exposure':'Same previously exposed folds and selected student recipe; no independent confirmation or automatic promotion.','provenance':provenance,'metric_source_sha256':METRIC_SHA,'allocation_sha256':sha(P/'allocation.json'),'evaluation_source_sha256':sha(__file__),'seconds':time.monotonic()-start,'limits':limits,'new_models':0,'blend_search':False,'metric_readouts':1}
    for record in provenance.values():assert sha(record['path'])==record['sha256'] and sha(Path(record['path']).with_suffix('.json'))==record['receipt_sha256']
    for rel,record in frozen.items():assert sha(ROOT/rel)==record['sha256']
    np.savez_compressed(out/'sufficient.npz',**stats,positions=np.asarray(positions),row_groups=groups[positions],fold_id=foldids,chains=chains)
    report['sufficient_sha256']=sha(out/'sufficient.npz');write(out/'report.json',report);print(json.dumps({'status':report['status'],'primary':primary,'bootstrap':report['bootstrap'],'GPU_drift':report['native_minus_prior_GPU_candidate']}),flush=True)
if __name__=='__main__':main()
