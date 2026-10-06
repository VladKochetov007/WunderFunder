"""Posthoc algebra of existing native six-sum statistics; no raw-data/prediction reads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import hashlib,json,math,resource,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'native-quality-01/evaluation/sufficient.npz'
INPUT_SHA='8d59367fd1641e133a86a74ca053046f72d92cdbbce6cb12e1019f03ee9a021f'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def moments(s):
    W,sy,sp,syy,spp,syp=map(float,s)
    my=sy/W;mp=sp/W;vy=syy/W-my*my;vp=spp/W-mp*mp;c=syp/W-my*mp
    return {'W':W,'mean_y':my,'mean_prediction':mp,'covariance_y_prediction':c,'variance_y':vy,'variance_prediction':vp,'correlation':c/math.sqrt(vy*vp)}

def main():
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    assert sha(INPUT)==INPUT_SHA and not (HERE/'decomposition.json').exists() and not (HERE/'note.txt').exists()
    with np.load(INPUT,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert set(arrays)=={'candidate','reference','positions','row_groups','fold_id','chains'}
    assert arrays['candidate'].shape==arrays['reference'].shape==(860,2,6)
    assert set(arrays['fold_id'].tolist())=={0,1} and len(np.unique(arrays['chains']))==250
    assert np.array_equal(arrays['candidate'][...,[0,1,3]],arrays['reference'][...,[0,1,3]])
    result={'schema':'existing-native-pooled-moment-decomposition-v1','input':str(INPUT),'input_sha256':INPUT_SHA,'source_sha256':sha(__file__),'data_scope':'Only the already produced sufficient.npz; no prediction, label, mask, raw-feature, model or other data file read.','six_sums':['W','sum_w_y','sum_w_prediction','sum_w_y_squared','sum_w_prediction_squared','sum_w_y_prediction'],'weight_definition':'Existing sums use scored post-warmup rows, y and predictions clipped to [-2,2], weights abs(clipped y).','profiles':{},'attribution':{},'formulas':{'fold_weights':'a_f = W_f / sum_f W_f, separately for each target','within_covariance':'sum_f a_f cov_f(y,p)','between_covariance':'sum_f a_f (mean_y_f-mean_y)(mean_p_f-mean_p)','within_variance':'sum_f a_f var_f','between_variance':'sum_f a_f (mean_f-mean)^2','pooled_correlation':'(C_within+C_between)/sqrt((Vy_within+Vy_between)*(Vp_within+Vp_between))','exact_change':'Δrho = ΔC * mean(1/D_candidate,1/D_reference) + mean(C_candidate,C_reference) * Δ(1/D), D=sqrt(Vy*Vp).','variance_attribution':'Split Δ(1/D) by ΔVp_within and ΔVp_between using common secant [1/D_candidate-1/D_reference]/[Vp_candidate-Vp_reference]. This is an exact algebraic decomposition of the existing delta, not an intervention, counterfactual score or fitted calibration.'},'posthoc':True,'exposed_data':True,'alternative_profiles_scored':False,'calibration_or_weight_optimization':False,'bootstrap_rerun':False}
    for profile in ('reference','candidate'):
        values=arrays[profile];record=[]
        for target in (0,1):
            global_m=moments(values[:,target].sum(0));folds={str(f):moments(values[arrays['fold_id']==f,target].sum(0)) for f in (0,1)}
            for fold in folds.values():fold['weight_fraction']=fold['W']/global_m['W']
            cw=sum(m['weight_fraction']*m['covariance_y_prediction'] for m in folds.values())
            cb=sum(m['weight_fraction']*(m['mean_y']-global_m['mean_y'])*(m['mean_prediction']-global_m['mean_prediction']) for m in folds.values())
            yw=sum(m['weight_fraction']*m['variance_y'] for m in folds.values());yb=sum(m['weight_fraction']*(m['mean_y']-global_m['mean_y'])**2 for m in folds.values())
            pw=sum(m['weight_fraction']*m['variance_prediction'] for m in folds.values());pb=sum(m['weight_fraction']*(m['mean_prediction']-global_m['mean_prediction'])**2 for m in folds.values())
            for actual,recomposed in [(global_m['covariance_y_prediction'],cw+cb),(global_m['variance_y'],yw+yb),(global_m['variance_prediction'],pw+pb)]:assert abs(actual-recomposed)<2e-13
            record.append({'target':target,'folds':folds,'pooled':global_m,'within':{'covariance_y_prediction':cw,'variance_y':yw,'variance_prediction':pw},'between':{'covariance_y_prediction':cb,'variance_y':yb,'variance_prediction':pb},'fold0_minus_fold1':{'mean_y':folds['0']['mean_y']-folds['1']['mean_y'],'mean_prediction':folds['0']['mean_prediction']-folds['1']['mean_prediction']}})
        result['profiles'][profile]={'targets':record,'pooled_WP':sum(x['pooled']['correlation'] for x in record)/2,'fold_WP':{str(f):sum(x['folds'][str(f)]['correlation'] for x in record)/2 for f in (0,1)}}
    for target in (0,1):
        r=result['profiles']['reference']['targets'][target];c=result['profiles']['candidate']['targets'][target]
        vy=r['pooled']['variance_y'];assert vy==c['pooled']['variance_y']
        invr=1/math.sqrt(vy*r['pooled']['variance_prediction']);invc=1/math.sqrt(vy*c['pooled']['variance_prediction'])
        average_inverse=(invr+invc)/2;average_cov=(r['pooled']['covariance_y_prediction']+c['pooled']['covariance_y_prediction'])/2
        dp=c['pooled']['variance_prediction']-r['pooled']['variance_prediction']
        secant=(invc-invr)/dp if dp else -.5/(math.sqrt(vy)*r['pooled']['variance_prediction']**1.5)
        changes={component:{key:c[component][key]-r[component][key] for key in ['covariance_y_prediction','variance_y','variance_prediction']} for component in ['within','between']}
        contributions={component+'_covariance':changes[component]['covariance_y_prediction']*average_inverse for component in ['within','between']}
        contributions.update({component+'_prediction_variance':average_cov*secant*changes[component]['variance_prediction'] for component in ['within','between']})
        delta=c['pooled']['correlation']-r['pooled']['correlation'];assert abs(sum(contributions.values())-delta)<2e-13
        result['attribution'][str(target)]={'pooled_correlation_delta':delta,'fold_correlation_deltas':{str(f):c['folds'][str(f)]['correlation']-r['folds'][str(f)]['correlation'] for f in (0,1)},'moment_changes':changes,'correlation_delta_contributions':contributions,'contribution_sum':sum(contributions.values()),'fold_mean_prediction_changes':{str(f):c['folds'][str(f)]['mean_prediction']-r['folds'][str(f)]['mean_prediction'] for f in (0,1)}}
    ref=result['profiles']['reference'];cand=result['profiles']['candidate'];delta=cand['pooled_WP']-ref['pooled_WP']
    assert abs(ref['pooled_WP']-.7023370762430894)<2e-14 and abs(cand['pooled_WP']-.702355321332776)<2e-14
    result['pooled_WP_delta']=delta;result['fold_WP_deltas']={str(f):cand['fold_WP'][str(f)]-ref['fold_WP'][str(f)] for f in (0,1)}
    contributions={key:sum(result['attribution'][str(t)]['correlation_delta_contributions'][key] for t in (0,1))/2 for key in result['attribution']['0']['correlation_delta_contributions']}
    result['WP_delta_contributions']=contributions;assert abs(sum(contributions.values())-delta)<2e-13
    assert sha(INPUT)==INPUT_SHA;result['seconds']=time.perf_counter()-start;result['peak_RSS_platform_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['NumPy_version']=np.__version__
    with (HERE/'decomposition.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'pooled_delta':delta,'fold_deltas':result['fold_WP_deltas'],'WP_delta_contributions':contributions,'target_attributions':result['attribution'],'seconds':result['seconds']},indent=2))
if __name__=='__main__':main()
