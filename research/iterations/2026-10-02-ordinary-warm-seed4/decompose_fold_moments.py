"""Post-hoc descriptive decomposition of existing sums; no prediction changes."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records={}
for name in ('ordinary-warm','ordinary-warm-seed4'):
    p=ROOT/f'research/iterations/2026-10-02-{name}/evaluation-01/sufficient.npz'
    with np.load(p,allow_pickle=False) as z:
        s=z['chain_stats'][:,0];meta=z['chain_metadata'];names=z['model_names'].tolist()
    result={}
    for i,model in enumerate(names):
        folds=np.stack([s[meta[:,0]==f,i].sum(0) for f in (0,1)])
        W,Y,P,YY,PP,YP=np.moveaxis(folds,-1,0)
        mass=W/W.sum(0);my=Y/W;mp=P/W;vy=YY/W-my*my;vp=PP/W-mp*mp;cov=YP/W-my*mp
        MY=(mass*my).sum(0);MP=(mass*mp).sum(0)
        within_cov=(mass*cov).sum(0);between_cov=(mass*(my-MY)*(mp-MP)).sum(0)
        within_vy=(mass*vy).sum(0);within_vp=(mass*vp).sum(0)
        between_vy=(mass*(my-MY)**2).sum(0);between_vp=(mass*(mp-MP)**2).sum(0)
        denominator=np.sqrt((within_vy+between_vy)*(within_vp+between_vp))
        within_part=within_cov/denominator;between_part=between_cov/denominator
        total=(within_cov+between_cov)/denominator
        pooled=s[:,i].sum(0);w,y,pred,yy,pp,yp=pooled.T
        direct=(yp/w-y*pred/w**2)/np.sqrt((yy/w-(y/w)**2)*(pp/w-(pred/w)**2))
        np.testing.assert_allclose(total,direct,rtol=0,atol=1e-14)
        result[model]=dict(pooled_WP=float(total.mean()),fold_WP=(cov/np.sqrt(vy*vp)).mean(-1).tolist(),
            fold_target_mean=my.tolist(),fold_prediction_mean=mp.tolist(),fold_prediction_SD=np.sqrt(vp).tolist(),
            within_covariance_contribution_to_pooled=within_part.tolist(),between_covariance_contribution_to_pooled=between_part.tolist(),
            fold_centered_correlation_diagnostic_only=(within_cov/np.sqrt(within_vy*within_vp)).tolist())
    control=names[2] if name=='ordinary-warm' else names[1]
    first=result[names[0]];base=result[control]
    result['paired_decomposition']={k:float(np.mean(first[k])-np.mean(base[k])) for k in (
        'within_covariance_contribution_to_pooled','between_covariance_contribution_to_pooled','fold_centered_correlation_diagnostic_only')}
    result['paired_decomposition']['pooled_delta']=first['pooled_WP']-base['pooled_WP']
    records[name]=dict(input_sha256=sha(p),models=result)
out=dict(status='DESCRIPTIVE_EXISTING_SUMS_ONLY',records=records,source_sha256=sha(Path(__file__)),
    interpretation='Within/between covariance terms share each model total variance denominator. Their paired changes sum exactly to pooled delta but are not causal effects. Fold-centered correlation is NOT the competition metric, is not used for selection, and does not define a deployable calibration. Folds were already exposed; no new predictions or fits.')
with (HERE/'fold-moment-decomposition.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
for name,r in records.items():print(name,json.dumps(r['models']['paired_decomposition']))
