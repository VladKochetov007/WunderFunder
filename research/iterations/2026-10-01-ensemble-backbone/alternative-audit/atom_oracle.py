"""Generated finite-population algebra audit; no Torch, data, model or WP reads.

All cases are fixed before evaluation. One-core BLAS environment, 120 CPU-second
limit. NumPy transcribes the inspected decoder equations rather than importing
the model. Zero CE mass denotes an unidentified categorical optimum.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCES=('scripts/train_rnn.py','src/wnn/atom_head.py','src/wnn/outcomes.py','src/wnn/nn.py')


def ratio(a,b):return np.divide(a,b,out=np.zeros_like(a),where=b>0)


def decoder(pi,g):
    # AtomRNNModel.forward: raw atoms, no inverse joint-CE tilt.
    return (pi@(g*abs(g)))/np.maximum(pi@abs(g),1e-6)


def code_readout(pi,m1,m0,tilted):
    # outcomes.readout: inverse mean-|clipped y| tilt; normalization cancels.
    if tilted:pi=pi/np.maximum(m0.mean(-1),1e-3)
    return (pi@m1)/np.maximum(pi@m0,1e-6)


def case(name,atoms,counts,row_weights):
    g=np.asarray(atoms,np.float64);counts=np.asarray(counts,np.float64)
    r=np.asarray(row_weights,np.float64);c=np.clip(g,-2,2)
    mass=counts*r;a=abs(c).mean(-1);w=mass[:,None]*abs(c)
    denom=w.sum(0);opt=ratio((w*c).sum(0),denom)
    ce_mass=mass*a;identified=bool(ce_mass.sum()>0)
    pi=ce_mass/ce_mass.sum() if identified else np.zeros(len(g))
    base=mass/mass.sum() if mass.sum()>0 else np.zeros(len(g))
    un=ratio(pi,a)
    estimates={'current_raw_atom_decode':decoder(pi,g),
               'clip_atoms_only':decoder(pi,c),
               'inverse_tilt_only':decoder(un,g),
               'inverse_tilt_and_clipped_atoms':decoder(un,c),
               'untilted_CE_and_clipped_atoms':decoder(base,c),
               'outcomes_exact_code_tilted':code_readout(pi,c*abs(c),abs(c),True),
               'outcomes_exact_code_untilted':code_readout(base,c*abs(c),abs(c),False)}
    risk_at_opt=ratio((w*(opt-c)**2).sum(0),denom)
    rows={}
    for key,p in estimates.items():
        risk=ratio((w*(p-c)**2).sum(0),denom)
        regret=risk-risk_at_opt
        np.testing.assert_allclose(regret,np.where(denom>0,(p-opt)**2,0),rtol=1e-11,atol=1e-12)
        rows[key]={'prediction':p.tolist(),'excess_target_weighted_MSE':regret.tolist()}
    for key in ('inverse_tilt_and_clipped_atoms','untilted_CE_and_clipped_atoms',
                'outcomes_exact_code_tilted','outcomes_exact_code_untilted'):
        np.testing.assert_allclose(estimates[key][denom>0],opt[denom>0],rtol=0,atol=1e-12)
    np.testing.assert_allclose((w*(opt-c)).sum(0),0,atol=1e-12)
    active=ce_mass>0
    ce_opt=-(pi[active]*np.log(pi[active])).sum() if identified else 0.0
    ce_untilted=-(pi[active]*np.log(base[active])).sum() if identified else 0.0
    assert ce_untilted>=ce_opt-1e-12
    pclip=np.clip(estimates['current_raw_atom_decode'],-2,2)
    return {'name':name,'atoms':g.tolist(),'counts':counts.tolist(),'row_weights':r.tolist(),
            'clipped_atoms':c.tolist(),'CE_row_tilt':a.tolist(),'isolated_CE_optimum':pi.tolist(),
            'CE_identified':identified,'target_risk_identified':(denom>0).tolist(),
            'CE_excess_using_untilted_distribution':float(ce_untilted-ce_opt),
            'weighted_MSE_optimum':opt.tolist(),'alternatives':rows,
            'current_after_prediction_clip':pclip.tolist(),
            'current_after_prediction_clip_excess_MSE':ratio((w*(pclip-c)**2).sum(0),denom).tolist()
                if not identified else (ratio((w*(pclip-c)**2).sum(0),denom)-risk_at_opt).tolist()}


def coarse_code_check():
    # Two contexts, same actual outcome-code bin: class0, spread>=7, U-sp=1.
    # Their clipped targets differ, so pooled per-code moments discard information.
    q=.12543
    y=q*np.array([[1,-9],[1,-12]],float)
    m0=abs(y).mean(0)[None];m1=(abs(y)*y).mean(0)[None]
    d=code_readout(np.ones(1),m1,m0,True)
    gap=d[None]-y
    assert np.max(abs(gap[:,1]))>.1
    return {'contexts':2,'same_code':'class0, spread_bin7plus, above_bin1',
            'targets':y.tolist(),'global_code_readout_both_contexts':d.tolist(),
            'conditional_Bayes_predictions':y.tolist(),'prediction_errors':gap.tolist(),
            'claim':'Inverse tilt does not restore within-code conditional moments lost by global tables.'}


def main():
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    start=time.monotonic();dest=HERE/'result.json'
    assert not dest.exists(),'one fixed oracle run; preserve old output'
    q=.65687
    cases=[case('class3_law_within_clip',q*np.array([[-3,1],[1,-3],[-1,-1],[-1,0],[0,-1]]),
                [18,7,11,9,5],[1,1,.75,.5,2]),
           case('class3_law_with_clipping',q*np.array([[-4,1],[2,-3],[-2,0],[0,-2]]),
                [3,2,5,7],[1,.8,1.5,.6]),
           case('clipping_and_zero_weights',[[3,.5],[-1,4],[.4,-.2],[0,0],[8,-9]],
                [6,4,8,3,9],[1,.8,.6,1,0]),
           case('one_zero_weight_target',[[0,1],[0,-2],[0,0]],[2,3,5],[1,1,1]),
           case('all_zero_row_weights',[[1,-1],[3,.5],[0,0]],[2,3,1],[0,0,0])]
    assert sum(cases[0]['alternatives']['current_raw_atom_decode']['excess_target_weighted_MSE'])>1e-4
    assert sum(cases[1]['alternatives']['inverse_tilt_only']['excess_target_weighted_MSE'])>1e-4
    assert not cases[-1]['CE_identified'] and cases[-1]['target_risk_identified']==[False,False]
    result={'status':'PASS_FIXED_GENERATED_ORACLE','seconds':time.monotonic()-start,
            'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
            'oracle_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'case_count':len(cases),'cases':cases,'coarse_code_check':coarse_code_check(),
            'scope':'NumPy equation transcription on constructed populations only; no Torch parity/training, actual data/model reads, WP computation or B result access.',
            'peak_RSS_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
                      'case0_optimum':cases[0]['weighted_MSE_optimum'],
                      'case0_actual':cases[0]['alternatives']['current_raw_atom_decode'],
                      'case1_untilt_only':cases[1]['alternatives']['inverse_tilt_only'],
                      'coarse_code':result['coarse_code_check']}))


if __name__=='__main__':main()
