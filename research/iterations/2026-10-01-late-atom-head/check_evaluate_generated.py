"""Generated-only independent centered-Pearson oracle for the three-profile grader."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def oracle(y,p):
    y=np.clip(y.astype(np.float64),-2,2);p=np.clip(p.astype(np.float64),-2,2);w=abs(y)
    W=w.sum(0);safe=np.where(W>0,W,1)
    yc=y-(w*y).sum(0)/safe;pc=p-(w*p).sum(0)/safe
    yy=(w*yc*yc).sum(0);pp=(w*pc*pc).sum(0)
    valid=(W>=1e-8)&(yy/safe>1e-16)&(pp/safe>1e-16)
    rho=np.zeros(2);np.divide((w*yc*pc).sum(0),np.sqrt(yy*pp),out=rho,where=valid)
    return rho,valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert a.output.resolve().is_relative_to(HERE) and not a.output.exists()
    started=time.monotonic();source=HERE/'evaluate.py'
    spec=importlib.util.spec_from_file_location('late_grader_generated_check',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(20261001);ys=[];ps=[];stats=[]
    for i,n in enumerate((13,19,11)):
        y=rng.normal(size=(n,2))*2.1
        if i==2:y[:,0]=0  # zero metric mass for target0 in an entire bootstrap draw
        pred=np.stack([y+rng.normal(size=(n,2))*s for s in (.5,.8,1.2)])
        ys.append(y);ps.append(pred);stats.append(np.stack([m.E.sufficient(y,p) for p in pred]))
    counts=np.array([len(y) for y in ys]);stats=np.asarray(stats)
    draws=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1],[2,0,1]],np.int16)
    got=m.summarize(stats,counts,draws);rho=[];valid=[]
    for draw in draws:
        rows=np.repeat(np.arange(3),draw);y=np.concatenate([ys[j] for j in rows])
        values=[oracle(y,np.concatenate([ps[j][k] for j in rows])) for k in range(3)]
        rho.append([v[0] for v in values]);valid.append([v[1] for v in values])
    rho=np.asarray(rho);valid=np.asarray(valid)
    for k,name in enumerate(m.MODELS):
        np.testing.assert_allclose(got['models'][name]['rho'],rho[3,k],atol=2e-12,rtol=0)
        np.testing.assert_allclose(got['models'][name]['WP'],rho[3,k].mean(),atol=2e-12,rtol=0)
    for name,(candidate,reference) in m.PAIRS.items():
        delta=rho[:,candidate]-rho[:,reference];ok=valid[:,candidate]&valid[:,reference];q=got['contrasts'][name]
        np.testing.assert_allclose(q['delta_rho'],delta[3],atol=2e-12,rtol=0)
        np.testing.assert_allclose(q['delta_WP'],delta[3].mean(),atol=2e-12,rtol=0)
        for result,values,keep in [(q['paired_chain_CI'],delta.mean(-1),ok.all(-1)),*[(q['paired_target_CI'][t],delta[:,t],ok[:,t]) for t in (0,1)]]:
            assert result['valid_draws']==int(keep.sum())
            np.testing.assert_allclose(result['interval95'],np.quantile(values[keep],[.025,.975]),atol=2e-12,rtol=0)
    q=got['contrasts'];np.testing.assert_allclose(q['primary_candidate_minus_incumbent']['delta_WP'],q['candidate_minus_CE0_control']['delta_WP']+q['CE0_control_minus_incumbent']['delta_WP'],atol=2e-12,rtol=0)
    receipt={'status':'PASS_GENERATED_INDEPENDENT_THREE_PROFILE_ORACLE','test_source_sha256':sha(__file__),'evaluator_sha256':sha(source),'numerical_module_sha256':m.BSHA,
        'checks':['clipped weighted centered-Pearson absolute profiles','three contrast signs and target deltas','same whole-chain bootstrap draw contrasts','zero-target-mass draws excluded from appropriate CI','primary equals sum of two diagnostic point contrasts'],
        'generated_sequences':3,'generated_rows':int(counts.sum()),'resampling_draws':len(draws),'tolerance_atol':2e-12,'tolerance_rtol':0,'actual_data_arrays_read':0,'model_inference':0,'fits':0,'seconds':time.monotonic()-started}
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
