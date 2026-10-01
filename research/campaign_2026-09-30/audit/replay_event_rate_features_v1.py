"""Independent raw TRAIN recurrence, ranks and counters for E282."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from replay_joint_origin_clock_v1 import ranks
ROOT=Path(__file__).resolve().parents[3]
AUTHOR=ROOT/'research/campaign_2026-09-30/data_reconstruction/event_rate_features'
HERE=Path(__file__).resolve().parent
OUTPUTS=(0,1,2,3,4,7)

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def corr(c: np.ndarray,a: np.ndarray) -> np.ndarray:
    def normalized(x: np.ndarray) -> np.ndarray:
        out=np.empty(x.shape,dtype=np.float64)
        for j in range(x.shape[1]):
            r=ranks(x[:,j]); r-=r.mean(); n=np.linalg.norm(r)
            out[:,j]=r/n if n else 0
        return out
    return normalized(c).T@normalized(a)

def features(x: np.ndarray) -> tuple[list[str],np.ndarray,np.ndarray]:
    events=[]; names=[]
    for inst in range(2):
        for name,start,stop in (('price',0,22),('volume',22,44),('trade',44,52)):
            e=np.zeros(len(x),bool); e[1:]=(x[1:,inst*52+start:inst*52+stop]!=x[:-1,inst*52+start:inst*52+stop]).any(1)
            events.append(e); names.append(f'i{inst}_{name}')
    market=np.stack(events,axis=1)
    endpoint=x[:,110]==np.float32(-5.199337482452393); endpoint[0]=False
    events.append(endpoint); names.append('quote_endpoint')
    decay=np.array([2.**(-1./h) for name in names for h in (8,64,512)])
    drives=np.repeat(np.stack(events,axis=1),3,axis=1)*(1-decay)
    rates=np.empty(drives.shape,dtype=np.float64); carry=np.zeros(21)
    for row in range(len(x)):
        carry=drives[row]+decay*carry; rates[row]=carry
    step=np.arange(len(x)); ages=[]
    for event in events:
        last=0; age=np.empty(len(x),dtype=np.float64)
        for row,active in enumerate(event):
            if active: last=row
            age[row]=row-last
        ages.append(age)
    labels=[f'{name}_rate_hl{h}' for name in names for h in (8,64,512)]
    labels += [f'{name}_row_age_control' for name in names]+['step_index_control']
    return labels,np.column_stack([rates,*ages,step]),market

def profile(x: np.ndarray,market: np.ndarray) -> dict:
    a=x[:,104:]; delta=a[99:].astype(np.float64)-a[98:-1].astype(np.float64); quiet=~market[99:].any(1)
    rows=[]
    for j in range(8):
        d=delta[:,j]; record={'feature':f'a{j}'}
        for prefix,mask in (('quiet',quiet),('refresh',~quiet)):
            for label,selection in (('negative',d<0),('zero',d==0),('positive',d>0)):
                record[f'{prefix}_{label}']=int((mask&selection).sum())
        record.update(distinct_values=int(np.unique(a[99:,j]).size),lower_endpoint_rows=int((a[99:,j]==np.float32(-5.199337482452393)).sum()),upper_endpoint_rows=int((a[99:,j]==np.float32(5.199337482452393)).sum()))
        rows.append(record)
    return {'rows':19901,'quiet_rows':int(quiet.sum()),'features':rows}

def compare(a: object,b: object) -> float:
    if isinstance(a,dict):
        assert set(a)==set(b)
        return max((compare(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((compare(u,v) for u,v in zip(a,b)),default=0.)
    if isinstance(a,float):
        assert abs(a-b)<1e-11,(a,b)
        return abs(a-b)
    assert a==b,(a,b)
    return 0.

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['fit','evaluate']); args=parser.parse_args()
    start=time.monotonic(); os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    selected=json.loads((AUTHOR/'selection.json').read_text()); previous=AUTHOR.parent/'additional_feature_relations/selection.json'; manifest=ROOT/'research/campaign_2026-09-29/artifacts/splits/fold0.json'
    for key,path in [('source_sha256',AUTHOR/'probe.py'),('preregistration_sha256',AUTHOR/'preregistration.json'),('previous_selection_sha256',previous),('canonical_manifest_sha256',manifest)]: assert selected[key]==sha(path)
    prev=json.loads(previous.read_text()); excluded=set(prev['excluded_components'])|{r['component'] for k in ('fit','evaluation') for r in prev[k]}
    assert selected['excluded_components']==sorted(excluded)
    available=json.loads(manifest.read_text())['sequences']['train']; available.sort(key=lambda r:hashlib.sha256(f"event-rate-features-v1:{r['row_group']}".encode()).digest())
    chosen=[]; used=set(excluded)
    for row in available:
        if row['component'] not in used: chosen.append({k:row[k] for k in ('row_group','component')}); used.add(row['component'])
        if len(chosen)==32: break
    assert chosen[:16]==selected['fit'] and chosen[16:]==selected['evaluation']
    groups=selected['fit' if args.phase=='fit' else 'evaluation']
    columns=[f'i{inst}_{kind}{j}' for inst in range(2) for kind,n in (('p',22),('v',22),('dp',4),('dv',4)) for j in range(n)]+[f'a{j}' for j in range(8)]
    pf=pq.ParquetFile(ROOT/'datasets/train.parquet'); allc=[]; alla=[]; profiles=[]
    for group in groups:
        table=pf.read_row_group(group['row_group'],columns=columns,use_threads=False)
        x=np.stack([table[name].to_numpy() for name in columns],axis=1)
        assert x.shape==(20000,112) and x.dtype==np.float32 and np.isfinite(x).all()
        names,c,events=features(x); allc.append(c[99:]); alla.append(x[99:,104:][:,OUTPUTS].astype(np.float64)); profiles.append({**group,**profile(x,events)})
    c=np.concatenate(allc); a=np.concatenate(alla); del allc,alla
    expected=json.loads((AUTHOR/('fit.json' if args.phase=='fit' else 'results.json')).read_text())
    if args.phase=='fit':
        rho=corr(c,a); indices=np.abs(rho[:21]).argmax(0).tolist()
        actual={'candidate_names':names,'output_features':[f'a{j}' for j in OUTPUTS],'fit_correlations':rho.tolist(),'selected_indices':indices,'selected_fit_correlations':[float(rho[i,j]) for j,i in enumerate(indices)],'fit_profiles':profiles}
        error=compare(actual,{k:expected[k] for k in actual})
    else:
        fitted=json.loads((HERE/'event_rate_independent_fit_v1.json').read_text())['numeric_replay']; indices=fitted['selected_indices']; rho=corr(c[:,indices],a)
        within=np.array([np.diag(corr(c[start:start+19901,indices],a[start:start+19901])) for start in range(0,len(a),19901)])
        age=np.diag(corr(c[:,[21+i//3 for i in indices]],a)); step=corr(c[:,[-1]],a)[0]; summaries=[]
        for j,i in enumerate(indices):
            support=int((np.sign(fitted['selected_fit_correlations'][j])*within[:,j]>=.95).sum())
            summaries.append({'feature':f'a{OUTPUTS[j]}','fixed_rate':names[i],'fit_spearman':fitted['selected_fit_correlations'][j],'evaluation_spearman':float(rho[j,j]),'within_component_median':float(np.median(within[:,j])),'support_components':support,'strong_rank_association_gate':support>=12,'matched_event_row_age_control_spearman':float(age[j]),'step_index_control_spearman':float(step[j])})
        nonsat=[]
        for group in range(16):
            sl=slice(group*19901,(group+1)*19901); diag=[]
            for j,i in enumerate(indices):
                y=a[sl,j]; mask=(y!=np.float32(-5.199337482452393))&(y!=np.float32(5.199337482452393))
                value=float(corr(c[sl][mask][:,[i]],y[mask,None])[0,0]) if mask.sum()>=2 else None
                diag.append({'rows':int(mask.sum()),'spearman':value})
            nonsat.append(diag)
        actual={'evaluation_rows':len(a),'summaries':summaries,'per_component_correlations':within.tolist(),'nonsaturated_per_component':nonsat,'evaluation_profiles':profiles}
        error=compare(actual,{k:expected[k] for k in actual})
    output={'status':'pass','phase':args.phase,'numeric_replay':actual,'maximum_numeric_discrepancy':error,'source_sha256':sha(Path(__file__)),'selection_sha256':sha(AUTHOR/'selection.json'),'author_source_sha256':sha(AUTHOR/'probe.py'),'author_result_sha256':sha(AUTHOR/('fit.json' if args.phase=='fit' else 'results.json')),'selection_reconstructed':True,'targets_masks_VALID_external_models_read':False,'resources':{'wall_seconds':time.monotonic()-start,'rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS)),'cpu_affinity':sorted(os.sched_getaffinity(0)),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'hard_cgroup_claim':False}}
    out=HERE/f'event_rate_independent_{args.phase}_v1.json'
    with out.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':output['status'],'phase':args.phase,'max_error':error,'resources':output['resources']}))
if __name__=='__main__':main()
