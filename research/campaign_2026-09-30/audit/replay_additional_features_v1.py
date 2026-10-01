"""Independent fixed-coordinate TRAIN-only E280 replay."""
from __future__ import annotations
import argparse,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr
from replay_event_rate_features_v1 import corr,compare
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
AUTHOR=ROOT/'research/campaign_2026-09-30/data_reconstruction/additional_feature_relations'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def proxies(x:np.ndarray)->tuple[list[str],np.ndarray]:
    names=[]; values=[]
    def add(name:str,value:np.ndarray)->None:names.append(name);values.append(value)
    for instrument in range(2):
        p=x[:,instrument*52:instrument*52+22].astype(np.float64);v=x[:,instrument*52+22:instrument*52+44].astype(np.float64)
        dp=x[:,instrument*52+44:instrument*52+48].astype(np.float64);dv=x[:,instrument*52+48:instrument*52+52].astype(np.float64)
        u=ndtr(v);prefix=f'i{instrument}_'
        for name,value in [('volume_z_sum',v.sum(1)),('volume_z_side_contrast',v[:,:11].sum(1)-v[:,11:].sum(1)),('volume_z_abs_sum',np.abs(v).sum(1)),('volume_cdf_sum',u.sum(1)),('volume_cdf_bid_fraction',u[:,:11].sum(1)/np.maximum(u.sum(1),1e-12)),('price_z_range',p.max(1)-p.min(1)),('price_z_side_contrast',p[:,:11].mean(1)-p[:,11:].mean(1)),('trade_price_z_sum',dp.sum(1)),('trade_volume_z_sum',dv.sum(1)),('trade_volume_cdf_sum',ndtr(dv).sum(1)),('trade_encoded_cross_product',(dp*dv).sum(1))]:add(prefix+name,value)
        for name,block in [('price',p),('volume',v),('trade',np.column_stack([dp,dv]))]:
            event=np.zeros(len(x),bool);event[1:]=(block[1:]!=block[:-1]).any(1)
            age=np.empty(len(x),np.float64);last=0
            for j,active in enumerate(event):
                if active:last=j
                age[j]=j-last
            add(prefix+name+'_row_age',age)
            add(prefix+name+'_refresh_count100',np.convolve(event.astype(np.int64),np.ones(100,dtype=np.int64),mode='full')[:len(x)])
    endpoint=x[:,110]==np.float32(-5.199337482452393);age=np.empty(len(x),np.float64);last=0
    for j,active in enumerate(endpoint):
        if active:last=j
        age[j]=j-last
    add('quote_endpoint_row_age',age);add('step_in_sequence',np.arange(len(x)))
    return names,np.column_stack(values)
def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','evaluate']);args=parser.parse_args();start=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    selection=json.loads((AUTHOR/'selection.json').read_text());manifest_path=ROOT/'research/campaign_2026-09-29/artifacts/splits/fold0.json';manifest=json.loads(manifest_path.read_text())
    assert selection['canonical_manifest_sha256']==sha(manifest_path) and selection['source_sha256']==sha(AUTHOR/'probe.py')
    train={r['row_group']:r['component'] for r in manifest['sequences']['train']};valid={r['row_group']:r['component'] for r in manifest['sequences']['valid']};excluded=set()
    for name,digest in selection['prior_selection_sha256'].items():
        p=ROOT/name;assert sha(p)==digest; prior=json.loads(p.read_text())
        for key in ('fit_train_groups','test_train_groups','map_fit_train_groups'):
            if 'spread_aux' in name and key=='fit_train_groups':continue
            excluded.update(train[g] for g in prior.get(key,[]))
        excluded.update(r['component'] for r in prior.get('selected',[]));excluded.update(valid[g] for g in prior.get('evaluation_valid_groups',[]))
    assert sorted(excluded)==selection['excluded_components'];rows=manifest['sequences']['train'];rows.sort(key=lambda r:hashlib.sha256(f"additional-feature-relations-v1:{r['row_group']}".encode()).digest());chosen=[];used=set(excluded)
    for row in rows:
        if row['component'] not in used:chosen.append({k:row[k] for k in ('row_group','component')});used.add(row['component'])
        if len(chosen)==32:break
    assert chosen[:16]==selection['fit'] and chosen[16:]==selection['evaluation']
    columns=[f'i{inst}_{kind}{j}' for inst in range(2) for kind,n in [('p',22),('v',22),('dp',4),('dv',4)] for j in range(n)]+[f'a{j}' for j in range(8)]
    pf=pq.ParquetFile(ROOT/'datasets/train.parquet');cs=[];ys=[]
    for member in selection['fit' if args.phase=='fit' else 'evaluation']:
        table=pf.read_row_group(member['row_group'],columns=columns,use_threads=False);x=np.stack([table[n].to_numpy() for n in columns],1);assert x.shape==(20000,112) and x.dtype==np.float32 and np.isfinite(x).all()
        names,c=proxies(x);cs.append(c);ys.append(x[:,104:109].astype(np.float64))
    c=np.concatenate(cs);a=np.concatenate(ys);del cs,ys
    wanted=json.loads((AUTHOR/('fit.json' if args.phase=='fit' else 'results.json')).read_text())
    if args.phase=='fit':
        rho=corr(c,a);indices=np.abs(rho).argmax(0).tolist();actual={'candidate_names':names,'fit_spearman':rho.tolist(),'selected_indices':indices,'selected_fit_correlations':[float(rho[i,j]) for j,i in enumerate(indices)],'fit_rows':len(a)}
    else:
        fitted=json.loads((HERE/'additional_features_independent_fit_v1.json').read_text())['numeric_replay'];indices=fitted['selected_indices'];rho=corr(c[:,indices],a);within=np.array([np.diag(corr(c[s:s+20000,indices],a[s:s+20000])) for s in range(0,len(a),20000)]);summaries=[]
        for j,i in enumerate(indices):
            sign=np.sign(fitted['selected_fit_correlations'][j]);w=within[:,j]
            summaries.append({'feature':f'a{j}','fixed_proxy':names[i],'fit_spearman':fitted['selected_fit_correlations'][j],'evaluation_spearman':float(rho[j,j]),'within_component_median_spearman':float(np.median(w)),'within_component_sign_adjusted_min':float((sign*w).min()),'components_with_sign_adjusted_rho_ge_0.95':int((sign*w>=.95).sum())})
        actual={'evaluation_rows':len(a),'summaries':summaries,'per_component_spearman':within.tolist()}
    error=compare(actual,{k:wanted[k] for k in actual});out={'status':'pass','numeric_replay':actual,'maximum_numeric_discrepancy':error,'selection_reconstructed':True,'source_sha256':sha(Path(__file__)),'author_result_sha256':sha(AUTHOR/('fit.json' if args.phase=='fit' else 'results.json')),'forbidden_data_read':False,'resources':{'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS)),'cpu_affinity':sorted(os.sched_getaffinity(0)),'hard_cgroup_claim':False}}
    with (HERE/f'additional_features_independent_{args.phase}_v1.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'pass','phase':args.phase,'max_error':error,'resources':out['resources']}))
if __name__=='__main__':main()
