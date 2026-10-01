"""Independent float64 tied-rank book copula statistics and prefix audit."""
from __future__ import annotations
import csv,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from replay_joint_origin_clock_v1 import ranks
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
AUTHOR=ROOT/'research/campaign_2026-09-30/data_reconstruction/book_copula'
SELECTION=AUTHOR.parent/'additional_feature_relations/selection.json'
MD=Path('/home/vlad/development/ultimate/data/MD/lob.csv')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(x:np.ndarray)->dict:
    finite=np.isfinite(x).all(1);x=x[finite].astype(np.float64)
    if len(x)<100:return {'rows':len(x),'status':'insufficient finite rows'}
    r=np.stack([ranks(x[:,j]) for j in range(x.shape[1])],axis=1);r-=r.mean(0);norm=np.linalg.norm(r,axis=0);r=np.divide(r,norm,out=np.zeros_like(r),where=norm!=0)
    c=r.T@r;e=np.linalg.eigvalsh(c)[::-1];trace=e.sum();within=np.concatenate([c[:11,:11][np.triu_indices(11,1)],c[11:,11:][np.triu_indices(11,1)]]);cross=c[:11,11:].reshape(-1)
    return {'status':'complete','rows':len(x),'nonfinite_rows':int((~finite).sum()),'active_columns':int((norm>0).sum()),'price_or_quantity_spectrum':(e/max(trace,1e-30)).tolist(),'top_eigenvalue_share':float(e[0]/max(trace,1e-30)),'top_two_share':float(e[:2].sum()/max(trace,1e-30)),'effective_rank':float(trace**2/max((e**2).sum(),1e-30)),'within_side_rho_quantiles':np.quantile(within,[.1,.5,.9]).tolist(),'cross_side_rho_quantiles':np.quantile(cross,[.1,.5,.9]).tolist(),'distinct_values_per_column':[int(np.unique(x[:,j]).size) for j in range(22)]}
def compare(actual:object,wanted:object)->float:
    if isinstance(actual,dict):
        assert set(actual)==set(wanted)
        return max((compare(actual[k],wanted[k]) for k in actual),default=0.)
    if isinstance(actual,list):
        assert len(actual)==len(wanted)
        return max((compare(a,b) for a,b in zip(actual,wanted)),default=0.)
    if isinstance(actual,float):
        error=abs(actual-wanted);assert error<1e-9,(actual,wanted);return error
    assert actual==wanted,(actual,wanted);return 0.
def main()->None:
    start=time.monotonic();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    contract=json.loads((AUTHOR/'preregistration.json').read_text());assert sha(AUTHOR/'probe.py')==contract['source_sha256'] and sha(SELECTION)==contract['selection_sha256']
    selected=json.loads(SELECTION.read_text());groups=selected['evaluation'];pf=pq.ParquetFile(ROOT/'datasets/train.parquet');columns=[f'i{inst}_{kind}{j}' for inst in range(2) for kind in ('p','v') for j in range(22)];competition=[]
    for member in groups:
        table=pf.read_row_group(member['row_group'],columns=columns,use_threads=False)
        for inst in range(2):
            for kind in ('p','v'):
                x=np.stack([table[f'i{inst}_{kind}{j}'].to_numpy() for j in range(22)],axis=1);assert x.shape==(20000,22) and x.dtype==np.float32
                competition.append({**member,'instrument':inst,'feature_group':kind,**describe(x)})
    digest=hashlib.sha256();rows=[]
    with MD.open() as stream:
        header=stream.readline();digest.update(header.encode());names=next(csv.reader([header]));indices=[names.index(f'{side}s[{j}].{kind}') for side in ('bid','ask') for j in range(11) for kind in ('price','amount')]
        for j in range(20000):
            line=stream.readline()
            if not line:break
            digest.update(line.encode());fields=next(csv.reader([line]));rows.append([float(fields[k]) if fields[k] else np.nan for k in indices])
    matrix=np.asarray(rows,dtype=np.float64);p=matrix[:,::2];v=matrix[:,1::2];valid=np.isfinite(matrix).all(1)&(p>0).all(1)&(v>=0).all(1);p=p[valid];v=v[valid];mid=(p[:,0]+p[:,11])*.5
    variants={'absolute_prices':p,'midpoint_centered_prices':p-mid[:,None],'midpoint_fractional_prices':p/mid[:,None]-1,'side_top_centered_prices':p-np.repeat(p[:,[0,11]],11,axis=1)}
    outside={'source':{'path':str(MD),'size_bytes':MD.stat().st_size,'mtime_ns':MD.stat().st_mtime_ns,'consumed_prefix_utf8_sha256':digest.hexdigest()},'snapshots_consumed':len(rows),'eligible_snapshots':len(p),'ineligible_snapshots':int((~valid).sum()),'price_variants':{name:describe(x) for name,x in variants.items()},'volume':describe(v)}
    expected=json.loads((AUTHOR/'results.json').read_text());error=max(compare(competition,expected['competition']),compare(outside,{k:expected['external'][k] for k in outside}))
    rng=np.random.default_rng(928);fixture=rng.integers(1,101,size=(1000,22)).astype(np.float64);before=describe(fixture);after=describe(fixture**3);assert compare(before,after)==0
    result={'status':'pass','source_sha256':sha(Path(__file__)),'author_source_sha256':sha(AUTHOR/'probe.py'),'author_results_sha256':sha(AUTHOR/'results.json'),'selection_sha256':sha(SELECTION),'competition':competition,'external':outside,'maximum_numeric_discrepancy':error,'all_rank_arithmetic':'float64 after rawf32 projection; external inputs float64','strict_monotone_tie_preserving_synthetic_identity':True,'targets_masks_VALID_models_read':False,'claim':'Descriptive fixed-prefix copula only; no source/depth/physical encoder identification or model gain. Row-dependent centering and clipping need not preserve ranks.','resources':{'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS)),'cpu_affinity':sorted(os.sched_getaffinity(0)),'hard_cgroup_claim':False}}
    with (HERE/'book_copula_independent_v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'pass','max_error':error,'groups':len(competition),'external_rows':len(p),'resources':result['resources']}))
if __name__=='__main__':main()
