"""Frozen train-only clock feasibility audit. No predictor fitting or WP queries."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
import argparse, collections, datetime, hashlib, json, pickle, resource, signal, sys, time
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keys(x): return [tuple(r) for r in np.round(x[:,:22],4)]
def proxy(x): return np.r_[False,(x[1:,0]!=x[:-1,0])&(x[1:,11]!=x[:-1,11])]
def modal(c):
    jump,n=c.most_common(1)[0]
    return (-int(jump[0]),int(jump[1])),n/sum(c.values())
def clocks(x,moves,spreads):
    p=proxy(x); code=np.r_[False,np.any(x[1:,:22]!=x[:-1,:22],axis=1)]; k=keys(x)
    known=np.zeros(len(x),bool); da=np.zeros(len(x),np.int64); db=da.copy()
    for e in np.flatnonzero(code):
        mv=moves.get((k[e-1],k[e]))
        if mv is not None: known[e]=True;da[e],db[e]=mv
    quote=(da!=0)|(db!=0); candidate=np.where(known,quote,p)
    a=np.cumsum(da); b=np.cumsum(db)
    for i,key in enumerate(k):
        want=spreads.get(key,-1)
        if want>=0: b[i]=a[i]-want
        elif i>0: b[i]=b[i-1]+db[i]
        else: b[i]=a[i]-1
    reconstructed=np.r_[False,(np.diff(a)!=0)|(np.diff(b)!=0)]
    return dict(proxy=p,code=code,known=known,quote=quote,candidate=candidate,reconstructed=reconstructed,keys=k)
def streaming_mask(x,moves,chunk=1):
    out=[];prev=None
    for start in range(0,len(x),chunk):
        for row in x[start:start+chunk]:
            if prev is None: out.append(False)
            else:
                pp=bool(row[0]!=prev[0] and row[11]!=prev[11]); changed=bool(np.any(row[:22]!=prev[:22]))
                mv=moves.get((tuple(np.round(prev[:22],4)),tuple(np.round(row[:22],4)))) if changed else None
                out.append(bool(mv[0] or mv[1]) if mv is not None else pp)
            prev=row
    return np.asarray(out,bool)
def qg(mask):
    gaps=np.diff(np.r_[0,np.flatnonzero(mask)])
    return dict(count=int(len(gaps)),quantiles={str(q):float(np.quantile(gaps,q)) if len(gaps) else None for q in (.5,.9,.99,1.)})
def counts(c,trans=None):
    p,k,q,ev=c['proxy'],c['known'],c['quote'],c['code']
    r=dict(rows_excluding_reset=len(p)-1,code_events=int(ev.sum()),proxy_events=int(p.sum()),known_events=int(k.sum()),unknown_events=int((ev&~k).sum()),known_shape_events=int((k&~q).sum()),known_quote_events=int((k&q).sum()),proxy_known_shape=int((p&k&~q).sum()),quote_missed_by_proxy=int((~p&k&q).sum()),proxy_and_quote=int((p&k&q).sum()),neither_on_known=int((~p&k&~q).sum()),changed_clock_rows=int((p!=c['candidate']).sum()),direct_vs_spread_on_known=int(((q!=c['reconstructed'])&k).sum()),spread_quote_on_no_code=int((c['reconstructed']&~ev).sum()),rounding_alias_events=sum(c['keys'][i-1]==c['keys'][i] for i in np.flatnonzero(ev)))
    if trans is not None:
        supp=amb=0
        for e in np.flatnonzero(k):
            co=trans[(c['keys'][e-1],c['keys'][e])]; purity=co.most_common(1)[0][1]/sum(co.values())
            supp+=int(sum(co.values())>=20 and purity>=.95);amb+=int(purity<.95)
        r.update(supported_known_events=supp,ambiguous_known_events=amb)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    start=time.monotonic(); plan=json.loads(args.plan.read_text());root=args.root
    sys.path.insert(0,str(root/'src'));from wnn.data import SequenceFile
    def timeout(*_):raise TimeoutError('480-second diagnostic limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(480)
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(args.plan),code_sha256=sha(__file__),processed_fit=[],processed_audit=[],sequences=[],validation={},input_hashes={})
    try:
        expected={root/'cache/chain_folds.npz':plan['inputs']['chain_folds_sha256'],root/'research/artifacts/market-classes-v1.npz':plan['inputs']['classes_sha256'],root/'cache/class3_moves_full.pkl':plan['inputs']['descriptive_cached_table_sha256']}
        for path,h in expected.items():
            actual=sha(path);assert actual==h,str(path);result['input_hashes'][str(path)]=actual
        data=root/'datasets/train.parquet';s=data.stat();result['train_data_stat']=dict(path=str(data),bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
        for name in ('src/wnn/data.py','src/wnn/ticks.py','src/wnn/features.py'):
            result['input_hashes'][name]=sha(root/name)
        src=SequenceFile(data);trans=collections.defaultdict(collections.Counter);spreadcounts=collections.defaultdict(collections.Counter)
        ids=plan['sampling']['frozen_ids'];assert not ({r['chain'] for r in ids['fit']}&{r['chain'] for r in ids['audit']})
        for row in ids['fit']:
            seq=src.read(row['row_group'],targets=True,mask=False);x,y=seq.features,seq.targets
            k=keys(x);ev=np.flatnonzero(np.any(x[1:,:22]!=x[:-1,:22],axis=1))+1;q=np.rint(y/.65687).astype(np.int64)
            for e in ev:trans[(k[e-1],k[e])][tuple(map(int,q[e]-q[e-1]))]+=1
            sp=-np.rint((y[:,0]+y[:,1])/.65687).astype(np.int64)
            for i in range(0,len(k),7):spreadcounts[k[i]][int(sp[i])]+=1
            result['processed_fit'].append(row)
        fresh={key:modal(co)[0] for key,co in trans.items()};spreads={key:co.most_common(1)[0][0] for key,co in spreadcounts.items()}
        with open(root/'cache/class3_moves_full.pkl','rb') as f:cached,cspreads=pickle.load(f)
        overlap=fresh.keys()&cached.keys();disagreements=sum(fresh[k]!=cached[k] for k in overlap)
        result['fit_summary']=dict(sequences=len(ids['fit']),keys=len(fresh),spread_codes=len(spreads),events=sum(sum(co.values()) for co in trans.values()),overlap_cached_keys=len(overlap),different_cached_keys=disagreements,keys_support_under20=sum(sum(co.values())<20 for co in trans.values()),keys_purity_under95=sum(modal(co)[1]<.95 for co in trans.values()))
        table=dict(plan_sha256=result['plan_sha256'],fit_ids=ids['fit'],moves=[dict(previous=list(map(float,k[0])),current=list(map(float,k[1])),move=v,support=sum(trans[k].values()),modal_purity=modal(trans[k])[1],jump_counts=[dict(jump=list(j),count=n) for j,n in trans[k].items()]) for k,v in fresh.items()],spreads=[dict(code=list(map(float,k)),spread=int(v)) for k,v in spreads.items()])
        (args.output/'architecture-fit-table.json').write_text(json.dumps(table)+'\n')
        parity=[]
        for row in ids['audit']:
            x=src.read(row['row_group'],targets=False,mask=False).features
            ff=clocks(x,fresh,spreads);cc=clocks(x,cached,cspreads)
            rr=dict(**row,fresh=counts(ff,trans),cached_descriptive=counts(cc),gap_proxy=qg(ff['proxy']),gap_fresh=qg(ff['candidate']),gap_cached_descriptive=qg(cc['candidate']))
            # Two implementations and fixed chunk boundaries; no targets in either transform.
            stream=streaming_mask(x,fresh,1);chunks=streaming_mask(x,fresh,500)
            checks=dict(stream_equals_batch=bool(np.array_equal(stream,ff['candidate'])),chunk_equals_batch=bool(np.array_equal(chunks,ff['candidate'])),prefixes={str(cut):bool(np.array_equal(clocks(x[:cut],fresh,spreads)['candidate'],ff['candidate'][:cut])) for cut in (1,99,500,5000,19999)})
            parity.append(checks);rr['causality_checks']=checks;result['sequences'].append(rr);result['processed_audit'].append(row)
        result['by_class']={}
        for cl in range(4):
            rr=[r for r in result['sequences'] if r['market_class']==cl]
            result['by_class'][str(cl)]={which:{k:sum(r[which][k] for r in rr) for k in rr[0][which]} for which in ('fresh','cached_descriptive')}
        # Inconsistent direct db=0 versus code spread narrowing necessarily gives reconstructed bid move.
        toy=np.zeros((2,22),np.float32);toy[1]=1;k=keys(toy);tc=clocks(toy,{(k[0],k[1]):(0,0)},{k[0]:2,k[1]:1})
        result['validation']=dict(all_prefix_chunk_stream_pass=all(r['stream_equals_batch'] and r['chunk_equals_batch'] and all(r['prefixes'].values()) for r in parity),toy_direct_move=bool(tc['quote'][1]),toy_spread_reconstructed_move=bool(tc['reconstructed'][1]),toy_shows_distinct_semantics=bool(tc['quote'][1]!=tc['reconstructed'][1]))
        result['status']='complete'
    except Exception as e:
        result['status']='timeout' if isinstance(e,TimeoutError) else 'error';result['error']=repr(e)
    finally:
        signal.alarm(0);result['wall_seconds']=time.monotonic()-start;result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output/'architecture-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k) for k in ('status','wall_seconds','peak_rss_kib','fit_summary','by_class','validation','error')}),flush=True)
    return result['status']!='complete'
if __name__=='__main__':sys.exit(main())
