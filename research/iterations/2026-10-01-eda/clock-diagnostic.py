"""CLOCK2: binary partition reproducibility; no forecast fitting, labels only on fit chains."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
import argparse, collections, datetime, hashlib, json, resource, signal, sys, time
from pathlib import Path
import numpy as np

BOOT=1000

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def statfp(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def sha_arrays(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def ratio(n,d):return float(n/d) if d else None
def ratio_mean(n,d):
    n,d=np.broadcast_arrays(np.asarray(n),np.asarray(d));ok=d>0
    return float(np.mean(n[ok]/d[ok])) if np.any(ok) else None
def ci(n,d):
    n,d=np.broadcast_arrays(np.asarray(n),np.asarray(d))
    ok=d>0
    return np.quantile(n[ok]/d[ok],[.025,.975]).tolist() if np.any(ok) else None

def codes(x):return [tuple(v) for v in np.round(x[:,:22],4)]
def flags(x):return np.r_[False,np.any(x[1:,:22]!=x[:-1,:22],axis=1)],np.r_[False,(x[1:,0]!=x[:-1,0])&(x[1:,11]!=x[:-1,11])]
def mask(x,table):
    ev,p=flags(x);kk=codes(x);out=p.copy()
    for e in np.flatnonzero(ev):
        v=table.get((kk[e-1],kk[e]))
        if v is not None:out[e]=bool(v[0] or v[1])
    return out

def stream_mask(x,table,chunk):
    out=[];prev=None
    for start in range(0,len(x),chunk):
        for row in x[start:start+chunk]:
            if prev is None:out.append(False)
            else:
                p=bool(row[0]!=prev[0] and row[11]!=prev[11]);v=None
                if np.any(row[:22]!=prev[:22]):v=table.get((tuple(np.round(prev[:22],4)),tuple(np.round(row[:22],4))))
                out.append(bool(v[0] or v[1]) if v is not None else p)
            prev=row
    return np.asarray(out,bool)

class RawAliases:
    def __init__(self):self.entries={};self.aliases={};self.raw_states=0
    def update(self,x):
        raw=np.ascontiguousarray(x[:,:22]);view=raw.view(np.dtype((np.void,88))).ravel();_,idx=np.unique(view,return_index=True)
        rounded=np.round(raw[idx],4)
        for rr,ii in zip(rounded,idx):
            key=rr.tobytes();value=raw[ii].tobytes();first=self.entries.get(key)
            if first is None:self.entries[key]=value;self.raw_states+=1
            elif value!=first:
                seen=self.aliases.setdefault(key,{first})
                if value not in seen:seen.add(value);self.raw_states+=1
    def summary(self):return dict(rounded_bit_keys=len(self.entries),distinct_raw_bit_states=self.raw_states,rounded_keys_with_multiple_raw_states=len(self.aliases),extra_raw_states_collapsed=sum(len(v)-1 for v in self.aliases.values()),note='Bit-signature alias diagnostic; numeric tuple keys equate signed zeros. Exact-change rows collapsing under tuple rounding are counted separately.')

def fit(rows,source,aliases,progress,label):
    trans={};chain_counts=[];chain_support=collections.Counter();identities=[];rounding_alias_events=0
    for row in rows:
        cc=collections.Counter()
        for rg in row['row_groups']:
            seq=source.read(rg,targets=True,mask=False);x,y=seq.features,seq.targets
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite fit data')
            aliases.update(x);ev,_=flags(x);idx=np.flatnonzero(ev)
            r=np.round(x[:,:22],4);q=np.rint(y/.65687).astype(np.int64)
            for e in idx:
                key=(tuple(r[e-1]),tuple(r[e]));rounding_alias_events+=int(key[0]==key[1]);jump=tuple(map(int,q[e]-q[e-1]))
                trans.setdefault(key,collections.Counter())[jump]+=1;cc[(key,jump)]+=1
            identities.append(dict(chain=row['chain'],row_group=rg,used_sha256=sha_arrays(x,y)))
        chain_counts.append(cc);chain_support.update({k for k,j in cc})
        progress.append(row['chain'])
        if len(progress)%32==0:print(json.dumps(dict(phase=label,completed_chains=len(progress))),flush=True)
    table={k:(-co.most_common(1)[0][0][0],co.most_common(1)[0][0][1]) for k,co in trans.items()}
    return dict(trans=trans,chain_counts=chain_counts,chain_support=chain_support,table=table,identities=identities,rounding_alias_events=rounding_alias_events)

def state(table,union):
    known=np.zeros((1,len(union)),bool);da=np.zeros((1,len(union)),np.int64);db=da.copy()
    for j,k in enumerate(union):
        v=table.get(k)
        if v is not None:known[0,j]=True;da[0,j],db[0,j]=v
    return dict(known=known,quote=(da!=0)|(db!=0),da=da,db=db)

def boot_state(model,weights,union,index):
    # FIRST modal signed jump; only THEN classify zero/nonzero. Never collapse categories first.
    columns={};bounds=[];outcomes=[]
    for key,co in model['trans'].items():
        start=len(outcomes)
        for jump in co:columns[(key,jump)]=len(outcomes);outcomes.append(jump)
        bounds.append((key,start,len(outcomes)))
    m=np.zeros((len(model['chain_counts']),len(columns)),np.float64)
    for i,cc in enumerate(model['chain_counts']):
        for keyjump,n in cc.items():m[i,columns[keyjump]]=n
    totals=weights.astype(np.float64)@m
    known=np.zeros((BOOT,len(union)),bool);da=np.zeros((BOOT,len(union)),np.int64);db=da.copy();ties=np.zeros_like(known)
    jumps=np.asarray(outcomes,np.int64)
    for key,a,b in bounds:
        block=totals[:,a:b];best=np.argmax(block,axis=1);mx=block[np.arange(BOOT),best];j=index[key]
        known[:,j]=mx>0;ties[:,j]=(mx>0)&(np.sum(block==mx[:,None],axis=1)>1)
        chosen=jumps[a+best];da[:,j]=-chosen[:,0];db[:,j]=chosen[:,1]
    return dict(known=known,quote=(da!=0)|(db!=0),da=da,db=db,ties=ties)

def candidate(state,proxy):return np.where(state['known'],state['quote'],proxy)
def metric_arrays(counts,totals,a,b):
    # counts[...,0/1] = code-event counts split by observed proxy bit; no-code rows never update.
    c0,c1=counts[:,:,0],counts[:,:,1];ct=c0+c1
    a0,a1=candidate(a,False),candidate(a,True);b0,b1=candidate(b,False),candidate(b,True)
    xor=(c0*(a0^b0)+c1*(a1^b1)).sum(1)
    other_proxy=totals[:,2]-c1.sum(1)
    union=(c0*(a0|b0)+c1*(a1|b1)).sum(1)+other_proxy
    both=a['known']&b['known'];known_den=(ct*both).sum(1)
    signed=(ct*both*((a['da']!=b['da'])|(a['db']!=b['db']))).sum(1)
    binary=(ct*both*(a['quote']^b['quote'])).sum(1)
    return dict(xor_per_row=(xor,totals[:,0]),xor_per_code_event=(xor,totals[:,1]),xor_per_union_update=(xor,union),known_both_binary=(binary,known_den),known_both_signed=(signed,known_den))

def pair_report(counts,totals,w,a,b,ab,bb):
    c=counts.sum(0,keepdims=True);t=totals.sum(0,keepdims=True)
    bootc=(w.astype(np.float64)@counts.reshape(len(counts),-1)).reshape(BOOT,counts.shape[1],2);boott=w.astype(np.float64)@totals
    point=metric_arrays(c,t,a,b);conditional=metric_arrays(bootc,boott,a,b);joint=metric_arrays(bootc,boott,ab,bb)
    report={}
    for k,(n,d) in point.items():
        report[k]=dict(numerator=int(n[0]),denominator=int(d[0]),estimate=ratio(n[0],d[0]),conditional_audit_chain_ci95=ci(*conditional[k]),joint_fit_and_audit_chain_ci95=ci(*joint[k]),conditional_zero_denominator_draws=int(np.sum(conditional[k][1]<=0)),joint_zero_denominator_draws=int(np.sum(joint[k][1]<=0)))
    return report,bootc,boott

def quality(model,union):
    row=np.zeros(len(union),np.int64);chains=row.copy();purity=np.zeros(len(union));ties=np.zeros(len(union),bool)
    for j,k in enumerate(union):
        if k in model['trans']:
            co=model['trans'][k];vals=list(co.values());row[j]=sum(vals);chains[j]=model['chain_support'][k];purity[j]=max(vals)/sum(vals);ties[j]=vals.count(max(vals))>1
    return dict(row_support=row,chain_support=chains,purity=purity,ties=ties)

def strata(a,b,qa,qb):
    ak,bk=a['known'][0],b['known'][0];both=ak&bk
    low=(qa['row_support']<20)|(qb['row_support']<20)|(qa['chain_support']<3)|(qb['chain_support']<3)
    impure=(qa['purity']<.95)|(qb['purity']<.95)
    return dict(both_supported_pure=both&~low&~impure,both_low_support=both&low,both_supported_low_purity=both&~low&impure,only_A=ak&~bk,only_B=~ak&bk,neither=~ak&~bk)

def qsummary(x):return {str(q):float(np.quantile(x,q)) if len(x) else None for q in (.1,.5,.9,.99,1.)}
def intervals(x,events):
    idx=np.flatnonzero(events);gap=np.diff(idx);lo,hi=idx[:-1],idx[1:]
    block={};bins=np.searchsorted([1,4,16,64],gap,side='left')
    for name,sl in [('i0_volume_code',slice(22,44)),('i0_trade_code',slice(44,52)),('i1_feature_code',slice(52,104))]:
        changed=np.r_[False,np.any(x[1:,sl]!=x[:-1,sl],axis=1)];cum=np.cumsum(changed)
        interior=cum[hi-1]-cum[lo]
        block[name]=dict(updates=int(interior.sum()),nonempty=int((interior>=1).sum()),at_least4=int((interior>=4).sum()),by_gap_bin=[dict(intervals=int((bins==j).sum()),updates=int(interior[bins==j].sum()),nonempty=int((interior[bins==j]>=1).sum())) for j in range(5)])
    return dict(closed_intervals=len(gap),gap=gap,blocks=block,prefix_before_first_event=int(idx[0]) if len(idx) else len(x)-1,right_censored_tail=int(len(x)-1-idx[-1]) if len(idx) else len(x)-1,sequence_without_events=int(not len(idx)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    started=time.monotonic();plan=json.loads(args.plan.read_text());root=args.root;args.output.mkdir(exist_ok=False,parents=True)
    result=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=digest(args.plan),script_sha256=digest(__file__),processed_fit_A=[],processed_fit_B=[],processed_audit=[],checks={},by_class={})
    def stop(*_):raise TimeoutError('CLOCK2 time budget')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(870)
    try:
        sys.path.insert(0,str(root/'src'));from wnn.data import SequenceFile
        import pickle
        files={'chain_folds':root/'cache/chain_folds.npz','classes':root/'research/artifacts/market-classes-v1.npz','historical_table':root/'cache/class3_moves_full.pkl'}
        hashes={k:digest(p) for k,p in files.items()}
        for k,h in hashes.items():assert h==plan['inputs'][k+'_sha256'],k
        result['input_hashes']=hashes;result['source_hashes']={n:digest(root/n) for n in ('src/wnn/data.py','src/wnn/ticks.py','src/wnn/features.py')}
        parquet=root/'datasets/train.parquet';before=statfp(parquet);result['data_before']=before;source=SequenceFile(parquet)
        cf=np.load(files['chain_folds']);cls=np.load(files['classes'])['train'];forbidden=set(map(int,cf['chain'][cf['split']=='valid']))
        actual=collections.defaultdict(list)
        for rg,ch in zip(cf['row_group'][cf['split']=='train'],cf['chain'][cf['split']=='train']):actual[int(ch)].append(int(rg))
        sample=plan['sampling']['resolved'];allsets=[]
        for role in ('fit_A','fit_B','audit'):
            ss={r['chain'] for r in sample[role]};allsets.append(ss)
            assert not ss&forbidden and not ss&set(sample['excluded_prior_chains'])
            for r in sample[role]:
                assert sorted(actual[r['chain']])==r['row_groups'] and 528 not in r['row_groups']
                assert np.all(cls[r['row_groups']]==r['market_class'])
        assert not (allsets[0]&allsets[1] or allsets[0]&allsets[2] or allsets[1]&allsets[2]);result['checks']['chain_partition']=True
        aliases_fitA,aliases_fitB,aliases_audit=RawAliases(),RawAliases(),RawAliases()
        A=fit(sample['fit_A'],source,aliases_fitA,result['processed_fit_A'],'fit_A');B=fit(sample['fit_B'],source,aliases_fitB,result['processed_fit_B'],'fit_B')
        with files['historical_table'].open('rb') as f:H,_=pickle.load(f)
        union=list(dict.fromkeys([*A['table'],*B['table'],*H]));index={k:i for i,k in enumerate(union)};K=len(union)
        sa,sb,sh=state(A['table'],union),state(B['table'],union),state(H,union);qa,qb=quality(A,union),quality(B,union)
        result['fit_summary']={}
        for name,model,qual in [('A',A,qa),('B',B,qb)]:
            result['fit_summary'][name]=dict(chains=128,sequences=len(model['identities']),transition_keys=len(model['table']),events=sum(sum(c.values()) for c in model['trans'].values()),tied_keys=int(qual['ties'].sum()),keys_support_under20=sum(sum(c.values())<20 for c in model['trans'].values()),rounding_alias_events=model['rounding_alias_events'])
        result['table_key_comparisons']={}
        for name,ta,tb in [('A_B',A['table'],B['table']),('A_H_descriptive',A['table'],H),('B_H_descriptive',B['table'],H)]:
            common=ta.keys()&tb.keys();result['table_key_comparisons'][name]=dict(common_keys=len(common),signed_different=sum(ta[k]!=tb[k] for k in common),binary_different=sum(bool(ta[k][0] or ta[k][1])!=bool(tb[k][0] or tb[k][1]) for k in common))
        counts=np.zeros((len(sample['audit']),K,2),np.int64);totals=np.zeros((len(sample['audit']),3),np.int64);chain_info=[];audit_ids=[];proxy_patterns=np.zeros(K,np.uint8);direct_xor=[]
        checked_classes=set()
        for ci_,row in enumerate(sample['audit']):
            rec=dict(chain=row['chain'],market_class=row['market_class'],row_groups=row['row_groups'],rows=0,code_events=0,proxy_events=0,rounding_alias_events=0,known_A=0,known_B=0,known_H=0,events_A=0,events_B=0,events_H=0,drop_A=0,drop_B=0,drop_H=0,miss_A=0,miss_B=0,miss_H=0,xor_A_B=0,xor_A_H=0,xor_B_H=0,closed_intervals=0,prefix_rows=0,censored_tail_rows=0,sequences_no_A_events=0,interval_blocks={})
            gaps={name:[] for name in ('proxy','A','B','H')}
            for rg in row['row_groups']:
                seq=source.read(rg,targets=False,mask=False);assert seq.targets is None and seq.scored is None;x=seq.features
                if not np.isfinite(x).all():raise ValueError('nonfinite audit data')
                aliases_audit.update(x);audit_ids.append(dict(chain=row['chain'],row_group=rg,feature_sha256=sha_arrays(x)))
                ev,p=flags(x);kk=codes(x);masks=[p.copy(),p.copy(),p.copy()];known=[0,0,0]
                for e in np.flatnonzero(ev):
                    key=(kk[e-1],kk[e]);j=index.get(key)
                    rec['rounding_alias_events']+=int(kk[e-1]==kk[e])
                    if j is not None:counts[ci_,j,int(p[e])]+=1;proxy_patterns[j]|=1<<int(p[e])
                    for z,table in enumerate((A['table'],B['table'],H)):
                        v=table.get(key)
                        if v is not None:known[z]+=1;masks[z][e]=bool(v[0] or v[1])
                rec['rows']+=len(x)-1;rec['code_events']+=int(ev.sum());rec['proxy_events']+=int(p.sum())
                for z,name in enumerate(('A','B','H')):
                    mm=masks[z];rec['known_'+name]+=known[z];rec['events_'+name]+=int(mm.sum());rec['drop_'+name]+=int((p&~mm).sum());rec['miss_'+name]+=int((~p&mm).sum());gaps[name].extend(np.diff(np.flatnonzero(mm)).tolist())
                gaps['proxy'].extend(np.diff(np.flatnonzero(p)).tolist());rec['xor_A_B']+=int((masks[0]^masks[1]).sum());rec['xor_A_H']+=int((masks[0]^masks[2]).sum());rec['xor_B_H']+=int((masks[1]^masks[2]).sum())
                eda=intervals(x,masks[0]);rec['closed_intervals']+=eda['closed_intervals'];rec['prefix_rows']+=eda['prefix_before_first_event'];rec['censored_tail_rows']+=eda['right_censored_tail'];rec['sequences_no_A_events']+=eda['sequence_without_events']
                for bn,bv in eda['blocks'].items():
                    dest=rec['interval_blocks'].setdefault(bn,dict(updates=0,nonempty=0,at_least4=0,by_gap_bin=[dict(intervals=0,updates=0,nonempty=0) for _ in range(5)]))
                    for k in ('updates','nonempty','at_least4'):dest[k]+=bv[k]
                    for j in range(5):
                        for k in ('intervals','updates','nonempty'):dest['by_gap_bin'][j][k]+=bv['by_gap_bin'][j][k]
                if row['market_class'] not in checked_classes:
                    checks=[]
                    for table,mm in zip((A['table'],B['table']),masks):
                        checks.append(all(np.array_equal(mask(x[:cut],table),mm[:cut]) for cut in (1,99,500,5000,19999)) and np.array_equal(stream_mask(x,table,1),mm) and np.array_equal(stream_mask(x,table,500),mm))
                    assert all(checks);checked_classes.add(row['market_class'])
            totals[ci_]=[rec['rows'],rec['code_events'],rec['proxy_events']];rec['gap_quantiles']={k:qsummary(v) for k,v in gaps.items()};rec['drop_fraction_A']=ratio(rec['drop_A'],rec['proxy_events']);rec['drop_fraction_B']=ratio(rec['drop_B'],rec['proxy_events']);rec['event_rate_A']=ratio(rec['events_A'],rec['rows']);rec['miss_fraction_A']=ratio(rec['miss_A'],rec['events_A']);chain_info.append(rec);result['processed_audit'].append(row['chain'])
            if len(chain_info)%16==0:print(json.dumps(dict(phase='audit',completed_chains=len(chain_info))),flush=True)
        result['checks']['prefix_stream_chunk']=True;result['raw_aliases']={'fit_A':aliases_fitA.summary(),'fit_B':aliases_fitB.summary(),'audit':aliases_audit.summary(),'audit_union_transition_keys_with_both_proxy_bits':int((proxy_patterns==3).sum())}
        # Release large raw-state dictionaries before dense bootstrap matrices.
        del aliases_fitA,aliases_fitB,aliases_audit
        result['chain_readouts']=chain_info;save(args.output/'clock-used-inputs.json',dict(fit_A=A['identities'],fit_B=B['identities'],audit=audit_ids))
        rng=np.random.default_rng(20261001);wa=rng.multinomial(128,np.full(128,1/128),size=BOOT);wb=rng.multinomial(128,np.full(128,1/128),size=BOOT)
        ba=boot_state(A,wa,union,index);bb=boot_state(B,wb,union,index)
        classes=np.array([r['market_class'] for r in chain_info]);fixed_strata=strata(sa,sb,qa,qb)
        for cl in range(4):
            select=np.flatnonzero(classes==cl);ct=counts[select];tt=totals[select];w=rng.multinomial(len(select),np.full(len(select),1/len(select)),size=BOOT)
            reports={};bootc=boott=None
            for name,a,b,ab,bb_ in [('A_B',sa,sb,ba,bb),('A_H_descriptive',sa,sh,ba,sh),('B_H_descriptive',sb,sh,bb,sh)]:
                reports[name],bootc,boott=pair_report(ct,tt,w,a,b,ab,bb_)
                expected=sum(chain_info[i]['xor_'+name.split('_descriptive')[0]] for i in select)
                assert reports[name]['xor_per_row']['numerator']==expected
            core=ct.sum(0);ctotal=core.sum(1);ss={};c0,c1=core[:,0],core[:,1]
            xor_key=c0[None]*(candidate(sa,False)^candidate(sb,False))+c1[None]*(candidate(sa,True)^candidate(sb,True))
            both=sa['known']&sb['known'];signed=(sa['da']!=sb['da'])|(sa['db']!=sb['db'])
            for name,sel in fixed_strata.items():
                den=float(ctotal[sel].sum());num=float(xor_key[0,sel].sum())
                dn=(bootc.sum(2)*sel).sum(1)
                if name=='neither':
                    outside=tt[:,1]-ct.sum((1,2));den+=float(outside.sum());dn+=w@outside
                nn=(bootc[:,:,0]*(candidate(sa,False)^candidate(sb,False))*sel+bootc[:,:,1]*(candidate(sa,True)^candidate(sb,True))*sel).sum(1)
                ss[name]=dict(events=int(den),binary_xor=int(num),binary_disagreement=ratio(num,den),conditional_chain_ci95=ci(nn,dn),known_signed_disagreements=int((ctotal*sel*both[0]*signed[0]).sum()))
            assert sum(v['events'] for v in ss.values())==int(tt[:,1].sum())
            extras={};gates={};interval_reports={}
            for name,st,bs in [('A',sa,ba),('B',sb,bb)]:
                drop=np.array([chain_info[i]['drop_'+name] for i in select]);proxy_n=tt[:,2]
                extras[name]=dict(known_events=sum(chain_info[i]['known_'+name] for i in select),total_code_events=int(tt[:,1].sum()),drop_proxy_numerator=int(drop.sum()),drop_proxy_denominator=int(proxy_n.sum()),drop_proxy_fraction=ratio(drop.sum(),proxy_n.sum()),drop_proxy_chain_ci95=ci(w@drop,w@proxy_n),equal_chain_drop_quantiles=qsummary(drop[proxy_n>0]/proxy_n[proxy_n>0]),equal_chain_drop_valid_chains=int((proxy_n>0).sum()),equal_chain_drop_undefined_chains=int((proxy_n<=0).sum()),audit_event_weighted_bootstrap_tie_probability=ratio((bs['ties']*ctotal).sum(),BOOT*(ctotal*st['known'][0]).sum()),audit_event_weighted_bootstrap_absent_probability=ratio(((~bs['known'])*ctotal*st['known']).sum(),BOOT*(ctotal*st['known'][0]).sum()))
                flips=metric_arrays(core[None],tt.sum(0,keepdims=True),st,bs)['xor_per_code_event'];extras[name]['fit_bootstrap_only_candidate_flip_mean']=ratio_mean(*flips);extras[name]['fit_bootstrap_only_candidate_flip_ci95']=ci(*flips)
                gates[name]=dict(chains_with_drop_at_least_1pct=int(np.sum((proxy_n>0)&(drop>=.01*proxy_n))),chains=len(select))
            ints=np.array([chain_info[i]['closed_intervals'] for i in select],float)
            for block in ('i0_volume_code','i0_trade_code','i1_feature_code'):
                data={k:np.array([chain_info[i]['interval_blocks'][block][k] for i in select],float) for k in ('updates','nonempty','at_least4')}
                interval_reports[block]={k:dict(numerator=int(v.sum()),closed_intervals=int(ints.sum()),estimate=ratio(v.sum(),ints.sum()),conditional_chain_ci95=ci(w@v,w@ints),equal_chain_ratio_quantiles=qsummary(v[ints>0]/ints[ints>0]),equal_chain_ratio_valid_chains=int((ints>0).sum()),equal_chain_ratio_undefined_chains=int((ints<=0).sum())) for k,v in data.items()}
                interval_reports[block]['by_gap_bin']=[{k:sum(chain_info[i]['interval_blocks'][block]['by_gap_bin'][j][k] for i in select) for k in ('intervals','updates','nonempty')} for j in range(5)]
            result['by_class'][str(cl)]=dict(chains=len(select),sequences=sum(len(chain_info[i]['row_groups']) for i in select),rows=int(tt[:,0].sum()),code_events=int(tt[:,1].sum()),proxy_events=int(tt[:,2].sum()),pairs=reports,support_strata=ss,fresh_clock_summaries=extras,materiality_chain_counts=gates,interval_EDA_fixed_A=interval_reports)
        result['checks']['sufficient_statistics_match_direct_scan']=True
        keyrecords=[]
        for j,key in enumerate(union):
            if not counts[:,j].sum():continue
            rec=dict(previous=list(map(float,key[0])),current=list(map(float,key[1])),audit_events=int(counts[:,j].sum()),audit_proxy_bits=int(proxy_patterns[j]))
            for name,model,qual,bs in [('A',A,qa,ba),('B',B,qb,bb)]:
                present=bs['known'][:,j];rec[name]=dict(known=key in model['table'],signed_move=model['table'].get(key),row_support=int(qual['row_support'][j]),chain_support=int(qual['chain_support'][j]),modal_purity=float(qual['purity'][j]),original_tie=bool(qual['ties'][j]),bootstrap_absent_probability=float(np.mean(~present)),bootstrap_nonzero_probability_given_present=float(np.mean(bs['quote'][present,j])) if present.any() else None,bootstrap_tie_probability=float(np.mean(bs['ties'][:,j])))
            keyrecords.append(rec)
        save(args.output/'clock-audit-key-stability.json',keyrecords)
        for name,model in [('A',A),('B',B)]:
            save(args.output/('clock-fit-'+name+'.json'),dict(fit_chains=sample['fit_'+name],moves=[dict(previous=list(map(float,key[0])),current=list(map(float,key[1])),move=model['table'][key],chain_support=model['chain_support'][key],jump_counts=[dict(jump=list(j),count=n) for j,n in co.items()]) for key,co in model['trans'].items()]))
        result['data_after']=statfp(parquet);assert before==result['data_after'];result['checks']['data_fingerprint_stable']=True
        c3=result['by_class']['3'];interval=c3['pairs']['A_B']['xor_per_code_event']['joint_fit_and_audit_chain_ci95'];upper=interval[1] if interval is not None else None
        material=all(v['chains_with_drop_at_least_1pct']>=.9*v['chains'] for v in c3['materiality_chain_counts'].values())
        result['decision']=dict(binary_partition_descriptor_pass=bool(upper is not None and upper<=.01 and material),joint_class3_xor_code_event_ci_upper=upper,both_maps_drop_ge1pct_on_ge90pct_chains=material,interpretation='Descriptive learned-partition stability only; old signed-key gate remains failed; no physical quote-truth, forecast improvement or training authorization.')
        result['status']='complete'
    except Exception as ex:
        import traceback
        result['status']='timeout' if isinstance(ex,TimeoutError) else 'error';result['error']=repr(ex);result['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0);result['wall_seconds']=time.monotonic()-started;result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(args.output/'clock-result.json',result)
        print(json.dumps({k:result.get(k) for k in ('status','wall_seconds','peak_rss_kib','fit_summary','decision','error')}),flush=True)
    return result['status']!='complete'
if __name__=='__main__':sys.exit(main())
