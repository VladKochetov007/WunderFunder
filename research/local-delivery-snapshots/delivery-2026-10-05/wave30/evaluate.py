"""Oracle class3 baseline addback diagnostic, not an inference candidate."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R=Path.home()/'WunderFunder-creative'
D=R/'research/iterations/2026-10-05-wave30'
specs={};comparisons=[]
for label,prefix in (('old','w18compcompress_distilled'),('new','w24comp_distilled')):
    ref={f'{prefix}_s{s}':.5 for s in (3,4)}
    specs[label]=ref
    for method in ('add_all','add_c3','blend_all','blend_c3'):
        specs[label+'_'+method]={**ref,'orgbase':0.}
        comparisons.append((label+'_'+method,label))
    comparisons.extend([(label+'_blend_c3',label+'_blend_all'),(label+'_add_c3',label+'_add_all')])

def wp(s):
    w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
    return ((syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))).mean(-1)

def main():
    family=sys.argv[1] if len(sys.argv)>1 else 'all'
    assert family in ('all','complement')
    selected={k:v for k,v in specs.items() if family=='all' or k=='reference' or k.startswith('complement_')}
    pairs=[(a,b) for a,b in comparisons if a in selected and b in selected]
    suffix='' if family=='all' else '-'+family
    report_path=D/f'composite-report{suffix}.json'
    assert not report_path.exists()
    cf=np.load(R/'cache/chain_folds.npz')
    seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold'])
        if (s=='valid' and f!=0) or (s=='train' and int(hashlib.sha256(f'half:{g}'.encode()).hexdigest(),16)%2==0))
    hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy').tolist())
    names=list(selected);report={'specs':selected,'comparisons':pairs,'results':{},
        'scope':'ORACLE GATE diagnostic ONLY: whole-sequence label-derived market_class chooses class3; never a deployable prediction. Known student parents trainhalf0+valid!=0. Compare fixed add.025B or blend.9P+.025B (baseline at.25 scale) on allclasses vs oracleclass3. No parameters fitted. Reused valid436/129 and train282/191 diagnostic with estimated mask; organizer provenance uncertain; fitted-model conditional intervals. Positive evidence would require separate causal predicted-class replay and native validation.'}
    for split in ('valid','train'):
        z=np.load(R/f'cache/{split}_targets.npz'); y=z['y'];groups=z['row_group']
        cmap={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s==split}
        ids=[i for i,g in enumerate(groups) if (cmap[g][1]==0 and cmap[g][0] not in seen) ] if split=='valid' else [i for i,g in enumerate(groups) if g in hold]
        chains,inv=np.unique([cmap[groups[i]][0] for i in ids],return_inverse=True)
        mask=z['mask'] if split=='valid' else np.load(R/'cache/propcache_prop_f0_train.npy',mmap_mode='r')
        arrays={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for sp in selected.values() for n in sp}
        moments=np.zeros((len(names),len(chains),2,6))
        for ii,i in enumerate(ids):
            yy=np.clip(y[i],-2,2).astype('float64');q=np.array(mask[i if split=='valid' else groups[i]],dtype='float64');q[:99]=0
            w=np.abs(yy)*q[:,None]
            row={n:np.asarray(a[i],dtype='float64') for n,a in arrays.items()}
            for j,sp in enumerate(selected.values()):
                p=sum(c*row[n] for n,c in sp.items())
                name=names[j]
                apply=('_all' in name) or ('_c3' in name and int(z['market_class'][i])==3)
                if apply:
                    if '_blend_' in name:p=.9*p+.025*row['orgbase']
                    elif '_add_' in name:p=p+.025*row['orgbase']
                p=np.clip(p,-2,2)
                assert np.isfinite(p).all()
                moments[j,inv[ii]]+=np.stack([w.sum(0),(w*yy).sum(0),(w*p).sum(0),(w*yy*yy).sum(0),(w*p*p).sum(0),(w*yy*p).sum(0)],-1)
        score=wp(moments.sum(1));rng=np.random.default_rng(20261005)
        draws=np.array([wp(moments[:,rng.integers(len(chains),size=len(chains))].sum(1)) for _ in range(2000)])
        result={'sequences':len(ids),'chains':len(chains),'scores':dict(zip(names,map(float,score))),'comparisons':{}}
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b);dd=draws[:,ia]-draws[:,ib]
            result['comparisons'][a+' - '+b]={'delta':float(score[ia]-score[ib]),'conditional_ci95':np.quantile(dd,[.025,.975]).tolist()}
        report['results'][split+'_clean']=result
        np.savez_compressed(D/f'composite-moments{suffix}-{split}.npz',moments=moments,names=names,chains=chains)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
