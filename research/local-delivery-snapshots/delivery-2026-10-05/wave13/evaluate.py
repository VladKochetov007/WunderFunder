"""Frozen paired comparisons: sentinel age vs padded control; complement vs direct FT."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R=Path.home()/'WunderFunder-creative'
D=R/'research/iterations/2026-10-05-wave13'
base={'d256bc1k_f0':.3,'cmp_fullft_s3':.1,'cmp_fullft_s4':.1,'cmp_tstream_s3':.125,'cmp_tstream_s4':.125}
specs={'cached':{**base,'orgbase':.0625},'cached32':{**base,'orgbase32':.0625},'frozen_torch':{**base,'w13frozen':.0625}}
for kind in ('fp32','half','int8','weight8','oldhalf'):
    specs['native_'+kind]={**base,'w13native_'+kind:.0625}
comparisons=[('cached32','cached'),('frozen_torch','cached'),('native_fp32','frozen_torch'),('native_oldhalf','cached')]
for kind in ('fp32','half','int8','weight8'):
    comparisons += [('native_'+kind,'native_oldhalf')]
for kind in ('half','int8','weight8'):
    comparisons += [('native_'+kind,'native_fp32')]
for mode in ('fullcorr','fullwmse','headwmse'):
    scales=(.25,.5,1.) if mode=='fullcorr' else (1.,)
    for scale in scales:
        name=mode+'_scale'+str(scale)
        specs[name]={**base,**{f'w13base_{mode}_s{seed}':.03125*scale for seed in (3,4)}}
        comparisons += [(name,'native_oldhalf'),(name,'native_fp32')]
    for seed in (3,4):
        name=f'{mode}_s{seed}'
        specs[name]={**base,f'w13base_{mode}_s{seed}':.0625}
        comparisons += [(name,'native_oldhalf')]

def wp(s):
    w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
    return ((syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))).mean(-1)

def main():
    family=sys.argv[1] if len(sys.argv)>1 else 'all'
    assert family in ('all','complement')
    selected={k:v for k,v in specs.items() if family=='all' or k=='reference' or k.startswith('complement_')}
    pairs=[(a,b) for a,b in comparisons if a in selected and b in selected]
    suffix='' if family=='all' else '-'+family
    report_path=D/f'wave-report{suffix}.json'
    assert not report_path.exists()
    cf=np.load(R/'cache/chain_folds.npz')
    seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold'])
        if (s=='valid' and f!=0) or (s=='train'))
    hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy').tolist())
    names=list(selected);report={'specs':selected,'comparisons':pairs,'results':{},
        'scope':'Reused true-mask valid fold0; ALL train chains excluded because d256 teacher used all train. No train holdout claim. Same frozen two-stream mixture, organizer component replaced. Three predeclared fullcorr scales are exploratory and require correction for selection. Baseline FT uses valid fold !=0 only; baseline original pretraining provenance uncertain. Organizer provenance not certified. Conditional fitted-model intervals.'}
    for split in ('valid',):
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
                p=np.clip(sum(c*row[n] for n,c in sp.items()),-2,2)
                assert np.isfinite(p).all()
                moments[j,inv[ii]]+=np.stack([w.sum(0),(w*yy).sum(0),(w*p).sum(0),(w*yy*yy).sum(0),(w*p*p).sum(0),(w*yy*p).sum(0)],-1)
        score=wp(moments.sum(1));rng=np.random.default_rng(20261005)
        draws=np.array([wp(moments[:,rng.integers(len(chains),size=len(chains))].sum(1)) for _ in range(2000)])
        result={'sequences':len(ids),'chains':len(chains),'scores':dict(zip(names,map(float,score))),'comparisons':{}}
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b);dd=draws[:,ia]-draws[:,ib]
            result['comparisons'][a+' - '+b]={'delta':float(score[ia]-score[ib]),'conditional_ci95':np.quantile(dd,[.025,.975]).tolist()}
        report['results'][split+'_clean']=result
        np.savez_compressed(D/f'wave-moments{suffix}-{split}.npz',moments=moments,names=names,chains=chains)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
