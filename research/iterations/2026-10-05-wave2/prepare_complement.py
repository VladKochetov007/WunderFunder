"""Teachers OOS on explicitly selected chains; keep all valid-fold0 chains out of student fit."""
from pathlib import Path
import hashlib
import json
import numpy as np

R=Path.home()/"WunderFunder-creative"
D=R/"research/iterations/2026-10-05-wave2"
def bucket(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest(),16)
def main():
    cf=np.load(R/"cache/chain_folds.npz")
    seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold'])
             if (s=='valid' and f!=0) or (s=='train' and bucket(f'half:{g}')%2==0))
    valid0=set(cf['chain'][(cf['split']=='valid')&(cf['fold']==0)])
    tr={int(g):int(c) for s,g,c in zip(cf['split'],cf['row_group'],cf['chain']) if s=='train'}
    eligible=[g for g,c in tr.items() if bucket(f'half:{g}')%2==1 and c not in seen and c not in valid0]
    fit=np.array([g for g in eligible if bucket(f'complement-wave2:{tr[g]}')%3!=0])
    hold=np.array([g for g in eligible if bucket(f'complement-wave2:{tr[g]}')%3==0])
    assert set(tr[g] for g in fit).isdisjoint(tr[g] for g in hold)
    assert set(tr[g] for g in fit).isdisjoint(seen|valid0)
    np.save(D/'complement_fit.npy',fit)
    np.save(D/'complement_hold.npy',hold)
    spec={'cmp_fullft_s3':.275,'cmp_fullft_s4':.275,'cmp_rawwmse_s3':.1,'cmp_rawwmse65r2_s4':.1,'orgbase':.0625}
    manifest={'fit_sequences':len(fit),'hold_sequences':len(hold),'fit_chains':len(set(tr[g] for g in fit)),
              'hold_chains':len(set(tr[g] for g in hold)),'spec':spec,
              'caveat':'Known composite-teacher training chains excluded; organizer baseline provenance not certified; all datasets adaptively reused.'}
    for split in ('train','valid'):
        path=R/f'cache/pred_w2_reference_{split}.npy'
        assert not path.exists(),path
        arrays={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for n in spec}
        shape=next(iter(arrays.values())).shape
        tmp=path.with_suffix('.partial.npy')
        out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape)
        for i in range(shape[0]):
            out[i]=sum(c*np.asarray(arrays[n][i],dtype=np.float32) for n,c in spec.items())
        out.flush();del out;tmp.rename(path)
    (D/'complement_split.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
