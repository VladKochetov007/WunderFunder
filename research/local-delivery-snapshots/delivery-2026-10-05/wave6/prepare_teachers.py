from pathlib import Path
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave6'
specs={'w6family_full':{'cmp_fullft_s3':.5,'cmp_fullft_s4':.5},'w6family_raw':{'cmp_rawwmse_s3':.5,'cmp_rawwmse65r2_s4':.5},
       'w6fullteacher':{'d256bc1k_f0':.3,'cmp_fullft_s3':.1,'cmp_fullft_s4':.1,'cmp_rawwmse_s3':.125,'cmp_rawwmse65r2_s4':.125,'orgbase':.0625}}
for split in ('train','valid'):
    for name,spec in specs.items():
        p=R/f'cache/pred_{name}_{split}.npy';assert not p.exists()
        arrays={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for n in spec}
        tmp=p.with_suffix('.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=next(iter(arrays.values())).shape)
        for i in range(len(out)):
            out[i]=sum(c*np.asarray(arrays[n][i],dtype=np.float32) for n,c in spec.items())
            assert np.isfinite(out[i]).all()
        out.flush();del out;tmp.rename(p)
(D/'teacher-manifest.json').write_text(json.dumps({'specs':specs,'scope':'KD teachers may be in-sample on student training rows. Evaluation must exclude ALL train chains because d256 teacher trained on all train. Organizer baseline provenance uncertain.'},indent=2))
print('DONE',flush=True)
