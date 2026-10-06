from pathlib import Path
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave24'
spec={'w15d256_s3':.3,'cmp_fullft_s3':.1,'cmp_fullft_s4':.1,'cmp_tstream_s3':.125,'cmp_tstream_s4':.125,'orgbase':.0625}
for split in ('train','valid'):
 path=R/f'cache/pred_w24comp_teacher_{split}.npy';assert not path.exists()
 arrays={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for n in spec};shape=next(iter(arrays.values())).shape
 tmp=Path(str(path)+'.partial');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape)
 for i in range(len(out)):
  out[i]=sum(c*np.asarray(arrays[n][i],np.float32) for n,c in spec.items());assert np.isfinite(out[i]).all()
 out.flush();del out;tmp.rename(path)
(D/'composite-teacher-manifest.json').write_text(json.dumps({'spec':spec,'scope':'Every known learned teacher parent restricted to train half0 and valid folds!=0. Original organizer pretraining provenance uncertain. Evaluate chain-clean 436 valid and fixed train diagnostic.'},indent=2))
