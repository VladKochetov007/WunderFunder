from pathlib import Path
import numpy as np
R=Path.home()/'WunderFunder-creative'
for split in ('train','valid'):
 path=R/f'cache/pred_w22teacher_{split}.npy';assert not path.exists()
 arrays=[np.load(R/f'cache/pred_w19distilled_s{s}_{split}.npy',mmap_mode='r') for s in (3,4)]
 tmp=Path(str(path)+'.partial');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=arrays[0].shape)
 for i in range(len(out)):
  out[i]=.5*(arrays[0][i]+arrays[1][i]);assert np.isfinite(out[i]).all()
 out.flush();del out;tmp.rename(path)
