from pathlib import Path
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave18'
specs={'w18final_family_full':{'w18final_ffull_s3':.5,'w18final_ffull_s4':.5},'w18final_family_raw':{'w18final_frawW_s3ft':.5,'w18final_frawW_s4ft':.5},'w18final_teacher':{'w18final_fbc1k_s5':.3,'w18final_ffull_s3':.1,'w18final_ffull_s4':.1,'w18final_frawW_s3ft':.125,'w18final_frawW_s4ft':.125,'orgbase':.0625}}
for split in ('train','valid'):
 for name,spec in specs.items():
  path=R/f'cache/pred_{name}_{split}.npy';assert not path.exists()
  arrays={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for n in spec};shape=next(iter(arrays.values())).shape
  tmp=Path(str(path)+'.partial');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape)
  for i in range(len(out)):
   out[i]=sum(c*np.asarray(arrays[n][i],np.float32) for n,c in spec.items());assert np.isfinite(out[i]).all()
  out.flush();del out;tmp.rename(path)
(D/'full-teacher-manifest.json').write_text(json.dumps({'specs':specs,'scope':'Final all-data teacher caches. Includes all validation labels in component training; NEVER report these as holdout metrics. Recipe analog to wave12 h256 successful fold0 experiment.'},indent=2))
