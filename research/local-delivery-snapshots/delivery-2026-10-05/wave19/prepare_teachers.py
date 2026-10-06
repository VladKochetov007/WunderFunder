from pathlib import Path
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave19'
specs={'w19family_twostream':{'cmp_tstream_s3':.5,'cmp_tstream_s4':.5},'w19teacher':{'d256bc1k_f0':.3,'cmp_fullft_s3':.1,'cmp_fullft_s4':.1,'cmp_tstream_s3':.125,'cmp_tstream_s4':.125,'orgbase':.0625}}
for split in ('train','valid'):
 for name,spec in specs.items():
  path=R/f'cache/pred_{name}_{split}.npy';assert not path.exists()
  arr={n:np.load(R/f'cache/pred_{n}_{split}.npy',mmap_mode='r') for n in spec};shape=next(iter(arr.values())).shape
  tmp=Path(str(path)+'.partial');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape)
  for i in range(len(out)):
   out[i]=sum(c*np.asarray(arr[n][i],np.float32) for n,c in spec.items());assert np.isfinite(out[i]).all(),(split,name,i)
  out.flush();del out;tmp.rename(path)
(D/'teacher-manifest.json').write_text(json.dumps({'specs':specs,'scope':'Stronger two-stream teacher replaces raw-only family in successful h256 distillation. Same original parent checkpoints, 2 epochs, optimizer and training groups as wave12. All-train teacher; evaluate only 425 valid fold0 chains excluding all train. No train-HO claim.'},indent=2))
print('DONE',flush=True)
